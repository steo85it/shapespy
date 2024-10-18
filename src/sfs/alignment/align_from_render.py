import logging
import re
import shutil
import time
from functools import partial

import numpy as np
from scipy import stats
import xarray as xr
from tqdm import tqdm
from p_tqdm import p_umap
from datetime import datetime

from shadowspy.render_dem import render_match_image
from shadowspy.image_util import read_img_properties
from mesh_operations import mesh_generation, split_merged
from mesh_operations.mesh_utils import load_mesh
from mesh_operations.merge_overlapping import merge_inout
from sfs.config import SfsOpt as SfsOptClass
from asp.functions import image_align, bundle_adjust, \
    mapproject, dem_mosaic
import os
import pandas as pd
import glob
from rasterio.enums import Resampling

# retrieve options
from asp.functions import set_asp
from isis.functions import set_isis

# set options
crop = True

def get_dxy_for_tile(pdir, seldir, images, meas_path, comp_path, dem_path, coarse_factor=5):

    opt = SfsOptClass.get_instance()
    set_isis(opt.isisdir, opt.isisdata)
    set_asp(opt.aspdir)
    
    print(f"- Getting dxy for images.")
    # print(pdir, seldir, images, meas_path, comp_path, dem_path)
    if not os.path.exists(seldir):
        os.makedirs(seldir)
    sfs_ia_dir = f"{seldir}sfs_ia"
    os.makedirs(f"{sfs_ia_dir}", exist_ok=True)
        
    corr_list = {}
    nb_inliers_matches = {}
    problematic_imgs = []
    for img in (pbar := tqdm(images, total=len(images))):
        
        pbar.set_description(f"Processing {img} for dxyz to DEM")

        meas = glob.glob(f"{os.path.dirname(meas_path[0])}/{img}_map.tif")[0]

        try:
            comp = glob.glob(f"{os.path.dirname(comp_path[0])}/{img}_??????????????.tif")[0]
        except:
            logging.warning(f"- {img} rendering returned None. Skip.")
            continue

        # check that img (from selected list) is in paths
        assert img in meas, f"* {img} not found in {meas}. Stop."
        assert img in comp, f"* {img} not found in {comp}. Stop."

        ds_meas = xr.open_dataset(meas, engine="rasterio", mask_and_scale=False)
        # downsample both real and simulated images to help IA
        for v, vers in {'meas': meas, 'comp': comp}.items():
            ds = xr.open_dataset(vers, engine="rasterio", mask_and_scale=False)
            # coarsen the image
            dsc = ds.coarsen(x=coarse_factor, boundary="trim").mean(skipna=True). \
                    coarsen(y=coarse_factor, boundary="trim").mean(skipna=True)
            # then interpolate them to the same nodes as the original one
            dsc = dsc.rio.reproject_match(ds, resampling=Resampling.cubic_spline)

            # rendered geotiff needs some love...
            if v == 'comp':
                dsc = dsc.where(dsc.band_data > 0)
                dsc = dsc.where(dsc.band_data < ds.band_data.max())
                dsc = dsc.where(dsc.band_data != dsc.band_data.rio.nodata)
                dsc.band_data.rio.write_nodata(ds_meas.band_data.rio.nodata, encoded=True, inplace=True)

            # save and update variables
            out_ = f"{sfs_ia_dir}/{img}_{v}_coarse.tif"
            
            dsc.band_data.rio.to_raster(out_)
            if v == 'meas':
                meas = out_
            else:
                comp = out_
                
        if os.path.exists(f"{sfs_ia_dir}/run-transform.txt"):
            os.remove(f"{sfs_ia_dir}/run-transform.txt")
        if os.path.exists(f"{sfs_ia_dir}/run-ecef-transform.txt"):
            os.remove(f"{sfs_ia_dir}/run-ecef-transform.txt")

        try:
            image_align(pdir, images=[comp, meas], output_prefix=f"{sfs_ia_dir}/run",
                        inlier_threshold=100,
                        ip_per_image=1000000,
                        ecef_transform_type="translation",
                        dem1=dem_path,
                        dem2=dem_path,
                        o=f"{sfs_ia_dir}/run-align-intensity.tif",
                        stdout=f"{sfs_ia_dir}/log_ia_{img}.txt")

            # retrieve number of matches and inliers from IA log
            with open(f"{sfs_ia_dir}/log_ia_{img}.txt", "r") as file:
                for line in file:
                    if (re.search("inliers\.", line) and re.search("Found ", line)):
                        nb_inliers = line.split('/')[0].split('Found ')[-1].strip()
                        nb_matches = line.split('/')[1].split(' inliers.')[0].strip()
                        # print(img, nb_matches, nb_inliers)

            df = pd.read_csv(f"{sfs_ia_dir}/run-ecef-transform.txt", header=None, index_col=None)
            df = df[0].str.split('\s+', expand=True)
            df = df.astype('float')
            df.to_csv(f"{sfs_ia_dir}/run-ecef-transform.txt", header=None, index=None, sep=' ')
            # if ecef-transform is the identity matrix, something went wrong...
            assert np.max(np.abs(df.values - np.identity(4))) > 1.e-12

            xyz = pd.read_csv(f"{sfs_ia_dir}/run-ecef-transform.txt", header=None, sep='\s+').iloc[:3, -1].values
            corr_list[img] = xyz
            nb_inliers_matches[img] = [int(nb_inliers), int(nb_matches)]
        except:
            problematic_imgs.append(img)

    # if got into problematic images, list them and exit
    # check df for None: if len(None)>0, extract
    if len(problematic_imgs) > 0:
        print(f"- Could not find matches for {len(problematic_imgs)} images ({problematic_imgs}). Still ok.")

    # generate maxlit of a priori DEM (from subsampled rendered images)
    dem_mosaic(imgs=[f"{sfs_ia_dir}/{img}_comp_coarse.tif" for img in images], dirnam=seldir,
           max=None, output_prefix=f"max_lit_prioridem_.tif")

    df = pd.DataFrame(np.vstack(list(corr_list.values())), columns=['x', 'y', 'z'])
    df['3d'] = np.linalg.norm(df.values, axis=1)
    df['img'] = np.vstack(list(corr_list.keys()))
    df['nb_inliers'] = np.vstack(list(nb_inliers_matches.values()))[:, 0]
    df['nb_matches'] = np.vstack(list(nb_inliers_matches.values()))[:, 1]
    # do not consider images with <10 matches
    df = df.loc[df.nb_inliers >= 10]
    print(f"- We filtered {len(df.loc[df.nb_inliers < 10])} images with <10 inliers (IA unreliable).")
    print(df)

    # remove outliers with mad estimator
    for xyz in ['x', 'y', 'z']:
        sigma_mad = stats.median_abs_deviation(df.loc[:, xyz].values)
        df[f'keep_{xyz}'] = [np.abs(d - df.loc[:, xyz].median()) <= 3. * sigma_mad for d in df.loc[:, xyz].values]
    df = df.loc[df.keep_x & df.keep_y]
    print(f"- We filtered {len(corr_list) - len(df)} images with x or y corrections > 3*sigma_mad.")
    print(df)
    print("- Mean+-std / median hifts after iteration (apply medians)")
    print(f"dx = {df.x.mean()}+-{df.x.std()} / {df.x.median()}")
    print(f"dy = {df.y.mean()}+-{df.y.std()} / {df.y.median()}")
    print(f"dz = {df.z.mean()}+-{df.z.std()} / {df.z.median()}")
    df.to_csv(f"{sfs_ia_dir}/img_dxyz_nbmi.csv")

    return df


def prepare_stacked_meshes(indir, ext, tif_path, base_resolution,
                           fartopo_path=None, max_extension=50e3, Rb=1737.4e3,
                           lonlat0=(0, -90), rescale_fact=1.e-3):

    # prepare mesh of the input dem
    start = time.time()
    logging.info(f"- Computing trimesh for {tif_path}...")

    # regular delauney mesh
    meshpath = f"{indir}{tif_path.split('/')[-1].split('.')[0]}"
    mesh_generation.make(base_resolution, [1], tif_path, out_path=indir, mesh_ext=ext,
                         rescale_fact=rescale_fact, lonlat0=lonlat0)
    shutil.move(f"{indir}b{base_resolution}_dn1{ext}", f"{meshpath}{ext}")
    shutil.move(f"{indir}b{base_resolution}_dn1_st{ext}", f"{meshpath}_st{ext}")
    print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

    # if no far_topo is needed, just return mesh path for inner mesh
    if fartopo_path is None:
        return meshpath

    # crop fartopo to box around dem to render
    da_in = xr.open_dataarray(tif_path)
    bounds = da_in.rio.bounds()
    demcx, demcy = np.mean([bounds[0], bounds[2]]), np.mean([bounds[1], bounds[3]])

    da_out = xr.open_dataarray(fartopo_path)
    print(da_in)
    print(da_out)
    
    # verify consistency of crs
    da_out_crs = da_out.rio.crs
    match = re.search(r'SPHEROID\[.*Sphere",(\d+),', f"{da_out_crs}")
    if match:
        radius = float(match.group(1))
        if radius != Rb:
            print(f"wrong planet with radius={radius} while passing Rb={Rb}.")
            exit()
    else:
        print(f"weird crs... SPHEROID not found. {da_out_crs}.")
        exit()

    # set a couple of layers at 1, 5 and max_extension km ranges
    outer_topos = []

    # extres = {20e3: 20, 60e3: 60, 100e3: 120, 150e3: 240, 300e3: 480}
    extres = {20e3: 120, 60e3: 360} #, 100e3: 120, 150e3: 240, 300e3: 480}
    extres = {ext: max(res, base_resolution) for ext, res in extres.items() if ext < max_extension}
    for extension, resol in extres.items():
        da_red = da_out.rio.clip_box(minx=demcx-extension, miny=demcy-extension,
                             maxx=demcx+extension, maxy=demcy+extension)
        fartopo_path = f"{indir}LDEM_{extension}KM_outer.tif"
        fartopomesh = fartopo_path.split('.')[0]
        da_red.rio.to_raster(fartopo_path)
        outer_topos.append({resol: f"{indir}LDEM_{extension}KM_outer.tif"})

    start = time.time()
    # Merge inner and outer meshes seamlessly
    print(outer_topos)
    # for iter 0, set inner mesh as stacked mesh
    shutil.copy(f"{meshpath}_st{ext}", f"{indir}stacked_st{ext}")
    labels_dict_list = {}
    for idx, resol_dempath in enumerate(outer_topos):
        resol = list(resol_dempath.keys())[0]
        dempath = list(resol_dempath.values())[0]

        outer_mesh_resolution = resol
        fartopo_path = dempath
        fartopomesh = fartopo_path.split('.')[0]

        print(f"- Adding {fartopo_path} ({fartopomesh}) at {outer_mesh_resolution}mpp.")

        # ... and outer topography
        mesh_generation.make(outer_mesh_resolution, [1], fartopo_path, out_path=indir, mesh_ext=ext,
                             rescale_fact=1e-3)
        shutil.move(f"{indir}b{outer_mesh_resolution}_dn1_st{ext}", f"{fartopomesh}_st{ext}")
        print(f"- Meshes generated after {round(time.time() - start, 2)} seconds.")

        stacked_mesh_path = f"{indir}stacked_st{ext}"
        input_totalmesh, labels_dict = merge_inout(load_mesh(stacked_mesh_path),
                                                   load_mesh(f"{fartopomesh}_st{ext}"),
                                                   output_path=stacked_mesh_path) #, debug=True)
        labels_dict_list[idx] = labels_dict
        print(f"- Meshes merged after {round(time.time() - start, 2)} seconds and saved to {stacked_mesh_path}.")

    start = time.time()
    # Split inner and outer meshes
    len_inner_faces = labels_dict_list[0]['inner']
    inner_mesh_path, outer_mesh_path = split_merged.split_merged(input_totalmesh, len_inner_faces, meshpath,
                                                                 fartopomesh, ext, Rb)
    # import pyvista as pv
    # grid = pv.read(meshpath_st_hr)
    # grid.plot(show_scalar_bar=True, show_axes=True)
    
    print(f"- Inner+outer meshes generated from merged after {round(time.time() - start, 2)} seconds.")
    return inner_mesh_path, outer_mesh_path # it's actually the total mesh


def render_adjusted(mapproj_path, pdir, dem_path, indir='',
                    base_resolution=2, point=True, outer_topo=False):
    """

    :param mapproj_path: list to mapproj images to simulate
    :param pdir: str
    :param dem_path: str, input DEM geotiff for rendering
    :param base_resolution: int
    :param point: bool, model Sun as point or extended
    :return: list(str)
    """

    SfsOpt = SfsOptClass.get_instance()
    
    # compute direct flux from the Sun
    img_names = [x.split('/')[-1].split('_map')[0] for x in mapproj_path]

    # Elevation/DEM GTiff input
    if indir == '':
        indir = f"{pdir}render/"
    os.makedirs(indir, exist_ok=True)
    # outdir = f"{pdir}render/out/"
    ext = '.vtk'

    if outer_topo:
        inner_mesh_path, outer_mesh_path = prepare_stacked_meshes(indir, ext, dem_path, base_resolution,
                                           fartopo_path=SfsOpt.get("prioridem_full"), lonlat0=(0, -90), max_extension=100e3, )
        outer_mesh_path += ext
    else:
        inner_mesh_path = prepare_stacked_meshes(indir, ext, dem_path, base_resolution, )
        outer_mesh_path = None

    # open index
    try:
        lnac_index = SfsOpt.get("rootdir") + SfsOpt.get("pds_index_name") + ".parquet"
        cumindex = pd.read_parquet(lnac_index)  
    except:
        lnac_index = SfsOpt.get("rootdir") + SfsOpt.get("pds_index_name") + ".pkl"
        cumindex = pd.read_pickle(lnac_index)

    # get list of images from mapprojected folder
    imgs_nam_epo_path = read_img_properties(img_names, cumindex)

    # imgs_nam_epo_path['meas_path'] = [f"{sfs_ia_dir}/run-{img}-meas-intensity.tif" # should point to prj/ba
    #imgs_nam_epo_path['meas_path'] = mapproj_path
    imgs_nam_epo_path['meas_path'] = [f"{os.path.dirname(mapproj_path[0])}/{img}_map.tif" for img in imgs_nam_epo_path.iloc[:,0].values]
    logging.info(f"- {len(imgs_nam_epo_path)} images found in path. Rendering input DEM.")

    outrasters = []
    for idx, row in (pbar := tqdm(imgs_nam_epo_path.iterrows(), total=len(imgs_nam_epo_path))):

        search_output = glob.glob(f"{indir}out/{row[0]}_*.tif")
        if len(search_output) > 0:
            logging.info(f"# Image {row[0]} already rendered to {search_output}. Skip.")
            out = search_output[0]
            outrasters.append(out)
            continue

        # adapt mdis epoch to shadowspy expected format
        if SfsOpt.calibrate[:4] == 'mdis':
            convert_date = datetime.strptime(row[1].strip(), '%Y-%m-%dT%H:%M:%S.%f')
            format_code = '%Y-%m-%d %H:%M:%S.%f'
            row[1] = convert_date.strftime(format_code)

        pbar.set_description(f"Processing {row[0].strip()}  and clipping to {row[2]}...")
        try:
            out = render_match_image(indir, meshes={'stereo': f"{inner_mesh_path}_st{ext}",
                                                'cart': f"{inner_mesh_path}{ext}"},
                       path_to_furnsh="/explore/nobackup/projects/pgda/LRO/data/furnsh/furnsh.LRO.def.spkonly.LOLA",
                       img_name=row[0], epo_utc=row[1], meas_path=row[2], basemesh=outer_mesh_path, point=point)
        except:
            logging.warning(f"- render_match_image failed for image {row[0]}. Continue.")
            out = None
            
        outrasters.append(out)
            
    return outrasters


# update .adjust file with transformation from alignment to dem
def align_img(img, tile, sel, seldir, prioridem_path): #, prj_img_path):
    
    SfsOpt = SfsOptClass.get_instance()
    
    procdir = f"{SfsOpt.procroot}tile_{tile}/"
    seldir = f"{procdir}sel_{sel}/"
    out_folder = "ba_align" # prj_img_path.split('/')[-2]
    # prj_img_out = f"{prj_img_path}{img}_map.tif"

    #adjustfil = f"{seldir}{out_folder}/run-{img}.adjust"
    #if os.path.exists(prj_img_out) and os.path.exists(adjustfil):
    #    logging.info(f"- {prj_img_out} and {adjustfil} already exist. Skip.")
    #    return
    
    os.makedirs(f"{seldir}ba/", exist_ok=True)
    os.makedirs(f"{seldir}{out_folder}/", exist_ok=True)
    shutil.copy(f"{procdir}ba/run-{img}.adjust", f"{seldir}ba/run-{img}.adjust")
    if os.path.islink(f"{procdir}{img}.cub"):
        os.remove(f"{procdir}{img}.cub")
    os.symlink(f"{procdir}{img}.cal.echo.cub", f"{procdir}{img}.cub")

    bundle_adjust([f"{img}.cub"],
                  prioridem_path,
                  dirnam=procdir,
                  num_passes=1,
                  apply_initial_transform_only=None,
                  input_adjustments_prefix=f"{seldir}ba/run",
                  output_prefix=f"{seldir}{out_folder}/run",
                  parallel=False,
                  use_csm=SfsOpt.use_csm,
                  initial_transform=f"{seldir}{out_folder}/run-ecef-transform.txt",
                  stdout=f"{seldir}{out_folder}/tmp_ba_align_{img}.log")

    
def project_img(img, tile, sel, seldir, prioridem_path, ba_prefix, prj_img_path):
    SfsOpt = SfsOptClass.get_instance()

    procdir = f"{SfsOpt.procroot}tile_{tile}/"
    seldir = f"{procdir}sel_{sel}/"
    # out_folder = prj_img_path.split('/')[-2]
    prj_img_out = f"{prj_img_path}{img}_map.tif"

    if os.path.exists(prj_img_out):
        logging.info(f"- {prj_img_out} already exists. Skip.")
        return

    os.makedirs(prj_img_path, exist_ok=True)
    mapproject(from_=f"{procdir}{img}.cub", to=prj_img_out,
                bundle_adjust_prefix=f"sel_{sel}/{ba_prefix}/run",
                dem=prioridem_path, dirnam=procdir,
                use_csm=SfsOpt.use_csm,
                stdout=f"{prj_img_path}{img}_mapproj.log")

    
def align_from_render(tile, sel, selected_images,
                      use_existing_transform=None,
                      new_rendering=True, base_resolution=None, point=True):

    SfsOpt = SfsOptClass.get_instance()
    set_isis(SfsOpt.isisdir, SfsOpt.isisdata)
    set_asp(SfsOpt.aspdir)

    procdir = f"{SfsOpt.procroot}tile_{tile}/"
    seldir = f"{procdir}sel_{sel}/"

    os.makedirs(f"{seldir}sfs_ia", exist_ok=True)

    if base_resolution is None:
        base_resolution = SfsOpt.get('targetmpp') * 2.
        assert base_resolution > 0, "* base_resolution must be greater than 0. Exit."

    # reset ba_align dir
    if os.path.exists(f"{seldir}ba_align"):
        shutil.rmtree(f"{seldir}ba_align")
    os.makedirs(f"{seldir}ba_align")

    prioridem_path = f"{procdir}ldem_{tile}.tif"

    # estimate a shift to align each image
    # set paths to computed/simulated and measured/real images
    meas_path = [f"{procdir}prj/ba/{img}_map.tif" for img in selected_images]

    if use_existing_transform is None:
        # get simulated images
        if new_rendering:
            comp_path = render_adjusted(meas_path, pdir=seldir,
                                        dem_path=prioridem_path, base_resolution=base_resolution, point=point,
                                        outer_topo=False)
        else:
            logging.info(f"- Reading renderings...")
            comp_path = []
            for img in selected_images:
                try:
                    comp_path.append(glob.glob(f"{seldir}render/out/{img}_??????????????.tif")[0])
                except:
                    comp_path.append(None)

        # set path to dem (used for transformation to ecef)
        dxyz_df = get_dxy_for_tile(procdir, seldir, selected_images, meas_path, comp_path, prioridem_path)

        # compute medians, generate transform file and apply to ba/*.adjust files
        median_dx = np.nanmedian(dxyz_df.x.values)
        median_dy = np.nanmedian(dxyz_df.y.values)
        median_dz = np.nanmedian(dxyz_df.z.values)

        # copy medians to transform file
        df = pd.DataFrame(np.identity(4))
        dxyz = [median_dx, median_dy, median_dz]
        df.iloc[:-1, -1] = dxyz
        assert np.sum(np.diag(df)) - len(df) == 0
        df.to_csv(f"{seldir}ba_align/run-ecef-transform.txt", header=None, index=None, sep=' ')

    else:
        assert os.path.exists(use_existing_transform), f"{use_existing_transform} does not exist. Stop."
        shutil.copy(use_existing_transform, f"{seldir}ba_align/run-ecef-transform.txt")
        dxyz_df = None

    # apply to all images
    prj_img_path = f"{seldir}prj/ba_align/"
    os.makedirs(prj_img_path, exist_ok=True)

    for x in tqdm(selected_images, total=len(selected_images), desc='aligning images'):
        align_img(x, tile, sel, seldir, prioridem_path) #, prj_img_path)
    mapproj_ba_path = p_umap(partial(project_img, tile=tile, sel=sel, seldir=seldir,
                                     prioridem_path=prioridem_path, ba_prefix="ba_align", prj_img_path=prj_img_path),
                             selected_images, desc='mapproj ba_align', total=len(selected_images))

    # generate maxlit from registered selected images
    dem_mosaic(imgs=[f"{prj_img_path}/{img}_map.tif" for img in selected_images], dirnam=seldir,
           max=None, output_prefix=f"max_lit_ba_align_sel{sel}.tif")

    # check that all aligned images have been correctly mapprojected
    assert len(glob.glob(f"{prj_img_path}*_map.tif")) == len(selected_images), \
        f"* Got {len(glob.glob(f'{prj_img_path}*_map.tif'))} map_tif and {len(selected_images)} sel_img. Stop."

    return dxyz_df


if __name__ == '__main__':

    tileid = 6
    align_from_render(tileid)
