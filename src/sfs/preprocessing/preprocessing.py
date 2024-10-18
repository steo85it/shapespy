import glob
import re
from functools import partial

import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from tqdm import tqdm
from p_tqdm import p_umap
import xarray as xr
from rasterio.enums import Resampling

from asp.functions import gdal_translate, mapproject, dem_mosaic, bundle_adjust, set_asp, usgscsm_cam_test, cam_test
from asp.resample_state_model import scale_json_parameters
from isis.functions import calibrate_lrocnac, calibrate_mdis, reduce as isis_reduce, set_isis
from sfs.config import SfsOpt as SfsOptClass
from asp.gen_csm import gen_csm_line, gen_csm_frame

from sfs.processing.sfs_pipeline_tools import get_tile_bounds

logging.basicConfig(level=logging.INFO)

# aspdir = SfsOpt.get("aspdir")
# isisdir = SfsOpt.get("isisdir")
# isisdata = SfsOpt.get("isisdata")
#
# set_asp(aspdir)
# set_isis(isisdir, isisdata)
num_threads = 1

####################
def load_project_img(idxrow, tileid, prioridem_path, bundle_adjust_prefix=None): #, isisdir=isisdir, isisdata=isisdata, aspdir=aspdir):

    SfsOpt = SfsOptClass.get_instance()

    # for index, row in cumindex.iterrows():
    idx, row = idxrow
    img = row.img_name
    procdir = f"{SfsOpt.procroot}tile_{tileid}/"

    # set_asp(aspdir)
    # set_isis(isisdir, isisdata)
    # SfsOpt.from_json(f"{procdir}config.json")

    print(f"- Processing {img}.IMG...")

    if SfsOpt.resample_images:
        print(f"- Using full res csm cameras.")
        if os.path.isfile(f"{procdir}{img}.cub"):
            os.remove(f"{procdir}{img}.cub")
        os.symlink(f"{procdir}{img}.cal.echo.cub", f"{procdir}{img}.cub")

        if SfsOpt.use_csm:

            assert os.path.isfile(f"{procdir}{img}.model_state.json"), f"* {procdir}{img}.model_state.json not found. Exit."

            if os.path.isfile(f"{procdir}{img}.json") or os.path.islink(f"{procdir}{img}.json"):
                os.remove(f"{procdir}{img}.json")
            os.symlink(f"{procdir}{img}.model_state.json", f"{procdir}{img}.json")
            
    if bundle_adjust_prefix == None:
        #prj_img_path = f"{procdir}prj/{SfsOpt.targetmpp}mpp/"
        prj_img_path = f"{procdir}prj/orig/"
        if not os.path.isdir(prj_img_path):
            os.makedirs(prj_img_path, exist_ok=True)
        prj_img_path += f"{img}_map.tif"

        if not os.path.exists(f"{prj_img_path}"):
            mapproject(from_=f"{procdir}{img}.cub", to=f"{prj_img_path}",
                       dem=prioridem_path, dirnam=procdir,
                       threads=num_threads, use_csm=SfsOpt.use_csm,
                       stdout=f"{prj_img_path}tmp_{img}.log")
        else:
            logging.warning(f"# {prj_img_path} already exists. Skip.")
    else:
        prj_img_path = f"{procdir}prj/{bundle_adjust_prefix}/"
        os.makedirs(prj_img_path, exist_ok=True)
        prj_img_path += f"{img}_map.tif"

        if os.path.exists(prj_img_path):
            if time.ctime(os.path.getmtime(prj_img_path)) < time.ctime(os.path.getmtime(f"{procdir}{bundle_adjust_prefix}/run-{img}.adjust")):
                logging.warning(f"# Found {prj_img_path} older than its .adjust counterpart. Remove and reproject.")
                os.remove(prj_img_path)
                                                                       
            
        if not os.path.exists(prj_img_path):
            mapproject(from_=f"{procdir}{img}.cub", to=prj_img_path,
                       bundle_adjust_prefix=f"{procdir}{bundle_adjust_prefix}/run",
                       dem=prioridem_path, dirnam=procdir,
                       threads=num_threads, use_csm=SfsOpt.use_csm,
                       stdout=f"{procdir}prj/{bundle_adjust_prefix}/tmp_{img}.log")
        else:
            logging.warning(f"# {prj_img_path} already exists. Skip.")

    # check if mapproj image has been correctly generated
    if not os.path.exists(prj_img_path):
        print(f"** Issue with tile_{tileid}:{img} projection. Check {prj_img_path} and relaunch.")
        #exit()

    return prj_img_path


##################
def load_calibrate_project_img(idxrow, tileid, prioridem_path, bundle_adjust_prefix=None, project_imgs=True):

    SfsOpt = SfsOptClass.get_instance()

    idx, row = idxrow
    img = row.img_name
    procdir = f"{SfsOpt.procroot}tile_{tileid}/"
    print(procdir)

    
    set_asp(SfsOpt.aspdir)
    set_isis(SfsOpt.isisdir, SfsOpt.isisdata)

    logging.info(f"- Processing {img}.IMG...")

    # link .IMG file in datadir to procdir
    if os.path.islink(f"{procdir}{img}.IMG") or os.path.exists(f"{procdir}{img}.IMG"):
        os.remove(f"{procdir}{img}.IMG")
    os.symlink(f"{SfsOpt.datadir}{img}.IMG", f"{procdir}{img}.IMG")

    if (not SfsOpt.use_csm) and os.path.exists(
            f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub"):
        
        print(f"{img} already calibrated in USGS ISIS. Skip.")

    elif SfsOpt.use_csm and (not SfsOpt.resample_images) and os.path.exists(
            f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub") and os.path.exists(
            f"{procdir}{img}.model_state.json"):

        # update links as needed
        if os.path.islink(f"{procdir}{img}.cub"):
            os.remove(f"{procdir}{img}.cub")
        os.symlink(f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub", f"{procdir}{img}.cub")
        
        print(f"{img} already calibrated in USGS ISIS (including CSM). Skip.")

    elif SfsOpt.use_csm and SfsOpt.resample_images and os.path.exists(
            f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub") and os.path.exists(
            f"{procdir}{img}.model_state_red{SfsOpt.targetmpp}.json"):

        # update links as needed
        if os.path.islink(f"{procdir}{img}.json"):
            os.remove(f"{procdir}{img}.json")
        os.symlink(f"{procdir}{img}.model_state_red{SfsOpt.targetmpp}.json", f"{procdir}{img}.json")
        
        print(f"{img} already calibrated in USGS ISIS (including resampled CSM). Skip.")
        
    else: # if not all expected cub/json exist, then generate them

        if not os.path.exists(f"{procdir}{img}.cal.echo.cub"):
            if SfsOpt.get('calibrate') == 'lrocnac':
                calibrate_lrocnac(img, dirnam=procdir)
            elif SfsOpt.get('calibrate')[:4] == 'mdis':
                calibrate_mdis(img, dirnam=procdir)
                # for compatibility
                if os.path.exists(f"{procdir}{img}.cal.echo.cub"):
                    os.remove(f"{procdir}{img}.cal.echo.cub")
                os.symlink(f"{procdir}{img}.cal.cub", f"{procdir}{img}.cal.echo.cub")
            else:
                print("** Choose lrocnac or mdis*ac as SfsOpt('calibrate') option.")
                exit()
        else:
            print(f"{img} already calibrated for USGS ISIS. Linking only.")

        # check that calibrated camera exists
        assert os.path.exists(f"{procdir}{img}.cal.echo.cub")

        if SfsOpt.get('resample_images'):
            MPP = row['RESOLUTION']
            redfact = max(1, SfsOpt.targetmpp // MPP + 1) #int(round(SfsOpt.targetmpp / MPP, 0)))
            logging.info(f"redfact img float/int/trunc/used: {img},"
                         f" {SfsOpt.targetmpp / MPP}, {int(round(SfsOpt.targetmpp / MPP, 0))}, {int(SfsOpt.targetmpp / MPP)}, {redfact}")

        if SfsOpt.use_csm:

            if SfsOpt.get('resample_images'):

                logging.info("- Producing reduced isis camera.")
                isis_reduce(from_=f"{img}.cal.echo.cub", to=f"{img}.cal.echo.red{SfsOpt.targetmpp}.cub",
                        algorithm="average",
                        dirnam=procdir, sscale=redfact, lscale=redfact)
            else:
                # just for compatibility
                if os.path.isfile(f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub"):
                    os.remove(f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub")
                os.symlink(f"{procdir}{img}.cal.echo.cub", f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub")


            if SfsOpt.calibrate in ['lrocnac', 'mdisnac']:
                gen_csm_line(f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub")  # if ALE is not installed properly, will break here
            else:
                logging.error(f'** No CSM option set for camera {SfsOpt.calibrate}. Check and relaunch.')
                exit()

            if SfsOpt.get('resample_images'):

                logging.info("- Producing reduced csm/model_state camera.")
                usgscsm_cam_test(input_camera_model=f"{procdir}{img}.json",
                                 output_model_state=f"{procdir}{img}.model_state.json",
                                 dirnam=procdir)
                # test csm camera
                # cam_test(procdir, img, cams=[f"{img}.cal.echo.cub", f"{img}.model_state.json"], sample_rate=100)
                scale_json_parameters(input_path=f"{procdir}{img}.model_state.json",
                                      output_path=f"{procdir}{img}.model_state_red{SfsOpt.targetmpp}.json",
                                      scale_factor=redfact)

                if os.path.isfile(f"{procdir}{img}.json"):
                    os.remove(f"{procdir}{img}.json")
                os.symlink(f"{procdir}{img}.model_state_red{SfsOpt.targetmpp}.json", f"{procdir}{img}.json")


        elif SfsOpt.get('resample_images'):

            isis_reduce(from_=f"{img}.cal.echo.cub", to=f"{img}.cal.echo.red{SfsOpt.targetmpp}.cub",
                        algorithm="average",
                        dirnam=procdir, sscale=redfact, lscale=redfact)

        else:
            # just for compatibility
            if os.path.isfile(f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub"):
                os.remove(f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub")
            os.symlink(f"{procdir}{img}.cal.echo.cub", f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub")


    # check that calibrated cameras exist for image
    assert os.path.isfile(f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub")

    # link camera with the requested resolution to img.cub file
    if os.path.exists(f"{procdir}{img}.cub") or os.path.islink(f"{procdir}{img}.cub"):
        os.remove(f"{procdir}{img}.cub")
    os.symlink(f"{procdir}{img}.cal.echo.red{SfsOpt.targetmpp}.cub", f"{procdir}{img}.cub")

    if project_imgs:
        # load project image
        prj_img_path = load_project_img(idxrow, tileid, prioridem_path=prioridem_path,
                                    bundle_adjust_prefix=bundle_adjust_prefix)
                                    # isisdir=isisdir, isisdata=isisdata, aspdir=aspdir)

        return prj_img_path

    else:
        return f"{procdir}{img}.cub"


def ba_and_mapproj(tileid, cumindex, prioridem_path, bundle_adjust_prefix,
                   input_adjustments_prefix=None, clean_match_files_prefix=None,
                   entry_point=0, stop_point=3,
                   parallel=True, use_mapproject=True):

    cwd = os.getcwd()
    
    SfsOpt = SfsOptClass.get_instance()
    set_asp(SfsOpt.aspdir)

    procdir = f"{SfsOpt.procroot}tile_{tileid}/"

    input_imgs = [x for x in cumindex.loc[:, 'img_name'].values]
    print(input_imgs)

    # Load configuration
    with open(SfsOpt.config_ba_path, 'r') as file:
        config = yaml.safe_load(file)

    if not os.path.exists(SfsOpt.nodes_list):
        logging.error(f"- File {SfsOpt.nodes_list} expected by bundle_adjust but not present. Exit.")
        exit()

    # Prepare arguments for bundle_adjust
    ba_kwargs = {
        "imglist": [f"{x}.cub" for x in input_imgs[:]],
        "dem": prioridem_path, #SfsOpt.prioridem_full,
        "heights_from_dem": prioridem_path, #SfsOpt.prioridem_full,
        "dirnam": procdir,
        "entry_point": entry_point,
        "stop_point": stop_point,
        "input_adjustments_prefix": input_adjustments_prefix,
        "o": f"{bundle_adjust_prefix}/run",
        "clean_match_files_prefix": clean_match_files_prefix,
        # "nodes_list": f"{os.getcwd()}/{SfsOpt.nodes_list}",
        "nodes_list": SfsOpt.nodes_list,
        "parallel": parallel,
        "mapproj_dem": prioridem_path, #SfsOpt.prioridem_full
        "use_mapproject": use_mapproject,
    }

    # Remove any duplicates from ba_kwargs that are already set
    config = {k: v for k, v in config.items() if
                 k not in ['imglist', 'dem', 'heights_from_dem', 'dirnam', 'entry_point', 'stop_point',
                           'input_adjustments_prefix', 'o', 'nodes_list', 'mapproj_dem']}

    # Add any additional parameters from config
    ba_kwargs.update(config)

    print(ba_kwargs)
    # Call bundle_adjust with **kwargs
    bundle_adjust(**ba_kwargs)

    # if only matches are generated, stop here (no adjust files, would crash)
    if stop_point == 2:
        logging.info("- Finished matching.")
        return

    # check that all adjust files have been produced
    for img in input_imgs:
        assert os.path.exists(f"{procdir}{bundle_adjust_prefix}/run-{img}.adjust"), \
            f"{procdir}{bundle_adjust_prefix}/run-{img}.adjust does not exist. Exit."

    # go back to original dir
    # assert os.getcwd() == cwd, f"Current working directory {os.getcwd()} is not the same as initial one {cwd}"
    os.chdir(cwd)

    return

        
def mapproj_maxlit(tileid, cumindex, prioridem_path, bundle_adjust_prefix,
                   input_adjustments_prefix=None, parallel=True, target_resolution=None):

    cwd = os.getcwd()
    
    SfsOpt = SfsOptClass.get_instance()
    set_asp(SfsOpt.aspdir)

    if target_resolution is None:
        target_resolution = SfsOpt.get('targetmpp')
    
    procdir = f"{SfsOpt.procroot}tile_{tileid}/"

    input_imgs = [x for x in cumindex.loc[:, 'img_name'].values]
    print(input_imgs)
    
    # project adjusted images
    if parallel:
        mapproj_ba_path = p_umap(partial(load_project_img, tileid=tileid,
                                         prioridem_path=prioridem_path, bundle_adjust_prefix=bundle_adjust_prefix),
                                 cumindex.iterrows(), total=len(cumindex))
    else:
        mapproj_ba_path = []
        for idxrow in cumindex.iterrows():
            mapproj_ba_path.append(
                load_project_img(idxrow, tileid, prioridem_path=prioridem_path, bundle_adjust_prefix=bundle_adjust_prefix))

    # check if all selected images have been pre-processed
    assert len([f"{procdir}{x}.IMG" for x in input_imgs]) == len(
        [f"{procdir}prj/{bundle_adjust_prefix}/{x}_map.tif" for x in input_imgs])

    # and produce max_lit mosaic
    if not os.path.exists(f"{procdir}max_lit_{bundle_adjust_prefix}_{tileid}.tif"):
        dem_mosaic(imgs=[f"{procdir}prj/{bundle_adjust_prefix}/{x}_map.tif" for x in cumindex.img_name.values],
                   dirnam=procdir, tr=target_resolution,
                   max=None, output_prefix=f"max_lit_{bundle_adjust_prefix}_{tileid}.tif")

    # go back to original dir
    # assert os.getcwd() == cwd, f"Current working directory {os.getcwd()} is not the same as initial one {cwd}"
    os.chdir(cwd)

    return glob.glob(f"{procdir}prj/{bundle_adjust_prefix}/*_map.tif")

    
    
def clip_ldem(tileid, rescale_by=None):
    """Clips the a priori DEM according to the input shp file bounds. Brings the DEM
    to desired resolution and rescales if necessary.
    Args:
        tileid (int): number of tile within working site
        rescale_by (bool): whether to rescale (if MB's DEM), default is None
    Returns:
        prioridem_path (str): path of new clipped a priori DEM
    """
    SfsOpt = SfsOptClass.get_instance()
    set_asp(SfsOpt.aspdir)

    procdir = f"{SfsOpt.procroot}tile_{tileid}/"
    
    # get tile bounds and extend
    input_shp_km_to_dem_crs = gpd.read_file(SfsOpt.input_shp)
    
    # Import and clip a priori DEM
    #if SfsOpt.local:
    #    prioridem = xr.open_dataset(SfsOpt.prioridem_full)
    #    demvar = [x for x in prioridem.data_vars][0]
    #    prioridem = prioridem.rio.clip_box(minx=minx, miny=miny,
    #                                       maxx=maxx, maxy=maxy)
    #else:
    prioridem = xr.open_dataset(SfsOpt.prioridem_full,
                                mask_and_scale=False)

    print(input_shp_km_to_dem_crs)
    input_shp_km_to_dem_crs = input_shp_km_to_dem_crs.to_crs(prioridem.rio.crs)
    print(input_shp_km_to_dem_crs)
    input_shp_km_to_dem_crs.to_file(f"{procdir}tmp_input_shp.shp", crs=prioridem.rio.crs)
    minx, miny, maxx, maxy = get_tile_bounds(f"{procdir}tmp_input_shp.shp", tileid, extend=0.1)
    print(minx, miny, maxx, maxy)
    
    # crs and min/max x/y need to be consistent
    print(prioridem)
    print(prioridem.rio.crs)
    print(SfsOpt.crs_stereo_meters)
    demvar = [x for x in prioridem.data_vars][0]
    prioridem = prioridem.rio.clip_box(minx=minx, miny=miny,
                                       maxx=maxx, maxy=maxy)
    
    # this should be better, acting only on the clipped DEM.
    prioridem = prioridem.rio.reproject(SfsOpt.crs_stereo_meters,
                                        resampling=Resampling.cubic_spline,
                                        resolution=SfsOpt.targetmpp)
    print(prioridem.rio.crs)
        
    if rescale_by != None:
        # needed for MB's dems (ASP doesn't recognize the "scale" property)
        # don't use with "DM2_final_adj_5mpp_surf.tif" (but do with other "surf")
        print(f"# applying rescaling to MB's dems")
        prioridem[demvar].values *= rescale_by
        prioridem[demvar].rio.set_attrs({'scale_factor': 1}, inplace=True)

    print(f"- Using {SfsOpt.prioridem_full} as prior.")

    # !! This assumes that the input crs is equivalent to this newer standard!!
    # No reprojection applied!!
    prioridem.rio.write_crs(SfsOpt.get("crs_stereo_meters"), inplace=True)
    prioridem[demvar].rio.to_raster(f"{procdir}ldem_{tileid}_orig.tif")

    prioridem_path = f"{procdir}ldem_{tileid}.tif"
    # Bring input DEM to desired resolution
    gdal_translate(procdir, filin=f"{procdir}ldem_{tileid}_orig.tif",
                   filout=prioridem_path,
                   tr=f"{SfsOpt.targetmpp} {SfsOpt.targetmpp}", r='cubicspline')
    return prioridem_path


def preprocess(tileid, cumindex, with_ba=True, parallel=True, rescale_by=None):

    SfsOpt = SfsOptClass.get_instance()
    set_asp(SfsOpt.aspdir)

    procdir = f"{SfsOpt.procroot}tile_{tileid}/"

    # clean up symlinks (done in img calib loop)
    #for f in glob.glob(f"{procdir}*.IMG"):
    #    os.remove(f)
    #for f in glob.glob(f"{procdir}*.cub"):
    #    if os.path.islink(f):
    #        os.remove(f)
    
    # sort by sun longitude
    cumindex = cumindex.sort_values(by='SUB_SOLAR_LONGITUDE').reset_index() #.loc[:100]
    # extract name of images
    input_imgs = [x for x in cumindex.loc[:, 'img_name'].values]

    # clip the a priori dem, bring dem to desired resolution, and rescale if necessary
    prioridem_path = clip_ldem(tileid, rescale_by=rescale_by)
    
    #  Produce calibrated cubes from all selected images, reduce resolution and mapproject
    if parallel:
        p_umap(partial(load_calibrate_project_img, tileid=tileid, prioridem_path=prioridem_path), cumindex.iterrows(),
               total=len(cumindex), desc="load_calibrate_project_img")
    else:
        for idxrow in tqdm(cumindex.iterrows(), total=len(cumindex), desc="load_calibrate_project_img"):
            load_calibrate_project_img(tileid=tileid, prioridem_path=prioridem_path, idxrow=idxrow)
        
    # check if all selected images have been pre-processed
    assert len([f"{procdir}{x}.IMG" for x in input_imgs]) == len([f"{procdir}{x}.cub" for x in input_imgs])

    #print("exiting after calib")
    #exit()
    
    # produce initial maxlit
    if not os.path.exists(f"{procdir}max_lit_{tileid}.tif"):
        dem_mosaic(imgs=[f"{procdir}prj/orig/{x}_map.tif" for x in input_imgs], dirnam=procdir, max=None,
                   tr=SfsOpt.get('targetmpp'), output_prefix=f"max_lit_{tileid}.tif")

    #print("exit after maxlit")
    #exit()
        
    # remove older links in prj folder
    old_links_prj = glob.glob(f"{procdir}prj/*_map.tif") 
    for fil in old_links_prj:
        if os.path.islink(fil):
            os.remove(fil)
        else:
            print(f"why is {fil} here?")
            
    # link camera with the requested resolution to img.cub file
    full_shadowed_images = []
    for idx, row in cumindex.iterrows():
        img = row.img_name
        if os.path.exists(f"{procdir}{img}.cub"):
            os.remove(f"{procdir}{img}.cub")
        os.symlink(f"{procdir}{img}.cal.echo.cub", f"{procdir}{img}.cub")

        #gen_csm(f"{procdir}{img}.cal.echo.cub")
        if SfsOpt.use_csm:
            assert os.path.exists(f"{procdir}{img}.json"), f"{procdir}{img}.json not found. Exit."    
        assert os.path.exists(f"{procdir}prj/orig/{img}_map.tif")
        
        # are these links useful?
        if os.path.islink(f"{procdir}prj/{img}_map.tif"):
            os.remove(f"{procdir}prj/{img}_map.tif")
        os.symlink(f"{procdir}prj/orig/{img}_map.tif", f"{procdir}prj/{img}_map.tif")

        # check that the image is not fully shadowed
        map_path = f"{procdir}prj/orig/{img}_map.tif"
        ds = xr.open_dataset(map_path, engine="rasterio")
        ds = ds.coarsen(x=20, boundary="trim").mean(). \
            coarsen(y=20, boundary="trim").mean()
        # remove shadows and add to "shadowed" list if <5% illuminated
        img_pix = ds.band_data.values.ravel()
        img_pix = img_pix[~np.isnan(img_pix)]
        img_pix_ill = img_pix[img_pix > SfsOpt.shadow_threshold]
        if len(img_pix) > 0:
            full_shadow = len(img_pix_ill) / len(img_pix) < 0.1 #changed from 0.05
            full_shadow = len(img_pix_ill) / len(img_pix) < 0.1 #changed from 0.05
        else:
            full_shadow = True

        if full_shadow:
            full_shadowed_images.append(img)

    # remove fully shadowed images from selected list
    logging.info(f"- {len(full_shadowed_images)} images fully shadowed and removed.")
    cumindex = cumindex[~cumindex['img_name'].isin(full_shadowed_images)]

    if with_ba:
        # print("CAUTION!! Starting after matching")
        ba_and_mapproj(tileid, cumindex, prioridem_path, bundle_adjust_prefix='ba_iter0', parallel=parallel) #, entry_point=2)

    return cumindex


def preprocess_mdis(mdis_index_df, geojson_file_path, nacwac='NAC', latrange=None):

    if latrange is None:
        latrange = [-90, -60]

    # select useful columns
    mdis_index_df = mdis_index_df[['VOLUME_ID', 'PATH_NAME', 'FILE_NAME', 'HORIZONTAL_PIXEL_SCALE', 'SUB_SOLAR_LONGITUDE',
                                   'EMISSION_ANGLE', 'INCIDENCE_ANGLE', 'EXPOSURE_DURATION', 'STANDARD_DEVIATION',
                                   'CENTER_LATITUDE', 'CENTER_LONGITUDE',
                                   'SATURATED_PIXEL_COUNT', 'MISSING_PIXELS', 'START_TIME', 'DATA_QUALITY_ID']]

    # add useful columns
    mdis_index_df.loc[:, 'img_name'] = [s.strip().split('.')[0] for s in mdis_index_df['FILE_NAME'].values]
    mdis_index_df.loc[:, 'SLEW_ANGLE'] = mdis_index_df.loc[:, 'EMISSION_ANGLE'].abs().values
    mdis_index_df['FILE_SPECIFICATION_NAME'] = mdis_index_df['VOLUME_ID'].str.strip() + "/" + \
                                                 mdis_index_df['PATH_NAME'].str.strip() + \
                                                 mdis_index_df["FILE_NAME"].str.strip()
    mdis_index_df.loc[:, 'RESOLUTION'] = mdis_index_df.loc[:, 'HORIZONTAL_PIXEL_SCALE'].values
    mdis_index_df.loc[:, 'NAC_LINE_EXPOSURE_DURATION'] = mdis_index_df.loc[:, 'EXPOSURE_DURATION'].values
    mdis_index_df.loc[:, 'ORIGINAL_PRODUCT_ID'] = mdis_index_df.loc[:, 'FILE_NAME'].values

    def get_matching_geojson_files(outdir, latrange):
        minlat, maxlat = latrange

        # Define a regex pattern to extract latitude bins from filenames
        pattern = re.compile(rf'MDIS_{nacwac}_(-?\d+)_(-?\d+)\.geojson')
        # List and filter matching files
        matching_files = [
            filename for filename in os.listdir(outdir)
            if (match := pattern.match(filename)) and
               (lat_bin_start := int(match.group(1))) < maxlat and
               (lat_bin_end := int(match.group(2))) > minlat
        ]

        return [outdir+x for x in matching_files]

    matching_files = get_matching_geojson_files(geojson_file_path, latrange)
    if len(matching_files) == 0:
        exit(
            f"Could not find any matching geojson files in {geojson_file_path}. "
        )

    gdf_list = []
    crs = gpd.read_file(matching_files[0]).crs # assuming that all images are in the same crs
    for filename in matching_files:
        _ = gpd.read_file(filename)
        gdf_list.append(_)
    gdf = gpd.GeoDataFrame(pd.concat(gdf_list), geometry='geometry', crs=crs)

    # Align geometries with main df of images
    coordinates_df = gdf['geometry']
    coordinates_df.index = gdf['img_name']

    mdis_index_df = mdis_index_df.merge(coordinates_df, left_on='img_name', right_index=True, how='inner')

    return mdis_index_df

