import logging
import os.path
import shutil
import time
from utils.coord_tools import unproject_stereographic, project_stereographic
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rasterio._io import Resampling
import itertools as it
from asp.functions import pc_align, set_asp, image_align, parallel_stereo
from utils.dem_tools import smooth_dem
from utils.xyzi_tools import process_geospatial_data
from sfs.config import SfsOpt


def check_alignment(tileid, sel, xyzi_full, xyzi, mapshp, priormap, sfsmap,
                    mask_shadows, shadow_threshold,
                    target_resolution, default_smoothing_resol,
                    pc_align=True, image_align=False, stereo_corr=False):

    opt = SfsOpt.get_instance()

    # define derived filenames
    sfs_map_coarse_fine = sfsmap.split('.tif')[0] + f'_coarse_fine.tif'
    priormap_trim = priormap.split('.tif')[0] + '_trim.tif'
    prefix = os.path.basename(sfsmap).split('_GLD')[0]
    
    procdir = f"{opt.procroot}tile_{tileid}/"
    seldir = f"{procdir}sel_{sel}/"
    
    # load ASP variables
    set_asp(opt.aspdir)

    # retrieve XYZI point cloud and crop to map
    # convert pc to meters
    # pc_km = pd.read_parquet(xyzi_1)
    # pc_meters = pc_km*1.e3
    # pc_meters.to_parquet(xyzi_1)

    # pc_km = pd.read_parquet(xyzi_2)
    # pc_meters = pc_km*1.e3
    # pc_meters.to_parquet(xyzi_2)

    if not os.path.exists(xyzi):
        logging.info("Starting processing LOLA XYZI...")
        start = time.time()
        process_geospatial_data(input_file_path=xyzi_full, output_file_path=xyzi, crop_to=mapshp)
        logging.info("Done processing LOLA XYZI.")
        print(round(time.time() - start, 2), "seconds")

    logging.info("Starting actual comparisons between SFS and LOLA XYZI...")
        
    # load map and mask with maxlit (if available)
    sfs_da = xr.open_dataarray(sfsmap)

    def extract_translation_vector(file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        # Regular expression to find the translation vector (lat, lon, z)
        pattern = r'Translation vector \(lat,lon,z\): Vector3\(([-\de.+]+),([-\de.+]+),([-\de.+]+)\)'

        match = re.search(pattern, content)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            z = float(match.group(3))
            return lat, lon, z
        else:
            raise ValueError("Translation vector (lat,lon,z) not found in the file")

   
    if pc_align:
        # test multiple smoothing settings
        tr_max_list = np.unique([10, 30, 100, default_smoothing_resol])
        for tr in tr_max_list:

            print(f"- Checking PCAl with a smoothing to {tr} meters...")
            
            # get a smoothed version of the dem (useful for image align and correlator)
            sfs_map_coarse_fine_da = smooth_dem(sfs_da, min_tr=target_resolution, max_tr=tr, sfs_map_coarse_fine=sfs_map_coarse_fine)

            # this MUST happen after smoothing
            if mask_shadows is not None:
                mask_da = xr.load_dataarray(mask_shadows)
                sfs_map_coarse_fine_da = xr.load_dataarray(sfs_map_coarse_fine_da)
                mask_upd = mask_da.rio.reproject_match(sfs_map_coarse_fine_da, resampling=Resampling.cubic_spline)
                sfs_map_coarse_fine_da = sfs_map_coarse_fine_da.where(mask_upd > shadow_threshold)
                sfs_map_coarse_fine_da.rio.to_raster(f"{procdir}sel_{sel}/products/{prefix}_GLDELMS_001.tif", compress='zstd')
            
            # run ASP pc_align on 5mpp SFS and LOLA point cloud:
            pc_align_kwargs = {'o': f'run_{tr}/run-DEM-nodata-final_5mpp-',
                               'csv_proj4': opt.crs_stereo_meters, 'highest_accuracy': None, 'max_displacement': 80.0,
                               'max_num_source_points': 5000000,
                               'outlier_ratio': 0.95, 'csv_format': '1:easting 2:northing 3:height_above_datum',
                               'compute_translation_only': None,
                               'diff_rotation_error': 0.99, 'num_iterations': 100,
                               'save_inv_transformed_reference_points': None,
                               }
            if os.path.isdir(f"{procdir}run_{tr}/"):
                shutil.rmtree(f"{procdir}run_{tr}/")
            call_pc_align(sfs_map_coarse_fine_da, procdir, sfsmap, xyzi, target_resolution, **pc_align_kwargs)

            # save default result to apply transformation
            if tr == default_smoothing_resol:
                if os.path.isdir(f"{procdir}run/"):
                    shutil.rmtree(f"{procdir}run/")
                shutil.copytree(f"{procdir}run_{tr}/", f"{procdir}run/")

        # get bounds, transform to lonlat, add correction from pc_align, reproject to stereo, compute dxy stereo displacement
        lbrt = sfs_da.rio.bounds()
        # lbrt_lonlat = sfs_da.rio.transform_bounds(crs_lonlat)
        lb_lonlat = unproject_stereographic(x=lbrt[0],y=lbrt[1],lon0=0.,lat0=-90.,R=1737.4e3)
        lb_lonlat = [item for sublist in lb_lonlat for item in np.atleast_1d(sublist)]
        dlatlonr = extract_translation_vector(glob.glob(f"{procdir}run/run-DEM-nodata-final_5mpp--log-pc_align-*.txt")[-1])
        lb_lonlat_upd = np.array(lb_lonlat) + np.array(dlatlonr)[1::-1]
        lb_upd = project_stereographic(lon=lb_lonlat_upd[0],lat=lb_lonlat_upd[1],lon0=0.,lat0=-90.,R=1737.4e3)
        dxy = np.array(lb_upd) - np.array(lbrt[:2])
        print("dxyr", dxy, dlatlonr[-1])
        # apply translation and save to raster
        sfs_da['x'] = sfs_da['x'] + dxy[0]
        sfs_da['y'] = sfs_da['y'] + dxy[1]
        sfs_da += dlatlonr[-1]
        sfs_da.rio.to_raster(f"{procdir}sel_{sel}/products/{prefix}_GLDELUP_001.tif", compress='zstd')
    #exit()

    if image_align:

        # get a smoothed version of the dem (useful for image align and correlator)
        smooth_dem(sfs_da, min_tr=target_resolution, max_tr=default_smoothing_resol, sfs_map_coarse_fine=sfs_map_coarse_fine)

        # run ASP image_align:
        image_align_kwargs = {'alignment_transform': 'translation', 'ecef_transform_type': 'translation',
                              'dem1': sfs_map_coarse_fine, 'dem2': priormap_trim, 'ip_per_image': 1000000,
                              'output_prefix': 'run_ia/ia', 'output_image': 'run_ia/ia.tif'}

        call_image_align(**image_align_kwargs)

    if stereo_corr:
        # run ASP stereo in correlator mode:
        corr_kernel_list = [101]  #[21, 61, 101, 151, 201]
        corr_search_list = [10, 50]
        tr_max_list = [10, 30]
        sfs_map_coarse_fine_tr = {}
        for tr in tr_max_list:
            # get a list of smoothed dem
            smoothed = sfs_map_coarse_fine.split('.tif')[0] + f'_{tr}.tif'
            smooth_dem(sfs_da, min_tr=target_resolution, max_tr=tr, sfs_map_coarse_fine=smoothed)
            sfs_map_coarse_fine_tr[tr] = smoothed

        corr_results = []
        for tr, ck, cs in it.product(tr_max_list, corr_kernel_list, corr_search_list):
            stereo_correlator_kwargs = {'correlator_mode': None, 'corr_kernel': [ck, ck],
                                        'corr_search': [-1 * cs, -1 * cs, cs, cs]} # corr_search=[xmin, ymin, xmax, ymax]

            _ = call_stereo_correlator(sfs_map_coarse_fine_tr[tr], priormap_trim, **stereo_correlator_kwargs)
            corr_results.append(_.values())

        corr_results_df = pd.DataFrame(corr_results, columns=_.keys())
        print(corr_results_df)
        corr_results_df.to_csv(f"{procdir}corr_results_df.csv", index=False)


def call_pc_align(sfs_da, procdir, sfsmap, xyzi, resolution, **kwargs):

    opt = SfsOpt.get_instance()
    
    # downsample 1.0mpp SFS by an integer # of pixels to get close to lola spot size, 5mpp:
    sfs_map_small = sfsmap.split('.tif')[0] + f'_{resolution}mpp.tif'
    da = sfs_da.rio.reproject(dst_crs=opt.crs_stereo_meters, resampling=Resampling.cubic_spline, resolution=resolution)
    da.rio.to_raster(sfs_map_small)

    # convert point cloud to csv
    xyzi_csv = xyzi.split('.parquet')[0] + '.csv'
    if not os.path.exists(xyzi_csv):
        pd.read_parquet(xyzi).to_csv(xyzi_csv, index=False, header=False)
    else:
        print(f"xyzi csv read from {xyzi_csv}.")

    #    da_1 = pd.read_parquet(xyzi_1)
    #     da_2 = pd.read_parquet(xyzi_2)
    #     da3 = pd.concat([da_1, da_2])
    #     da3.sort_values(by=['Z'])
    #
    #     concat_csv = f"{procdir}root/SL3_Site01_XYZI_crop_tile5_sorted.csv"
    #     da3.to_csv(concat_csv)
    #     da3.plot.scatter(x='X', y='Y')
    #     plt.savefig(f"{procdir}root/tile5_scatter.png")

    # # run ASP pc_align on 5mpp SFS and LOLA point cloud:
    pc_align(dirnam=procdir, point_clouds=[xyzi_csv, sfs_map_small], stdout=None, **kwargs)
    # >>>>>>>>>>>>>>>>>>>>>.
    # #                 1                  0                  0 0.1351334447972476
    # #                 0                  1                  0  4.255861452082172
    # #                 0                  0                  1 0.7863083467818797
    # #                 0                  0                  0                  1
    #


def call_image_align(**kwargs):
    sfs_map_coarse_fine = kwargs['dem1']
    priormap_trim = kwargs['dem2']

    da = xr.open_dataarray(sfs_map_coarse_fine)
    #
    # # trim and downsample LDEM to same region extent AND resolution, if not already:
    xr.open_dataarray(priormap).rio.reproject_match(da, resampling=Resampling.cubic_spline).rio.to_raster(priormap_trim)

    #
    # # run ASP image_align:
    image_align(procdir, [sfs_map_coarse_fine, priormap_trim], stdout=None, **kwargs)
    # >>>>>>>>>>>>>>
    # # output ECEF transform:
    # # more run_ia/ia-ecef-transform.txt
    # # 1 0 0 0.65253320449846797
    # # 0 1 0 4.1209450309106614
    # # 0 0 1 0.896253801882267
    # # 0 0 0 1
    #


def call_stereo_correlator(sfs_map_coarse_fine, priormap_trim, **kwargs):
    # # run ASP stereo in correlator mode:
    # # Note: you must change s with each run of parallel_stereo otherwise it will use
    # # output files from the previous run.
    # s=01

    parallel_stereo(procdir, [sfs_map_coarse_fine, priormap_trim], out_prefix='run_corr/run',
                    stdout=f'{procdir}run_corr.log', **kwargs)
    #
    # # read GeoTiff with Horizontal (x) and Vertical (y) disparity maps, and pixel quality:
    da_f = xr.open_dataarray(f'{procdir}run_corr/run-F.tif')

    # Replace all 0 values in band=2 (Q) with NaN
    da_f_q_nan = da_f.sel(band=3).where(da_f.sel(band=3) != 0, np.nan)

    # # compute statistics of Horiz disparity, masking bad pixels:
    # Multiply the two grids (Q and H) element-wise
    x_disp = da_f_q_nan * da_f.sel(band=1)

    # Multiply the two grids (Q and V) element-wise
    y_disp = da_f_q_nan * da_f.sel(band=2)

    print(f"Processing with {os.path.basename(sfs_map_coarse_fine)} "
          f"& kernel={kwargs['corr_kernel']} "
          f"& search={kwargs['corr_search']}:")
    print('x disparities, mean/median/std:',
          x_disp.mean().values, x_disp.median().values, x_disp.std().values)
    print('y disparities, mean/median/std:',
          y_disp.mean().values, y_disp.median().values, y_disp.std().values)

    if 'show' in kwargs:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        x_disp.plot(robust=True, ax=axes[0], label='x-disp')
        y_disp.plot(robust=True, ax=axes[1], label='y-disp')
        plt.show()
    # x disparities, mean/median/std: -10.259035 -10.0 1.3364316
    # y disparities, mean/median/std: 11.234481 11.0 1.190319

    # remove output so that it is not reused
    exp_name = (f"tr{os.path.basename(sfs_map_coarse_fine).split('_')[-1].split('.tif')[0]}_"
                f"ck{kwargs['corr_kernel'][0]}_cs{abs(kwargs['corr_search'][0])}")

    # shutil.rmtree(f"{procdir}run_corr/", )
    shutil.move(f"{procdir}run_corr/", f"{procdir}run_corr_{exp_name}/")

    return {'tr_smooth': os.path.basename(sfs_map_coarse_fine),
            'ckernel': kwargs['corr_kernel'], 'csearch': kwargs['corr_search'],
            'x_mean': x_disp.mean().values, 'x_median': x_disp.median().values, 'x_std': x_disp.std().values,
            'y_mean': y_disp.mean().values, 'y_median': y_disp.median().values, 'y_std': y_disp.std().values, }


if __name__ == '__main__':

    sel = 0

    procdir = '/home/sberton2/nobackup/RING/code/sfs_helper/examples/HLS/A3CLS/proc/tile_3/'
    # procdir = '/home/tmckenna/nobackup/sfs_helper/examples/HLS/A3BA/proc/tile_0/'
    # xyzi_full = f"{procdir}LDEM_80S_ADJ.parquet"
    xyzi_full = "/explore/nobackup/people/mkbarker/GCD/grid/20mpp/v4/public/final/LDEM_80S_ADJ.XYZI"
    # xyzi_full = f"{procdir}../../root/Site01_final_adj_5mpp_surf.tif"
    xyzi = f"{procdir}LDEM_80S_ADJ_XYZI_crop.parquet"
    # xyzi = f"{procdir}../../root/Site01_final_adj_XYZI_crop.parquet"
    # mapshp = f"{procdir}../../root/clip_A33.shp"
    mapshp = f"{procdir}../../root/clip_A30.shp"
    priormap = f"{procdir}ldem_3_orig.tif"
    # priormap = f"{procdir}S100_GLDELEV_001.tif"
    # logging.warning("Using sfs for both mapssssss!!!!!")
    # sfsmap = f"{procdir}alternate_sel_plots/translated_dem_sel{sel}_3.tif"
    # sfsmap = f"{procdir}sel_{sel}/products/A300_GLDELUP_001.tif" #checking that the corrections were applied from pc align the first time
    # sfsmap = f"{procdir}sfs1_sel{sel}_0.02_0.001/run-DEM-final.tif"
    sfsmap = f"{procdir}sel_0/products/A303_GLDELEV_001.tif"
    # sfsmap = "/home/emazaric/CH3_SfS_MKB_preliminary.tif"
    # mask_shadows = f"{procdir}sel_0/products/A303_GLDOMOS_001.tif"
    # mask_shadows = f"{procdir}sel_{sel}/max_lit_aligned_0_sel{sel}.tif"
    mask_shadows = f"{procdir}sel_{sel}/products/A303_GLDOMOS_001.tif"
    shadow_threshold = 0.005
    default_smoothing_resol = 30

    # define proj4 string:
    try:
        from sfs.config import SfsOpt

        opt = SfsOpt.get_instance()
        crs_stereo_meters = opt.crs_stereo_meters
        aspdir = opt.aspdir
    except:
        crs_lonlat = "+proj=lonlat +units=m +a=1737.4e3 +b=1737.4e3 +no_defs"
        crs_stereo_meters = ('+proj=stere +lat_0=-90 +lon_0=0 +lat_ts=-90 +k=1 +x_0=0 +y_0=0 +units=m '
                             '+a=1737400 +b=1737400 +no_defs')
        # aspdir = "/home/sberton2/.local/share/StereoPipeline-3.3.0-Linux/"
        aspdir = "/home/sberton2/nobackup/.local/opt/StereoPipeline-3.3.0-Linux/"

    check_alignment(procdir, xyzi_full, xyzi, mapshp, priormap, sfsmap,
                    mask_shadows, shadow_threshold, default_smoothing_resol)
