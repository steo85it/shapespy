import sys
import logging
logging.basicConfig(level=logging.INFO)

import argparse
import os
import shutil
import sys
from importlib import resources
from time import time
import pandas as pd
import yaml

from sfs.site import Site
from sfs.preprocessing.import_cumindex import pds3_to_df
from sfs.preprocessing.preprocessing import clip_ldem
from sfs.config import SfsOpt


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    test_reg = Site(site='test_region', 
                    rootroot='/home/tmckenna/nobackup/new_setup/examples/Lunar_SP/test_region/', 
                    datadir=f'{os.environ["HOME"]}/nobackup/RING/data/LROC/',
                    prioridem_full='/home/tmckenna/nobackup/new_setup/examples/Lunar_SP/test_region/Site20v2_final_adj_5mpp_surf.tif',
                    imgs_to_remove={'0':['M1134119406RE', 'M1129398695LE', 'M1156128241LE', 'M1156099752RE', 'M1132598556RE', 'M1096381399RE', 'M194009213LE', 'M1198408205LE', 'M175113007RE', 'M172759995RE', 'M190249818RE', 'M1125544489RE', 'M1097310438RE', 'M1102018785RE', 'M176145539RE', 'M1139650769LE', 'M1160078961RE', 'M1131759693LE', 'M1101097181RE', 'M1113785512LE', 'M1317861059LE', 'M1134133622LE', 'M185553422RE', 'M185546298LE', 'M1259018640LE', 'M1259018640RE', 'M1146718551RE', 'M157321873RE', 'M187905200LE', 'M1149059531RE', 'M190256940LE', 'M1120845745RE', 'M192608755LE', 'M1212523042LE', 'M1212516007RE', 'M1214872046LE', 'M1214865008RE', 'M1186669763LE', 'M1132605665RE', 'M1252470864LE', 'M143217208RE', 'M1341641403LE', 'M176138791LE', 'M1106726441LE', 'M1372078480LE', 'M1137311397RE', 'M1188356318LE', 'M1372057398RE', 'M1198422276RE', 'M1096374268RE', 'M1198415251RE', 'M178483980LE', 'M1343714631LE', 'M104285126LE', 'M1172559348LE']},
                    input_shp='/home/tmckenna/nobackup/new_setup/examples/Lunar_SP/test_region/clip_te0.shp',
                    config_ba_path='/home/tmckenna/nobackup/new_setup/src/sfs/preprocessing/default_config_ba.yaml',
                    nodes_list='/home/tmckenna/nobackup/new_setup/examples/Lunar_SP/test_region/default_nodes_list.txt',
                    nimg_to_select=60,
                    crs_stereo_meters="/home/tmckenna/nobackup/new_setup/examples/aux/lunar_stereo_sp_meters.wkt",
                    pds_index_name='CUMINDEX',
                    pds_index_path='/explore/nobackup/projects/pgda/LRO/data/LROC/',
                    source_url='https://pds.lroc.asu.edu/data/',
                    calibrate='lrocnac',
                    min_resolution=2.,
                    resolution_name='RESOLUTION',
                    imglist_full='lnac_te.in',
                    sfs_smoothness_weight=0.02,
                    sfs_initial_dem_constraint_weight=1e-3,
                    )
    
    # selout_path = rough_imgsel_sites(tileid, input_shp=SfsOpt.get('input_shp'), latrange=[-90, -70])[0]
    filtered_selection_path = f"{test_reg.procdir}filtered_selection_{test_reg.siteid}_{test_reg.tileid}.csv"
    not_aligned_images_path = f"{test_reg.procdir}not_aligned_{test_reg.siteid}_{test_reg.tileid}.parquet"
    outstats = f"{test_reg.procdir}check_ba/stats_{test_reg.siteid}_{test_reg.tileid}_exp.parquet"
    final_selection_path = f"{test_reg.procdir}final_selection_{test_reg.tileid}_sel{test_reg.sel}.csv"

    print("- Running ", test_reg.steps_to_run)
        
    test_reg.init_pipeline()

    if test_reg.steps_to_run['rough_selection']:
        test_reg._rough_selection()
    else:
        test_reg.sfs_opt.get('imglist_full')
        
        # check 
    if test_reg.steps_to_run['verify_download']:
        test_reg._verify_download(test_reg.selout_path)

    tmp_selection = pd.read_csv(f"/home/tmckenna/nobackup/new_setup/examples/Lunar_SP/test_region/down_selection_0_sel0.csv")
     
    
    if test_reg.steps_to_run['preprocess']:
        filtered_tmp_selection = test_reg._preprocess(tmp_selection, filtered_selection_path)
    else:
        filtered_tmp_selection = pd.read_csv(filtered_selection_path)

    if test_reg.steps_to_run['clean_dataset']:
        test_reg._clean_dataset(filtered_tmp_selection, not_aligned_images_path, outstats)

    if test_reg.steps_to_run['refine_align']:
        test_reg._refine_align(filtered_tmp_selection, not_aligned_images_path, outstats)

    # set new shp and nimg to select for smaller 0.1x0.1 km region and clip ldem
    test_reg.sfs_opt.set('input_shp', '/home/tmckenna/nobackup/new_setup/examples/Lunar_SP/test_region/clip_0.1_te0.shp')
    test_reg.sfs_opt.set('nimg_to_select', 3)
    clip_ldem(test_reg.tileid, rescale_by=True)

    if test_reg.steps_to_run['final_selection']:
        final_selection = test_reg._final_selection(not_aligned_images_path, final_selection_path)
    else:
        try:
            final_selection = pd.read_csv(final_selection_path)
        except:
            logging.error(f"* No final selection found. Exit.")
            exit()

    # choose
    os.chdir(test_reg.seldir)
    test_reg.latest_selection = final_selection

    if test_reg.steps_to_run['align_to_dem']:
        test_reg._align_to_dem(base_resolution=2, new_rendering=True)

    if test_reg.steps_to_run['sfs']:
        test_reg._run_sfs()

    if test_reg.steps_to_run['postpro']:
        test_reg._postpro()
    