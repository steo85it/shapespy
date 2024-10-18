import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import glob

import scipy.sparse as sp
import xarray as xr
from p_tqdm import p_umap

from asp.functions import image_align, set_asp, bundle_adjust, mapproject, dem_mosaic

from sfs.config import SfsOpt
from sfs.preprocessing.preprocessing import load_project_img

logging.basicConfig(level=logging.INFO)

def align_from_discrepancies(tileid, csvfil, map_nac_dir, diroffmap, parallel=True):

    opt = SfsOpt.get_instance()
    set_asp(opt.aspdir)

    pdir = f"{opt.procroot}tile_{tileid}/"
    dem_path = f"{pdir}ldem_{tileid}.tif"

    os.makedirs(diroffmap, exist_ok=True)

    df_list = []
    for filin in glob.glob(csvfil):
        df_list.append(pd.read_csv(filin, index_col=0))

    df = pd.concat(df_list)
    dfred = df.loc[:, ['img1', 'img2', 'median_x', 'median_y', 'nb_matches', 'nb_ba_matches']]
    dfred.to_csv(f"/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/Lunar_SP/IM1/proc/tile_0/align_lsqr.csv")
    exit()

    # remove rows with < 200 ba_matches (unreliable)
    dfred = dfred.loc[dfred.nb_ba_matches >= 1200]
    # dfred = dfred.loc[dfred.nb_matches >= 10000]

    # collect list of images
    imgs = np.unique(dfred.img1.tolist() + dfred.img2.tolist())
    imgs_dict = {x: idx for idx, x in enumerate(imgs)}
    imgs1_idx = [*map(imgs_dict.get, dfred.img1.values)]
    imgs2_idx = [*map(imgs_dict.get, dfred.img2.values)]

    # store partials (+-1) to sparse format
    coop = sp.coo_array((1*np.ones(len(dfred)), (np.arange(len(dfred)), imgs1_idx)), shape=(len(dfred), len(imgs)))
    coom = sp.coo_array((-1*np.ones(len(dfred)), (np.arange(len(dfred)), imgs2_idx)), shape=(len(dfred), len(imgs)))
    A_coo = coop+coom

    # dense version
    # generate empty template with all pairs
    # template = pd.DataFrame(index=np.arange(len(dfred)), columns=imgs)
    # template.values[np.arange(len(dfred)), imgs1_idx] = 1
    # template.values[np.arange(len(dfred)), imgs2_idx] = -1
    # template.set_index([dfred.median_x, dfred.median_y], inplace=True)
    # print(template)

    offsets_df = pd.DataFrame.from_dict(imgs_dict, orient='index', columns=['id'])
    offsets_df['img'] = imgs
    offsets_df['xoff'] = sp.linalg.lsqr(A_coo, dfred.median_x.values)[0]
    offsets_df['yoff'] = sp.linalg.lsqr(A_coo, dfred.median_y.values)[0]

    df_merged = dfred.merge(offsets_df, left_on='img1', right_index=True)
    df_merged = df_merged.merge(offsets_df, left_on='img2', right_index=True, suffixes=('_1', '_2'))
    df_merged['median_x_upd'] = df_merged[['median_x', 'xoff_1', 'xoff_2']].sum(axis=1)
    df_merged['median_y_upd'] = df_merged[['median_y', 'yoff_1', 'yoff_2']].sum(axis=1)
    print(df_merged[['median_x', 'median_x_upd', 'median_y', 'median_y_upd']])
    print(df_merged[['median_x', 'median_x_upd', 'median_y', 'median_y_upd']].abs().mean(axis=0))
    exit()

    logging.info(f'- Got {len(offsets_df)} refined offsets from LSQR of discrepancies. Computing adjust files.')

    # plot histo
    # fig, axes = plt.subplots(1, 1)
    # offsets_df.xoff.hist(alpha=0.5, color='b')
    # offsets_df.yoff.hist(alpha=0.5, color='r')
    # plt.show()

    # for nacmap in tqdm(glob.glob(map_nac)):
    def offset_to_adjust(nacmap):

        img = nacmap.split('/')[-1].split('_map')[0]

        nac_ds = xr.open_dataset(nacmap)
        # nac_ds.band_data.plot(ax=ax)
        # print(f"xy offset, img {img}:", offsets_df.loc[img, 'xoff'], offsets_df.loc[img, 'yoff'])
        nac_ds['x'] = nac_ds.x + offsets_df.loc[img, 'xoff']
        nac_ds['y'] = nac_ds.y + offsets_df.loc[img, 'yoff']
        nac_ds.band_data.rio.to_raster(f"{diroffmap}{img}_off.tif")

        image_align(pdir, [f"{diroffmap}{img}_off.tif", f"{pdir}prj/ba/{img}_map.tif"],
                    output_prefix=f"{diroffmap}run",
                    inlier_threshold=100,
                    ip_per_image=1000000,
                    ecef_transform_type="translation",
                    dem1=dem_path,
                    dem2=dem_path,
                    o=f"{diroffmap}run-align-intensity.tif",
                    stdout=f"{diroffmap}log_ia_{img}.txt")

        bundle_adjust([f"{img}.cub"], dem_path, dirnam=pdir,
                      num_passes=1, apply_initial_transform_only=True,
                      input_adjustments_prefix=f"{pdir}ba/run",
                      output_prefix=f"{diroffmap}run", parallel=False,
                      use_csm=opt.use_csm,
                      initial_transform=f"{diroffmap}run-ecef-transform.txt",
                      stdout=f"{diroffmap}tmp_ba.log")

        prj_img_path = f"{diroffmap}{img}_map.tif"
        mapproject(from_=f"{pdir}{img}.cub", to=prj_img_path,
                   bundle_adjust_prefix=f"{diroffmap}run",
                   dem=dem_path, dirnam=pdir, use_csm=opt.use_csm,
                   stdout=f"{diroffmap}tmp_mapproj.log")

    if parallel:
        p_umap(offset_to_adjust, [f"{map_nac_dir}{x}_map.tif" for x in imgs], total=len(imgs))
    else:
        for nacmap in [f"{map_nac_dir}{x}_map.tif" for x in imgs]:
            offset_to_adjust(nacmap)


def align_with_ba(tileid, matches_path, bundle_adjust_prefix, excluded=[], parallel=True):

    opt = SfsOpt.get_instance()

    pdir = f"{opt.procroot}tile_{tileid}/"
    prefix = 'ba'
    dem_path = f"{pdir}ldem_{tileid}.tif"

    #     f"{procdir}ba/run-*-clean.match"
    imgs = [x.split('/')[-1].split('_map')[0] for x in glob.glob(f"{pdir}prj/{prefix}/*_map.tif")]
    imgs_from_matches = []
    for match_path in glob.glob(matches_path):
        img_pair = ['M1'+x.split('M1')[-1]+'E' for x in match_path.split('run-')[-1].split('E.cal.echo')[:-1]]
        if (img_pair[0] in imgs) & (img_pair[1] in imgs):
            imgs_from_matches.extend(img_pair)
    input_imgs = [x for x in np.unique(imgs_from_matches) if x not in excluded]

    # bundle_adjust([f"{x}.cal.echo.cub" for x in input_imgs[:]],
    #               dem_path, dirnam=pdir,
    #               # overlap_list=f"{procdir}overlap_list.txt",
    #               num_passes=3,
    #               # ip_per_image=10000,
    #               use_csm=True,
    #               ip_per_tile=2000,  # 693,
    #               tri_weight=0.05, tri_robust_threshold=0.05,
    #               min_triangulation_angle=0.1,
    #               heights_from_dem=dem_path, heights_from_dem_weight=.05,
    #               ip_detect_method=0, parameter_tolerance=1e-8,
    #               min_matches=200,
    #               # match_first_to_last=1,
    #               max_pairwise_matches=5500,
    #               clean_match_files_prefix=f"{pdir}{prefix}/run",
    #               # rotation_weight=5000,
    #               # max_disp_error=1000,
    #               num_iterations=550,
    #               # save_intermediate_cameras=1,
    #               camera_weight=0,
    #               # entry_point=1,
    #               input_adjustments_prefix=f"{prefix}/run",
    #               # parallel=False, output_prefix="ba/run",
    #               o=f"{bundle_adjust_prefix}/run",
    #               # nodes_list=f"{procroot}../../tmp_nodeslist",
    #               # threads=1,#processes=30,# parallel_options='--sshdelay 0.1',
    #               robust_threshold=1
    #               )

    # check that all adjust files have been produced
    for img in input_imgs:
        assert os.path.exists(f"{pdir}{bundle_adjust_prefix}/run-{img}.adjust")
    # exit()

    # project adjusted images
    if parallel:
        mapproj_ba_path = p_umap(partial(load_project_img, tileid=tileid, bundle_adjust_prefix=bundle_adjust_prefix, prioridem_path=dem_path),
                                 pd.DataFrame(input_imgs, columns=['img_name']).iterrows(),
                                 total=len(input_imgs))
    else:
        mapproj_ba_path = []
        for idxrow in pd.DataFrame(input_imgs, columns=['img_name']).iterrows():
            mapproj_ba_path.append(load_project_img(idxrow, tileid, bundle_adjust_prefix=bundle_adjust_prefix, prioridem_path=dem_path))

    # check if all selected images have been pre-processed
    assert len([f"{pdir}{x}.IMG" for x in input_imgs]) == len([f"{pdir}prj/{bundle_adjust_prefix}/{x}_map.tif" for x in input_imgs])

    # and produce max_lit mosaic
    dem_mosaic(imgs=[f"{pdir}prj/{bundle_adjust_prefix}/{x}_map.tif" for x in input_imgs], dirnam=pdir,
               max=None, output_prefix=f"max_lit_{bundle_adjust_prefix}_{tileid}.tif")

if __name__ == '__main__':

    pdir = "/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/Lunar_SP/IM1/proc/tile_0/"
    csvfil = f"{pdir}stats/ba/ba_match_stats_.csv"
    map_nac_dir = f"{pdir}prj/ba/"
    diroffmap = f"{pdir}prj/off/"
    tileid = 0

    align_from_discrepancies(tileid, csvfil, map_nac_dir, diroffmap)
    align_with_ba(tileid, matches_path, excluded=[], parallel=False)