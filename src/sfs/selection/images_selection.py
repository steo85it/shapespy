import glob
import os
import time
import logging

import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from tqdm import tqdm
from p_tqdm import p_umap, t_map
from functools import partial
import shutil

from sfs.config import SfsOpt as SfsOptClass

from sfs.selection.selection_tools import optim_imgsel_shadows, filter_imgsel
from asp.functions import set_asp
from utils.gridding import new_grid
from isis.functions import set_isis
from sfs.processing.sfs_pipeline_tools import get_tile_bounds
from sfs.preprocessing.import_cumindex import pds3_to_df


def plot_coverage_stats(filename, pixel_table, pixels):
    from utils.mitsuta import angrange
    from functools import reduce

    groupby_pixel = pixel_table.groupby('pixel')
    nimg_per_pixel = groupby_pixel.img_name.nunique()
    std_sol_long = groupby_pixel[['SUB_SOLAR_LONGITUDE']].std()
    avg_sol_long = groupby_pixel.apply(lambda x: angrange(x['SUB_SOLAR_LONGITUDE'].values))
    min_resol = groupby_pixel[['RESOLUTION']].min()

    dfs = [pixels.reset_index().rename({'index': 'pixel'}, axis='columns'), nimg_per_pixel.reset_index(),
           min_resol.reset_index(), std_sol_long.reset_index(), avg_sol_long.reset_index()]
    pixel_map = reduce(lambda left, right: pd.merge(left, right, on=['pixel'],
                                                    how='outer'), dfs)
    pixel_map = pixel_map.to_crs(SfsOpt.crs_stereo_meters)
    fig, axes = plt.subplots(2, 2, sharey=True, figsize=(14, 14), constrained_layout=True)
    pixel_map.plot("img_name", legend=True, ax=axes[0, 0], legend_kwds={'shrink': 0.9}, vmin=0, vmax=20)
    pixel_map.plot("RESOLUTION", legend=True, ax=axes[0, 1], legend_kwds={'shrink': 0.9}, vmin=1., vmax=2.)
    pixel_map.plot("SUB_SOLAR_LONGITUDE", legend=True, ax=axes[1, 0], legend_kwds={'shrink': 0.9}, vmin=0, vmax=180)
    pixel_map.plot(0, legend=True, ax=axes[1, 1], legend_kwds={'shrink': 0.9}, vmin=0, vmax=180)
    axes[0, 0].set_title('number of NAC images')  # , fontsize=8.)
    axes[0, 1].set_title('Resolution (min, m/pix)')
    axes[1, 0].set_title('Sun azimuth (std, deg)')  # , fontsize=8.)
    axes[1, 1].set_title('Sun azimuth (range, deg)')  # , fontsize=8.)
    fig.supxlabel("km from SP (x)")  # , fontsize=10.)
    fig.supylabel("km from SP (y)")  # , fontsize=8.)
    plt.savefig(filename)
    print(f"- Stats saved to {filename}.")

    return pixel_map


def read_index(tileid):
    
    SfsOpt = SfsOptClass.get_instance()
    procdir = f"{SfsOpt.procroot}tile_{tileid}/"

    os.makedirs(procdir, exist_ok=True)

    # Read preliminary selection, tile boundaries, and generate internal grid
    # -----------------------------------------
    # read previously selected nacs
    # df = pd.read_csv(f"{procroot}nac_total_per_tile.csv", header=None)
    # df.columns = ['nacid', 'tile']
    # print(f"{imglist_full}{tileid}.in")
    # TODO!! input dependent
    print(f"{SfsOpt.rootroot}{SfsOpt.imglist_full}")
    imglist_tile = glob.glob(f"{SfsOpt.rootroot}/{SfsOpt.imglist_full}")[0]
    # imglist_tile = glob.glob(f"{imglist_full}.in")[0]
    # print(imglist_tile)
    # exit()
    df = pd.read_csv(imglist_tile)  #, header=None) #, sep='\s+')
    df = df.rename({'img_name': 'nacid'}, axis=1)
    #df.columns = ['nacid', 'dum0', 'dum1', 'dum2']
    #print(df)
    # print(df.loc[df['tile'] == tileid].nacid.values)
    # retrieve cell shapefile
    # tile = xr.open_dataset(f"/home/sberton2/Lavoro/projects/Lunar_SP/220615/proc_pgda/tile_{tileid}/sfs/run-DEM-final.tif")

    if SfsOpt.input_tif == None:
        # input_shp_loc = glob.glob(f"{input_shp}")[0]
        # input_shp_loc = glob.glob(f"{input_shp}.shp")[0]
        #print("testing", SfsOpt.get("input_shp"))
        minx, miny, maxx, maxy = get_tile_bounds(tileid=tileid, filin=SfsOpt.get("input_shp"), extend=0.1)
    else:  #xcept:
        minx, miny, maxx, maxy = get_tile_bounds(filin=SfsOpt.input_tif, extend=0.)

    tile_bounds = (minx, miny, maxx, maxy)

    # Prepare "pixels" (sub-cells)
    # Dividing each cell in a 50x50 grid of "pixels" to check coverage by images
    pixels = new_grid(n_cells_per_side=SfsOpt.pixels_per_cell_per_side, bounds=[minx, miny, maxx, maxy])
    pixels = gpd.GeoDataFrame(pixels, columns=['geometry'],
                              crs=SfsOpt.crs_stereo_meters)

    # Read LROC index and extract image properties for the selection
    # ---------------------------------------
    try:
        try:
            cumindex = pd.read_parquet(f"{SfsOpt.rootdir}{SfsOpt.get('pds_index_name')}.parquet")
        except:
            cumindex = pd.read_pickle(f"{SfsOpt.rootdir}{SfsOpt.get('pds_index_name')}.pkl")
    except:
        cumindex = pds3_to_df(SfsOpt.rootdir, SfsOpt.get('pds_index_name'))

    # add missing columns
    if SfsOpt.get('calibrate')[:4] == 'mdis':
        print(cumindex.columns)
        cumindex['SLEW_ANGLE'] = cumindex['EMISSION_ANGLE'].abs().values
        cumindex['FILE_SPECIFICATION_NAME'] = 'None'
        cumindex['NAC_LINE_EXPOSURE_DURATION'] = cumindex.loc[:, 'EXPOSURE_DURATION'].values
        cumindex['ORIGINAL_PRODUCT_ID'] = cumindex.loc[:, 'FILE_NAME'].values
        cumindex['RESOLUTION'] = cumindex.loc[:, SfsOpt.get("resolution_name")].values
        cumindex['IMAGE_LINES'] = 'None'

    cumindex = cumindex[
        ['PRODUCT_ID', 'START_TIME', 'SUB_SOLAR_LONGITUDE', 'INCIDENCE_ANGLE',
         'RESOLUTION', 'SLEW_ANGLE', 'FILE_SPECIFICATION_NAME',
         'DATA_QUALITY_ID', 'NAC_LINE_EXPOSURE_DURATION', 'IMAGE_LINES', 'EMISSION_ANGLE',
         ]]
    # keeping this for compatibility reasons TODO check and replace
    cumindex['img_name'] = [s.strip() for s in cumindex[
        'PRODUCT_ID'].values]  # [x.split('/')[-1].split('.')[0] for x in cumindex['FILE_SPECIFICATION_NAME'].values]

    # shortcut
    cumindex['sol_lon'] = (cumindex['SUB_SOLAR_LONGITUDE'].values / 30).astype(int) * 30
    input_imgs = df.nacid.values  # [x + 'E' for x in df.nacid.values]
    cumindex = cumindex.loc[cumindex['img_name'].isin(input_imgs)]

    #csv_name = f"{rootdir}{SfsOpt.get('calibrate')}_clean_{tileid}.in"
    #cumindex[['img_name', 'sol_lon', SfsOpt.get("resolution_name"), 'INCIDENCE_ANGLE']].to_csv(csv_name, index=None,
    #                                                                                           header=None, sep='\t')

    # save selection to pickle
    #cumindex.to_pickle(f"{procdir}final_selection_{tileid}.pkl")
    cumindex.to_csv(f"{procdir}rough_selection_{tileid}.csv", index=None)

    return cumindex, tile_bounds, pixels


def distinct_samples(final_selection, num_sel, img_pool, frac_diff=0.5):
    """Finds a number of image samples that are distinct from the top sample in the 
    final selection DataFrame, and distinct from each other. Distinctness quantified as at least
    S-M / N-1 different images where S is the pool of images to choose from (img_pool), M is the 
    number of images in each sample (nimg_to_select), and N is the desired number of semi-unique 
    samples (num_sel).
    Args:
        final_selection (DataFrame): sorted dataframe where columns are sample_imgs, and rows with selection idx
        img_pool (int): number of images available to be in the samples
        num_sel (int): number of sample indexes to return
        frac_diff (float): percentage of images to be different between selections, should be value from 0 to 1
    Returns: 
        (DataFrame) final_selection information of samples that are semi-unique
    """

    SfsOpt = SfsOptClass.get_instance()

    print(final_selection)
    final_selection = final_selection.reset_index(drop=True)
    top = final_selection[0].values
    indexes = [0]
    if img_pool > SfsOpt.nimg_to_select:
        dist_num = (img_pool - SfsOpt.nimg_to_select) / (num_sel - 1)
        diff_images = SfsOpt.nimg_to_select * frac_diff
        threshold = min(dist_num, diff_images)
    else:
        SfsOpt.nimg_to_select = img_pool
        dist_num = 0
        threshold = 0

    print(f'Number of different images threshold: {str(threshold)}.')

    for i in tqdm(range(len(final_selection))):
        if (SfsOpt.nimg_to_select - np.sum(np.isin(final_selection.loc[i].values, top))) >= threshold:
            indexes.append(i)
            top = np.append(top, final_selection.loc[i].values)
    #selections = [final_selection[x] for x in indexes]
    print(indexes)
    
    # warning if the number of selections available is different than requested
    if num_sel > len(indexes):
        logging.warning(f'Number of selections available is {len(indexes)} when you requested {num_sel}.')
    elif num_sel < len(indexes):
        logging.warning(f'Congratulations! You got more selections than you asked for: {len(indexes)} instead of {num_sel}.')
    return final_selection.loc[np.unique(indexes)]


def get_optimal_selections(idx_seed, all_merged, num_pixels, nimg_to_select):
    # needs df with columns ['img_id', 'img_name', 'pixel', 'SUB_SOLAR_LONGITUDE/sollon'],
    selected_img_name = optim_imgsel_shadows(idx_seed, all_merged, num_pixels,
                                             min(len(all_merged.groupby('img_name')), nimg_to_select))

    all_merged_sel = all_merged.loc[all_merged['img_name'].isin(selected_img_name)]

    num_per_pixel = all_merged_sel.groupby(by='pixel')['img_name'].nunique()
    num_less = (num_per_pixel < 2).sum()
    num_more = (num_per_pixel >= 2).sum()

    percent_covered_pixels = len(all_merged_sel.pixel.unique()) / num_pixels * 100.

    all_samples = [num_less, num_more, percent_covered_pixels]

    # return pd.DataFrame([selected_img_name])
    return (selected_img_name, all_samples)


def select(tileid, num_sel=5, frac_diff=0.5, exclude=None, **kwargs):

    SfsOpt = SfsOptClass.get_instance()
    
    if exclude is None:
        exclude = []
        
    procdir = f"{SfsOpt.procroot}tile_{tileid}/"

    os.makedirs(procdir, exist_ok=True)

    # Read preliminary selection, tile boundaries, and generate internal grid
    # -----------------------------------------
    cumindex, tile_bounds, pixels = read_index(tileid)
    minx, miny, maxx, maxy = list(tile_bounds)

    # only select useful images
    cumindex = filter_imgsel(procdir, cumindex, **kwargs)

    print(f"- We select among {len(cumindex)} images...")
    print(f"- {len(cumindex.loc[cumindex['SLEW_ANGLE'].abs() > SfsOpt.get('max_slew')])} images with slew_angle>"
          f"{SfsOpt.get('max_slew')} deg have been removed.")
    cumindex = cumindex.loc[cumindex['SLEW_ANGLE'].abs() <= SfsOpt.get('max_slew')]
    # cumindex = cumindex.loc[cumindex['INCIDENCE_ANGLE'] >= 90.]

    print(f"- {len(cumindex.loc[cumindex['RESOLUTION'] > SfsOpt.get('min_resolution')])} "
          f"images with resolution>{SfsOpt.get('min_resolution')} mpp have been removed.")
    cumindex = cumindex.loc[cumindex['RESOLUTION'] <= SfsOpt.get("min_resolution")]

    cumindex['sol_lon'] = (cumindex['SUB_SOLAR_LONGITUDE'].values / 30).astype(int) * 30
    cumindex = cumindex.reset_index().rename({'index': 'img_id'}, axis=1)[:]

    print(f"- Based on 'exclude' list, remove {exclude}")
    cumindex = cumindex.loc[cumindex.img_name.isin(exclude) == False]

    print(len(cumindex), len(cumindex.groupby('img_name').count()))
    
    print(f"At this point, we are left with {len(cumindex)} images.")

    # Associate footprints (excluding shadows) and inner grid (pixels)
    # ------------------------------------
    if True: #not os.path.exists(f"{procdir}all_merged_{tileid}.parquet"): # really that cumbersome?
        all_merged = []
        full_shadowed_images = []
        for idx, row in tqdm(enumerate(cumindex.iterrows()), total=len(cumindex), desc='preparing_pixels'):
            row_dict = pd.Series.to_dict(row[-1])
            img_name = row_dict['img_name']
            # map_path = f"{procdir}prj/{SfsOpt.targetmpp}mpp/{img_name}_map.tif"
            # map_path = f"{procdir}prj/ba/{img_name}_map.tif"
            map_path = f"{procdir}prj/orig/{img_name}_map.tif"
            # read image and clip to tile
            ds = xr.open_dataset(map_path, engine="rasterio")
            ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

            # check how much to coarsen to avoid empty dimensions
            min_side = np.min([len(ds.x), len(ds.y)])
            coarsen_step = min_side / SfsOpt.get("pixels_per_cell_per_side")

            # deal with smallish overlaps
            if coarsen_step < 1.:
                coarsen_step = 1
            else:
                coarsen_step = int(coarsen_step)
            
            # maybe too fine/slow?
            # print(img_name, len(ds.x), len(ds.y), min_side, coarsen_step)
            ds = ds.coarsen(x=coarsen_step, boundary="trim").mean(). \
                coarsen(y=coarsen_step, boundary="trim").mean()

            # remove shadows
            ds = ds.where(ds.band_data > SfsOpt.shadow_threshold)

            # dataset to df
            df = ds.band_data.to_dataframe().dropna().reset_index()
            # to geopandas
            gdf = gpd.GeoDataFrame(
                df.band_data, geometry=gpd.points_from_xy(df.x, df.y), crs=SfsOpt.crs_stereo_meters)
            # bin image pixels to "large pixels" for analysis
            merged = gpd.sjoin(gdf, pixels, how='inner', op='intersects').rename({'index_right': 'pixel'}, axis=1)

            merged['img_name'] = img_name
            merged['img_id'] = row_dict['img_id']
            merged['SUB_SOLAR_LONGITUDE'] = row_dict['SUB_SOLAR_LONGITUDE']
            merged['RESOLUTION'] = row_dict['RESOLUTION']

            merged = merged.astype({'img_id': 'uint32',
                                    'SUB_SOLAR_LONGITUDE': 'float32',
                                    'RESOLUTION': 'float32',
                                    'pixel': 'uint32'})

            if len(merged) == 0:
                print(f"Check img {img_name} overlap with region. Exit.")
                exit()
                
            # WHY 10%? This does not work for very large areas covered by a large number of small images
            # do not use image if illum pixels cover <10% of tile surface
            #if len(np.unique(merged.pixel)) < 0.1 * len(pixels):
            #    full_shadowed_images.append(img_name)
            #else:
            all_merged.append(merged)

        all_merged = pd.concat(all_merged)
        # drop rows with same image (but different "sub-pixel" intensities)
        all_merged = all_merged.drop_duplicates(subset=['pixel', 'img_id'], keep='first')
        # save to file
        all_merged.to_parquet(f"{procdir}all_merged_{tileid}.parquet")
        # save fully shadowed images to file, too
        pd.DataFrame([full_shadowed_images]).to_csv(f"{procdir}full_shadowed_imgs_{tileid}.csv")
    else:  # issues with conflicting images, best recomputing
        try:
            full_shadowed_images = pd.read_csv(f"{procdir}full_shadowed_imgs_{tileid}.csv", sep=',',
                                               usecols=['0']).values[:, 0]
        except:
            full_shadowed_images = []

        all_merged = pd.read_parquet(f"{procdir}all_merged_{tileid}.parquet")
        # drop rows with same image (but different "sub-pixel" intensities)
        all_merged = all_merged.drop_duplicates(subset=['pixel', 'img_id'], keep='first')
        print(f"- List of overlapping images read from {procdir}all_merged_{tileid}.parquet")

    # at this lower resolution, one might get a few additional shadowed images...
    print(f"- {len(full_shadowed_images)} fully shadowed images removed.")
    print(full_shadowed_images)
    cumindex = cumindex.loc[~cumindex['img_name'].isin(full_shadowed_images)]

    try:
        assert len(all_merged.groupby('img_name')) == len(cumindex)  # if it fails, recompute all_merged
    except:
        shutil.move(f"{procdir}all_merged_{tileid}.parquet", f"{procdir}all_merged_{tileid}.bak")
        print(f"{len(all_merged.groupby('img_name'))} != {len(cumindex)}. "
              f"{procdir}all_merged_{tileid}.parquet has been removed. Please relaunch selection.")
        exit()

    print(f"- {len(all_merged.groupby('img_name'))} images overlap with the tile.")

    cumindex = cumindex.loc[cumindex.img_name.isin([x for x in all_merged.loc[:, 'img_name'].unique()])]

    axs = cumindex[['RESOLUTION', 'SUB_SOLAR_LONGITUDE', 'DATA_QUALITY_ID', 'SLEW_ANGLE',
                    'FILE_SPECIFICATION_NAME', 'NAC_LINE_EXPOSURE_DURATION', 'IMAGE_LINES',
                    'INCIDENCE_ANGLE', 'EMISSION_ANGLE']].hist(figsize=(15, 10), layout=(3, 3))
    plt.tight_layout()
    filnam = f"{procdir}img_hist.pdf"
    plt.savefig(filnam)
    #exit()

    # drop useless columns and reduce size
    all_merged = all_merged.drop(columns=['band_data', 'geometry', 'RESOLUTION'])

    print('all_merged:')
    print(all_merged)

    print(f"- Selecting {SfsOpt.nimg_to_select} out of {len(all_merged.groupby('img_name'))} images.")

    start = time.time()
    num_realizations = 50  # Number of times you want to run the function
    num_pixels = len(pixels)

    if True:  # parallel:
        results = p_umap(partial(get_optimal_selections, all_merged=all_merged, num_pixels=num_pixels,
                                 nimg_to_select=SfsOpt.nimg_to_select), range(num_realizations), num_cpus=10)
    else:
        results = t_map(partial(get_optimal_selections, all_merged=all_merged, num_pixels=num_pixels,
                                nimg_to_select=SfsOpt.nimg_to_select), range(num_realizations))

    # extract results
    img_selections = np.vstack([result[0] for result in results])
    img_selections = pd.DataFrame(img_selections)
    df_metrics = pd.DataFrame([result[1] for result in results],
                              columns=['num_less', 'num_more',
                                       'percent_covered_pixels'])

    # Sort by 'num_less' and 'num_more', then select the top 20
    top20_idx = df_metrics.sort_values(by=['num_less', 'num_more'], ascending=[True, False]).head(20)
    top20 = img_selections.loc[top20_idx.index]
    # Save the top 20 selections to a CSV
    top20.to_csv(f'{procdir}top20samp_selection.csv')

    # save the top selection and some "semi-distinct" other selections
    final_selection = distinct_samples(img_selections, num_sel=num_sel, img_pool=len(cumindex), frac_diff=frac_diff)
    print(final_selection)

    # update pixels table with final selection to get the correct reports, else last random selection reported
    all_merged_sel = []
    for i in range(len(final_selection)):  #added loop, append, and ['sample_imgs'].iloc[i]
        all_merged_sel.append(all_merged.loc[all_merged['img_name'].isin(final_selection.iloc[i].values)])

    # loop through top selections and save each one in csv
    selection_paths = np.empty(len(final_selection), dtype='O')
    for i in range(len(final_selection)):  # added for loop
        # print(f"- Before going back to cumindex, we have {len(final_selection)} images.")
        input_imgs = final_selection.iloc[i].values  #changed from [0] # [x + 'G' for x in final_selection]
        if SfsOpt.get('calibrate') == 'lrocnac':
            new_cumindex = cumindex.loc[cumindex['img_name'].isin(input_imgs)]
        else:
            new_cumindex = cumindex.loc[cumindex['img_name'].apply(lambda x: any(word in x for word in input_imgs))]
        # print(f"- After passing filter to cumindex, we have {len(cumindex)} images.")
        # check that we really have all the images the algo has selected
        assert len(input_imgs) == len(new_cumindex)

        csv_name = f"{SfsOpt.rootdir}{SfsOpt.get('calibrate')}_clean_{tileid}_sel{i}.in"
        new_cumindex[['img_name', 'sol_lon', 'RESOLUTION', 'INCIDENCE_ANGLE']].to_csv(csv_name, index=None,
                                                                                      header=None, sep='\t')

        # save selection to csv
        # new_cumindex.to_pickle(f"{procdir}final_selection_{tileid}_sel{i}.pkl")
        new_cumindex.to_csv(f"{procdir}final_selection_{tileid}_sel{i}.csv", index=None)
        selection_paths[i] = f"{procdir}final_selection_{tileid}_sel{i}.csv"

        print(f'- Saved selection number {str(i)} to {selection_paths[i]}')

    return selection_paths
