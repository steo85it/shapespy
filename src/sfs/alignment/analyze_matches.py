import glob
import logging
import subprocess
import os
import sys

import seaborn as sns

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point
import itertools as it
from p_tqdm import p_umap

from sfs.config import SfsOpt
from asp.functions import set_asp
from sfs.alignment.ba_align_res_stat import plot_img_alignments

def analyze_matches(procdir, img1, img2, img1_path, img2_path, nb_matches):

    match_coords = pd.read_csv(f"{procdir}run-{img1}_map__{img2}_map.txt",
                               skiprows=1, header=None, sep="\s+")

    pds_ds = xr.open_dataset(img1_path)
    smithed_ds = xr.open_dataset(img2_path)
    crs = smithed_ds.rio.crs

    # transform matches from figure frame to crs
    xy_matches = {img1: [], img2: []}
    for idx, val in xy_matches.items():
        if idx == img1:
            match = match_coords.iloc[:nb_matches, :2]
            transform_ds = pds_ds.rio.transform()
        else:
            match = match_coords.iloc[nb_matches:, :2]
            transform_ds = smithed_ds.rio.transform()

        transform_ds = np.vstack(transform_ds).reshape(3, 3)
        xyrot = transform_ds[:2, :2]
        xyoff = transform_ds[:2, -1]
        match_xy = np.einsum('ij, jk -> ik', match.values, xyrot) + xyoff
        # print(match_xy)
        match_xy_gpd = gpd.GeoSeries(map(Point, match_xy))
        match_xy_gpd.set_crs(crs, inplace=True)
        xy_matches[idx] = match_xy

    try:
        xerr = xy_matches[img1][:, 0] - xy_matches[img2][:, 0]
        yerr = xy_matches[img1][:, 1] - xy_matches[img2][:, 1]
        xy_residuals = np.vstack([xerr, yerr])
    except:
        print(f"Issue at {img1} and {img2}. Check {procdir}run-{img1}_map__{img2}_map.txt.")
        print(xy_matches[img1])
        print(xy_matches[img2])


    twoderr = np.linalg.norm(xy_residuals, axis=0)
    # percent_bad = len(twoderr[np.where(twoderr > 1.5)]) / len(twoderr) * 100.
    # print(percent_bad)

    # save stacked matches to file
    df = pd.DataFrame(np.hstack([xy_matches[img1], xy_matches[img2]]),
                      columns=[f'x_{img1}', f'y_{img1}', f'x_{img2}', f'y_{img2}'])
    # df.to_csv(f"{procdir}../matches_xy_{img1}_{img2}.txt", index=None)

    df['dx'] = df[f'x_{img1}'] - df[f'x_{img2}']
    df['dy'] = df[f'y_{img1}'] - df[f'y_{img2}']
    df['dxy'] = twoderr
    # print(df)
    percent_bottom_x = df['dx'].quantile(0.10)
    percent_top_x = df['dx'].quantile(0.90)
    percent_bottom_y = df['dy'].quantile(0.10)
    percent_top_y = df['dy'].quantile(0.90)
    # print(percent10_x, percent90_x)
    # print(percent10_y, percent90_y)
    df['keep'] = (df.dx > percent_bottom_x) & (df.dx < percent_top_x) & (df.dy > percent_bottom_y) & (df.dy < percent_top_y)
    
    return df

def get_stats(dfred, index_path):

    #print(dfred)
    dtype_dict = {
    'img1': str,
    'img2': str,
    'mean_dx': float,
    'mean_dy': float,
    'median_dx': float,
    'median_dy': float,
    'std_dx': float,
    'std_dy': float,
    'nb_ba_matches': int
    }

    # Convert column data types
    dfred = dfred.astype(dtype_dict)
    #print(dfred.dtypes)
    
    # collect list of images
    imgs = np.unique(dfred.img1.tolist() + dfred.img2.tolist())

    # extract useful info from index
    try:
        cumindex = pd.read_parquet(index_path)
    except:
        cumindex = pd.read_pickle(index_path)

    cumindex = cumindex.loc[cumindex.PRODUCT_ID.str.strip().isin(imgs)]
    imgs = cumindex.sort_values(by='SUB_SOLAR_LONGITUDE').PRODUCT_ID.str.strip().values
    subsolar_lon = cumindex.sort_values(by='SUB_SOLAR_LONGITUDE').SUB_SOLAR_LONGITUDE.values

    # generate empty template with all pairs
    template = pd.DataFrame(np.zeros((len(imgs), len(imgs))), index=imgs, columns=imgs)

    piv_x = dfred.pivot_table(index='img1', columns='img2', values='median_dx')
    piv_x = template + piv_x
    piv_x = piv_x.fillna(0)
    # put all x corrections to the low triangle
    pivxl = np.tril(piv_x.values)
    pivxu = np.triu(piv_x.values)
    cmb_xl = np.nan_to_num(pivxl) - np.nan_to_num(pivxu.T)
    cmb_xu = cmb_xl.T
    cmb_x = cmb_xl - cmb_xu
    piv_x[:] = np.where(cmb_x != 0., cmb_x, np.nan)
    piv_x = piv_x.loc[imgs, imgs]

    piv_x.index = np.vstack([piv_x.index,
                             subsolar_lon,
                             piv_x.median(axis=1).round(2).values,
                             abs(piv_x - piv_x.median(axis=1)).mean(axis=1).round(2).values
                             ]).T

    piv_std_x = dfred.pivot_table(index='img1', columns='img2', values='std_dx')
    piv_std_x = template + piv_std_x
    piv_std_x = piv_std_x.fillna(0)

    # put all x corrections to the low triangle
    pivxl = np.tril(piv_std_x.values)
    pivxu = np.triu(piv_std_x.values)
    cmb_xl = np.nan_to_num(pivxl) - np.nan_to_num(pivxu.T)
    cmb_xu = cmb_xl.T
    cmb_x = cmb_xl - cmb_xu
    piv_std_x[:] = np.where(cmb_x != 0., cmb_x, np.nan)
    piv_std_x = piv_std_x.loc[imgs, imgs]

    piv_std_x.index = np.vstack([piv_std_x.index,
                                 subsolar_lon,
                                 piv_std_x.median(axis=1).round(2).values,
                                 abs(piv_std_x - piv_std_x.median(axis=1)).mean(axis=1).round(2).values
                                 ]).T

    piv_y = dfred.pivot_table(index='img1', columns='img2', values='median_dy')
    piv_y = template + piv_y
    piv_y = piv_y.fillna(0)

    # put all x corrections to the low triangle
    pivyl = np.tril(piv_y.values)
    pivyu = np.triu(piv_y.values)
    cmb_yu = np.nan_to_num(pivyu) - np.nan_to_num(pivyl.T)
    cmb_yl = cmb_yu.T
    cmb_y = cmb_yu - cmb_yl
    piv_y[:] = np.where(cmb_y != 0., cmb_y, np.nan)
    piv_y = piv_y.loc[imgs, imgs]
    piv_y.index = np.vstack([piv_y.index, piv_y.median(axis=1).round(2).values,
                             abs(piv_y - piv_y.median(axis=1)).mean(axis=1).round(2).values]).T[:, 1:]

    piv_nbmatches = dfred.pivot_table(index='img1', columns='img2', values='nb_ba_matches')
    piv_nbmatches = template + piv_nbmatches
    piv_nbmatches = piv_nbmatches.fillna(0)

    # put all x corrections to the low triangle
    piv_nbmatchesl = np.tril(piv_nbmatches.values)
    piv_nbmatchesu = np.triu(piv_nbmatches.values)
    cmb_nbmatchesu = np.nan_to_num(piv_nbmatchesu) + np.nan_to_num(piv_nbmatchesl.T)
    cmb_nbmatchesl = cmb_nbmatchesu.T
    cmb_nbmatches = cmb_nbmatchesu + cmb_nbmatchesl
    piv_nbmatches[:] = np.where(cmb_nbmatches != 0., cmb_nbmatches, np.nan)
    piv_nbmatches = piv_nbmatches.loc[imgs, imgs]

    return piv_x, piv_y, piv_nbmatches


def plot_match_stats(procdir, df, outpng, maxc, imglist=None):

    opt = SfsOpt.get_instance()
    
    # extract statistics and build pivot table
    piv_x, piv_y, piv_nbmatches = get_stats(df, index_path=f"{opt.rootdir}{opt.pds_index_name}.parquet")

    # plot
    f, axes = plt.subplots(1, 3, figsize=(70, 20))
    sns.heatmap(piv_x, annot=True, fmt="5.1f", linewidths=.5, ax=axes[0], cmap='bwr',
                vmin=-maxc, vmax=maxc, cbar=False, annot_kws={"fontsize": 2})
    axes[0].set_title('dx (meters)')
    # secondary_axis(location='right', functions=(lambda x: x[0], lambda x: x[0]))
    sns.heatmap(piv_y, annot=True, fmt="5.1f", linewidths=.5, ax=axes[1], cmap='bwr',
                vmin=-maxc, vmax=maxc, cbar=False, annot_kws={"fontsize": 2})
    axes[1].set_title('dy (meters)')
    sns.heatmap(piv_nbmatches, annot=True, fmt="3.0f", linewidths=.5, ax=axes[2], cmap='viridis_r',
                vmin=0, vmax=500, cbar=False, yticklabels=False, annot_kws={"fontsize": 2})
    axes[2].set_title('nb_matches')
    plt.suptitle("Median residuals value per NAC pair (ID, SUBSOLAR_LONGITUDE, MEDIAN, STD)")
    plt.savefig(outpng)
    logging.info(f"- Alignment statistics saved to {outpng}.")
    plt.show()

    return piv_x, piv_y

def analyze_matches_plot_stats(tileid, matchdir, imgdir, selindex, outstats, outpng, maxc, ):

    opt = SfsOpt.get_instance()
    
    outstats_dir = ('/').join(outstats.split('/')[:-1])
    print(outstats_dir)
    os.makedirs(outstats_dir, exist_ok=True)

    #print(outstats)
    #print(os.path.exists(outstats))
    if os.path.exists(outstats):
        df = pd.read_parquet(outstats)
        logging.info(f"- Reading stats from {outstats}.")
        plot_match_stats(matchdir, df, outpng=outpng, maxc=maxc)
        return outstats, outpng
    
    # sel_images = pd.read_csv(selindex, sep=',').img_name.values
    sel_images = selindex.img_name.values[:]
    logging.info(f"- Looping over combinations among {len(sel_images)} images to compute matching stats.")

    def parse_match(img1img2):

        img1, img2 = img1img2

        #print(f"{matchdir}run-{img1}_map__{img2}_map.match",
        #      os.path.exists(f"{matchdir}run-{img1}_map__{img2}_map.match"))
        if not os.path.exists(f"{matchdir}run-{img1}_map__{img2}_map.match"):
            return

        if not os.path.exists(f"{matchdir}run-{img1}_map__{img2}_map.txt"):
            # convert binary match file to ascii
            subprocess.call(
                [f"{opt.aspdir}bin/parse_match_file.py",
                 f"{matchdir}run-{img1}_map__{img2}_map.match",
                 f"{matchdir}run-{img1}_map__{img2}_map.txt"])

    # parse match files (if not already done)
    assert len(glob.glob(f"{matchdir}run-*_map__*_map.match")) > 0, f"** {matchdir} contains 0 match files... weird."
    if len(glob.glob(f"{matchdir}run-*_map__*_map.match")) != \
        len(glob.glob(f"{matchdir}run-*_map__*_map.txt")):
        p_umap(parse_match, it.combinations(sel_images, 2), total=len(sel_images)*(len(sel_images)-1)/2.,
               desc='converting match files')
    else:
        logging.info("- All match files have already been parsed.")
            
    def stats_from_matches(img1img2, show=False):

        img1, img2 = img1img2

        if os.path.exists(f"{matchdir}resid_df_{img1}_{img2}.parquet") & os.path.exists(f"{matchdir}stats_df_{img1}_{img2}.parquet"):
            df = pd.read_parquet(f"{matchdir}resid_df_{img1}_{img2}.parquet")
            stats = pd.read_parquet(f"{matchdir}stats_df_{img1}_{img2}.parquet").values.tolist()
            return df, stats
        
        if not os.path.exists(f"{matchdir}run-{img1}_map__{img2}_map.txt"):
            return None, None

        img1_tif_path = f"{imgdir}{img1}_map.tif"
        img2_tif_path = f"{imgdir}{img2}_map.tif"

        if (not os.path.exists(img1_tif_path)) | (not os.path.exists(img2_tif_path)):
            return None, None

        # retrieve interesting info from ascii matches file
        with open(f"{matchdir}run-{img1}_map__{img2}_map.txt") as input_file:
            head = [next(input_file) for _ in range(1)]
        nb_matches = int(head[0].split(' ')[0])

        if nb_matches > 100:

            df = analyze_matches(matchdir, img1, img2, img1_tif_path, img2_tif_path, nb_matches)

            if not isinstance(df, pd.DataFrame):
                return None, None

            ratio_kept = len(df[df.keep == True])/len(df)
            if  ratio_kept < 0.5:
                logging.warning(f"# outliers check kept {ratio_kept*100.}%. Weird, check.") 
            
            df = df[df.keep == True]
            df = df[['dx', 'dy']]

            dx_mean = df.dx.mean()
            dy_mean = df.dy.mean()
            dx_median = df.dx.median()
            dy_median = df.dy.median()
            dx_std = df.dx.std()
            dy_std = df.dy.std()

            stats = [img1, img2, dx_mean, dy_mean, dx_median, dy_median, dx_std, dy_std, nb_matches]

            if show:
                fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
                gdf = gpd.GeoDataFrame(df[['dx', 'dy', 'dxy']],
                                       geometry=[Point(xy) for xy in zip(df[f'x_{img1}'], df[f'y_{img1}'])])
                gdf.plot(column='dx', cmap='viridis', legend=True, ax=axes[0])
                gdf.plot(column='dy', cmap='viridis', legend=True, ax=axes[1])
                gdf.plot(column='dxy', cmap='viridis', legend=True, ax=axes[2])
                plt.show()
            #print(df)
            #print(stats)            

            df.to_parquet(f"{matchdir}resid_df_{img1}_{img2}.parquet")
            pd.DataFrame([stats]).to_parquet(f"{matchdir}stats_df_{img1}_{img2}.parquet")
            
            return df, stats
        else:
            return None, None

    results = p_umap(stats_from_matches, it.combinations(sel_images, 2), total=len(sel_images)*(len(sel_images)-1)/2.,
                     desc='analyzing matches')

    dfl = [result[0] for result in results if result[0] is not None]
    stats = [result[1] for result in results if result[1] is not None]

    if len(dfl) == 0:
        print(f"- No residuals for {outstats}. Returning None, None.")
        return None, None
        
    df = pd.concat(dfl).reset_index(drop=True)
    df.to_parquet(outstats.replace('stats', 'xyres'))

    logging.info(f"- Plotting stats to {outpng}.")
    # plot stats
    df = pd.DataFrame(np.vstack(stats))
    df.columns = ['img1', 'img2', 'mean_dx', 'mean_dy', 'median_dx', 'median_dy', 'std_dx', 'std_dy', 'nb_ba_matches']
    print(df)
    df.to_parquet(outstats)
    plot_match_stats(matchdir, df, outpng=outpng, maxc=maxc)

    return outstats, outpng

if __name__ == '__main__':

    site = sys.argv[1]
    tileid = sys.argv[2]

    # procdir = f"/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/Lunar_SP/{site}/proc/"
    procdir = f"/home/sberton2/Lavoro/projects/HLS/{site}/proc/"

    resid_df = []
    for exp, maxc in {'ba3': 0.5}.items():
    # for exp, maxc in {'ba': 50, 'ba2': 0.5}.items():
        matchdir = f"{procdir}tile_{tileid}/{exp}/"
        imgdir = f"{procdir}tile_{tileid}/prj/ba/"
        outpng = f"{procdir}stats_{site}_{tileid}_{exp}.png"
        selindex = f"{procdir}../root/lnac_{site[:2]}{tileid}.in"
        selindex = pd.read_csv(selindex, sep=',')
        outstats = f"{procdir}stats_{site}_{tileid}_{exp}.parquet"

        #if os.path.exists(outstats):
        #    os.remove(outstats)
        analyze_matches_plot_stats(tileid, matchdir, imgdir, selindex, outstats, outpng, maxc,)

        resid_df.append(pd.read_parquet(outstats.replace('stats', 'xyres')))

    bad_images = plot_img_alignments(outstats, outpng=outpng.replace('stats', 'boxbars'))
    print("bad images:", bad_images)

    print(resid_df)
    #print(resid_df[0].columns)
    #print(resid_df[1].columns)

    #fig, axes = plt.subplots(1, 2)
    #resid_df[0].dx.hist(ax=axes[0], color='r')
    #resid_df[1].dx.hist(ax=axes[1], color='g')
    #plt.savefig(outpng.replace('stats', 'xyres'))
