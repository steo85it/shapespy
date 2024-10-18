import os.path
import shutil

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt, rcParams
import geopandas as gpd
from shapely import Point
import subprocess

from sfs.config import SfsOpt
from asp.functions import ipfind, ipmatch, set_asp
msize = rcParams['lines.markersize'] ** 2

def get_disparities(img1_path, img2_path, procdir):

    aspdir = SfsOpt.get("aspdir")
    set_asp(aspdir)

    img1 = img1_path.split('/')[-1].split('.')[0]
    img2 = img2_path.split('/')[-1].split('.')[0]

    # only generate matches if .txt of results does not exist
    if True or not os.path.exists(f"{procdir}{img1}__{img2}.txt"):
        # find IPs and match them
        if True or not os.path.exists(f"{procdir}{img1}.vwip"):
            ipfind(procdir, images=[img1_path], debug_image=0, nodata_radius=1,
                   interest_operator='IAGD', #'OBALoG',
                   # gain=1, # print_ip=10,
                   ip_per_tile=20000, normalize=None,
                   stdout=f"{procdir}log_ipfind_{img1}.txt")
            if f"{img1_path.split('.')[0]}.vwip" != f"{procdir}{img1}.vwip":
                shutil.copy(f"{img1_path.split('.')[0]}.vwip",
                        f"{procdir}{img1}.vwip")
        if True or not os.path.exists(f"{procdir}{img2}.vwip"):
            ipfind(procdir, images=[img2_path], debug_image=0, nodata_radius=1,
                   interest_operator='IAGD', #'OBALoG',
                   # gain=1, # print_ip=10,
                   ip_per_tile=20000, normalize=None,
                   stdout=f"{procdir}log_ipfind_{img2}.txt")
            if f"{img2_path.split('.')[0]}.vwip" != f"{procdir}{img2}.vwip":
                shutil.copy(f"{img2_path.split('.')[0]}.vwip",
                        f"{procdir}{img2}.vwip")
        # compute matches
        ipmatch(procdir, images=[img1_path, img2_path],
                # inlier_threshold=10, matcher_threshold=0.6,
                vwip=[f"{procdir}{img1}.vwip", f"{procdir}{img2}.vwip"],
                stdout=f"{procdir}log_ipmatch_{img1}_{img2}.txt") #, debug_image=None)

        # convert binary match file to ascii
        subprocess.call(
            [f"{aspdir}bin/parse_match_file.py",
             f"{procdir}{img1}__{img2}.match",
             f"{procdir}{img1}__{img2}.txt"])

    # retrieve interesting info from ascii matches file
    with open(f"{procdir}{img1}__{img2}.txt") as input_file:
        head = [next(input_file) for _ in range(1)]
    nb_matches = int(head[0].split(' ')[0])
    match_coords = pd.read_csv(f"{procdir}{img1}__{img2}.txt",
                               skiprows=1, header=None, sep="\s+")

    pds_ds = xr.open_dataset(img1_path)
    smithed_ds = xr.open_dataset(img2_path)
    crs = smithed_ds.rio.crs

    # transform matches from figure frame to europa coords
    xy_matches = {'pds': [], 'smithed': []}
    for idx, val in xy_matches.items():
        if idx == 'pds':
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

    xerr = xy_matches['pds'][:, 0] - xy_matches['smithed'][:, 0]
    yerr = xy_matches['pds'][:, 1] - xy_matches['smithed'][:, 1]
    xy_residuals = np.vstack([xerr, yerr])

    twoderr = np.linalg.norm(xy_residuals, axis=0)
    # twoderr_list[(maxlit_1, maxlit_2)] = twoderr
    percent_bad = len(twoderr[np.where(twoderr > 1.5)]) / len(twoderr) * 100.
    # print(twoderr)
    # print(np.max(twoderr), np.min(twoderr), np.mean(twoderr), np.median(twoderr), np.std(twoderr))

    # save stacked matches to file
    df = pd.DataFrame(np.hstack([xy_matches['pds'], xy_matches['smithed']]),
                      columns=['x_pds', 'y_pds', 'x_smithed', 'y_smithed'])
    df.to_csv(f"{procdir}../matches_xy_{img1}_{img2}.txt", index=None)

    # plot projected images with matched IPs
    if False: 
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        pds_ds.band_data.plot(robust=True, ax=axes[0], cmap='viridis')
        axes[0].scatter(x=xy_matches['pds'][:, 0], y=xy_matches['pds'][:, 1],
                        c=twoderr, cmap='jet', #vmax=int(np.max(2., np.nanmin(twoderr))),
                        s=msize / 8)
        # axes[0].set_xlim(-1.3e6, -1.12e6)  # img dependent
        # axes[0].set_ylim(-5.e4, 1.e4)  # img dependent
        axes[0].set_title(img1)

        smithed_ds.band_data.plot(robust=True, ax=axes[1], cmap='viridis')
        axes[1].scatter(x=xy_matches['smithed'][:, 0], y=xy_matches['smithed'][:, 1],
                        c=twoderr, cmap='jet', #vmax=int(np.max(2., np.nanmin(twoderr))),
                        s=msize / 8)
        # axes[1].set_xlim(-1.3e6, -1.12e6)  # img dependent
        # axes[1].set_ylim(-5.e4, 1.e4)  # img dependent
        axes[1].set_title(img2)

        # axes[2].hist(np.where(twoderr <= 2., twoderr, 2.1), density=True)
        axes[2].hist(twoderr, density=True)

        imgdir = f"{procdir}../img/"
        os.makedirs(imgdir, exist_ok=True)
        outfig = f"{imgdir}prj_matches_{img1}_{img2}.png"
        plt.suptitle(f"{img1}_{img2}_mean={round(np.mean(twoderr), 2)} m_median={round(np.median(twoderr), 2)} "
                     f"m_std={round(np.std(twoderr), 2)}, npoints={len(twoderr)}, >1.5m={round(percent_bad, 2)}%")
        plt.savefig(outfig)
        # plt.show()
        plt.clf()
        plt.close()
    else:
        outfig = None
        
    return outfig, xy_residuals
