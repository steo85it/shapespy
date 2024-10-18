import glob
import os
import shutil
import subprocess
from functools import partial
import logging

import pandas as pd
from p_tqdm import p_umap

import numpy as np

from sfs.config import SfsOpt
from sfs.alignment.align_util import get_disparities
from asp.functions import set_asp
from isis.functions import set_isis

logging.basicConfig(level=logging.INFO)

def match_fun(match_path, mapdir, outdir):

    # init ASP and ISIS
    aspdir = SfsOpt.get("aspdir")
    isisdir = SfsOpt.get("isisdir")
    isisdata = SfsOpt.get("isisdata")
    set_asp(aspdir)
    set_isis(isisdir, isisdata)

    os.makedirs(outdir, exist_ok=True)

    # check nb of matches
    subprocess.call(
        [f"{aspdir}bin/parse_match_file.py",
         match_path,
         f"{match_path.split('.match')[0]}.txt"])

    if not os.path.exists(f"{match_path.split('.match')[0]}.txt"):
        print(f"Does not exist: {match_path.split('.match')[0]}.txt. Return.")
        return
        
    # retrieve interesting info from ascii matches file
    with open(f"{match_path.split('.match')[0]}.txt") as input_file:
        head = [next(input_file) for _ in range(1)]
    nb_matches = int(head[0].split(' ')[0])

    if nb_matches < 2:
        print(f"- Just found {nb_matches} matches in {match_path}. Skip.")
        return

    #print(match_path)
    img_pair = ['M1'+x.split('M1')[-1]+'E' for x in match_path.split('run-')[-1].split('E.cal.echo')[:-1]]
    print(f'{img_pair}, {nb_matches}')
    
    # select a pair of images
    img1 = f"{mapdir}{img_pair[0]}_map.tif"
    img2 = f"{mapdir}{img_pair[1]}_map.tif"

    tmpdir = f"{outdir}{img_pair[0]}_{img_pair[1]}/"
    os.makedirs(tmpdir, exist_ok=True)
    
    # check disparities
    if not os.path.exists(img1):
        print(f"{img1} does not exist. Skip.")
        return

    img1_ = f"{tmpdir}{img1.split('/')[-1]}"
    if os.path.islink(img1_):
        os.remove(img1_)
    os.symlink(img1, img1_)

    img2_ = f"{tmpdir}{img2.split('/')[-1]}"
    if os.path.islink(img2_):
        os.remove(img2_)
    os.symlink(img2, img2_)

    # get disparities
    try:
        outpng, xy_residuals = get_disparities(img1_, img2_, f"{tmpdir}")
        print(outpng)
    except:
        print(f"Issue with {img_pair}. Skip")
        return
        
    #clean up
    shutil.rmtree(tmpdir)
    
    # save stats
    residuals = np.linalg.norm(xy_residuals, axis=0)
    resid_dict = {}
    resid_dict[f"{img_pair[0]}_{img_pair[1]}"] = {'img1': img_pair[0], 'img2': img_pair[1],
                                                  'mean': np.mean(residuals), 'median': np.median(residuals), 'std': np.std(residuals),
                                                  'mean_x': np.mean(xy_residuals[0, :]), 'median_x': np.median(xy_residuals[0, :]), 'std_x': np.std(xy_residuals[0, :]),
                                                  'mean_y': np.mean(xy_residuals[1, :]), 'median_y': np.median(xy_residuals[1, :]), 'std_y': np.std(xy_residuals[1, :]),
                                                  'nb_matches': len(residuals), 'nb_ba_matches': nb_matches,
                                                  '%>1.5': len(residuals[np.where(residuals > 1.5)]) / len(residuals) * 100.,
                                                  'resx': xy_residuals[0,:].tolist(), 'resy': xy_residuals[1,:].tolist()}
    return resid_dict


def prepare_match_stats(procdir, prefix, matches_path, imglist=[]):

    outdir = f"{procdir}stats/{prefix}"
    os.makedirs(outdir, exist_ok=True)

    match_files = glob.glob(matches_path)[:]
    if len(imglist) > 0:
        imgs = [x.split('/')[-1].split('_map')[0] for img in imglist
                for x in glob.glob(f"{procdir}prj/{prefix}/{img}_map.tif")]
    else:
        imgs = [x.split('/')[-1].split('_map')[0] for x in glob.glob(f"{procdir}prj/{prefix}/*_map.tif")]

    # only select match files between images contained in list
    match_files_sel = []
    imgs_in_match_file = []
    for match_path in match_files[:]:
        img_pair = ['M1'+x.split('M1')[-1]+'E' for x in match_path.split('run-')[-1].split('E.cal.echo')[:-1]]
        if (img_pair[0] in imgs) & (img_pair[1] in imgs):
            match_files_sel.append(match_path)
            imgs_in_match_file.extend(img_pair)
            
    match_files = match_files_sel[:]
    print(f"- Total number of match files to process: {len(match_files)}")
    print(f"- Total images in match files: {len(set(imgs_in_match_file))}")

    out = p_umap(partial(match_fun, outdir=f"{procdir}{prefix}", mapdir=f"{procdir}prj/{prefix}"),
                 match_files, total=len(match_files))

    resid_dict = {k: v for x in out if x is not None for k, v in x.items()}

    # clean up
    [os.remove(x) for x in glob.glob(f"{procdir}{prefix}*.vwip")]
    [os.remove(x) for x in glob.glob(f"{procdir}{prefix}*debug.png")]

    #print(resid_dict)
    resid_df = pd.DataFrame.from_dict(resid_dict).T
    print(resid_df)
    csvout = f"{outdir}{prefix[:-1]}_match_stats_.csv"
    resid_df.to_csv(csvout)
    print(f"- Residuals and stats saved to {csvout}.")

    return csvout
    
if __name__ == '__main__':

    prefix = "ba/"
    tileid = 0
    
    #procdir = "/home/sberton2/Lavoro/projects/Lunar_SP/DM2/proc_test/tile_4/"
    procdir = f"/explore/nobackup/people/sberton2/RING/code/sfs_helper/examples/HLS/CH3/proc/tile_{tileid}/"
    outdir = f"{procdir}stats/{prefix}"
    img_sublist = None # f"{procdir}../tile_5/final_selection_5.csv"

    prepare_match_stats(procdir, prefix, matches_path=f"{procdir}ba/run-*-clean.match") #, img_sublist)
