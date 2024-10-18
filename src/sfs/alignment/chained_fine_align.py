import glob
import os.path
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm

from p_tqdm import p_umap

from sfs.alignment.align_dataset import align_as_chain
from sfs.alignment.prepare_match_stats import match_fun
from sfs.alignment.plot_dispmap_matches import get_stats
# executor is the submission interface (logs are dumped in the folder)
from asp.functions import set_asp
from isis.functions import set_isis

local = False
crs_lonlat = "+proj=lonlat +units=m +a=1737.4e3 +b=1737.4e3 +no_defs"
crs_stereo = '+proj=stere +lat_0=-90 +lon_0=0 +lat_ts=-90 +k=1 +x_0=0 +y_0=0 +units=km +a=1737.4e3 +b=1737.4e3 +no_defs'  # km, else for meters a*1000, b*1000
crs_stereo_meters = '+proj=stere +lat_0=-90 +lon_0=0 +lat_ts=-90 +k=1 +x_0=0 +y_0=0 +units=m +a=1737.4e3 +b=1737.4e3 +no_defs'  # km, else for meters a*1000, b*1000

if True:
    pdir = "/explore/nobackup/people/sberton2/RING/code/sfs_helper/examples/Lunar_SP/CH3/proc/tile_0/"
    datadir = f'{os.environ["HOME"]}/nobackup/RING/data/LROC/'
    prioridem_path = f"{pdir}ldem_6.tif"
    # lowres_imgs = [x.split('/')[-1].split('.map.tif')[0] + 'E' for x in mapproj_path]
    cumindex = pd.read_pickle(f"/explore/nobackup/people/sberton2/RING/code/sfs_helper/examples/Lunar_SP/RDG/root/CUMINDEX_LROC.pkl")
    img_sublist = None # f"{pdir}../tile_5/final_selection_5.csv"
    
    aspdir = "/explore/nobackup/people/sberton2/.local/opt/StereoPipeline-3.2.1-alpha-2023-01-24-x86_64-Linux/"
    isisdir = f"{os.environ['HOME']}/nobackup/.conda/envs/isis72/"
    isisdata = f"{os.environ['HOME']}/nobackup/isis-data"

set_asp(aspdir)
set_isis(isisdir, isisdata)

num_threads = 9

for i in np.arange(0,1,1):

    if i == 0:
        prev_adj = f"ba/run" #
        indir = f"ba" #
    else:
        prev_adj = f"ia_{i-1}/run"
        indir = f"ia_{i-1}"

    outdir = f"ia_{i}"
    ref_img_idx = 0
    nmatches_rank = 0
    mapproj_path = glob.glob(f"{pdir}prj/{indir}/*_map.tif")
    
    # create useful dirs
    if os.path.exists(f"{pdir}tmp/"):
        shutil.rmtree(f"{pdir}tmp/")
    os.makedirs(f"{pdir}tmp/")

    if os.path.exists(f"{pdir}log_slurm/"):
        shutil.rmtree(f"{pdir}log_slurm/")

    # ** Load data
    # -------------
    # select from CUMINDEX all the available lowres images
    lowres_imgs = [x.split('/')[-1].split('_map.tif')[0] for x in mapproj_path]
    print(lowres_imgs)
    
    # strip leading/trailing spaces from image name
    cumindex['PRODUCT_ID'] = cumindex[['PRODUCT_ID']].apply(lambda x: x.str.strip())
    print(cumindex.PRODUCT_ID)
    # TMP just for testing
    cumindex = cumindex.loc[cumindex.PRODUCT_ID.isin(lowres_imgs)]
    print(cumindex.PRODUCT_ID)
    # exit()
    #- choose a reference image among the chosen lot overlapping the area
    sorted_imgs = cumindex.sort_values(by='SUB_SOLAR_LONGITUDE')
    print(sorted_imgs[['PRODUCT_ID', 'INCIDENCE_ANGLE', 'SUB_SOLAR_AZIMUTH', 'SUB_SOLAR_LONGITUDE']])
    imgs = sorted_imgs.PRODUCT_ID.values[:]

    # prepare list of matching pairs
    csvfil = f"{pdir}stats/{indir}/*match*.csv"

    # check old alignment results
    piv_x, piv_y, piv_nbmatches = get_stats(pdir, csvfil, outpng=f"{pdir}tmp/", maxc=20)
    img_idx = [x[0] for x in piv_x.index]
    median_x_old = [float(x[1]) for x in piv_x.index]
    std_x_old = [float(x[2]) for x in piv_x.index]
    median_y_old = [float(x[0]) for x in piv_y.index]
    std_y_old = [float(x[1]) for x in piv_y.index]
    stats_old = pd.DataFrame(np.vstack([img_idx, median_x_old, median_y_old, std_x_old, std_y_old])).T
    stats_old.columns = ["img_idx", "median_x_old", "median_y_old", "std_x_old", "std_y_old"]
    print(stats_old)
    #stats_old['sqmed'] = np.sqrt(np.square(stats_old[["median_x_old", "median_y_old"]]).sum(axis=1))
    #stats_old['min_std'] = np.min(stats_old[["std_x_old", "std_y_old"]])
    #print(stats_old)
    #exit()
    
    df_list = []
    for filin in tqdm(glob.glob(csvfil)):
        df_list.append(pd.read_csv(filin, index_col=0))

    df = pd.concat(df_list)
    dfred = df.loc[:, ['img1', 'img2', 'median_x', 'median_y', 'nb_matches', 'nb_ba_matches']]
    print(dfred)

    # select subset of images (from full BA)
    if img_sublist != None:
        print(f"- Reducing matches to sublist of images {img_sublist}.")
        subl_imgs = pd.read_csv(img_sublist).PRODUCT_ID.str.strip().values
    
        # only select rows with imgs contained in list
        dfred = dfred.loc[(dfred.img1.isin(subl_imgs)) & (dfred.img2.isin(subl_imgs))]

    print(dfred.index)
    print(len(list(set(dfred.img1.tolist()+dfred.img2.tolist()))))
    imgs = [x for x in list(set(dfred.img1.tolist()+dfred.img2.tolist())) if x in imgs]
    
    # - loop over all other images in forward order of subsolar longitude (starting from the reference image long) looking for matches and compute xy-offsets to align
    # rearrange starting from reference image
    imgs = np.hstack([imgs[ref_img_idx:], imgs[:ref_img_idx]]).tolist()
    print(imgs)
    print(len(imgs))
    #exit()

    print("- Aligning images...")
    #exit()
    aligned = align_as_chain(pdir, imgs, indir=indir, outdir=outdir, nmatches_rank=nmatches_rank, prev_adj=prev_adj) #, aligned=aligned)
    print(aligned)
    #exit()
    
    # check alignment after iter
    match_files = glob.glob(f"{pdir}ba/run-*-clean.match")[:]
    sel_matches = {}
    for match_path in match_files:
        p1, p2 = ['M1' + x.split('M1')[-1] + 'E' for x in match_path.split('run-')[-1].split('E.cal.echo.ba')[:-1]]
        if (p1 in imgs) & (p2 in imgs):
            sel_matches[f"{p1}_{p2}"] = match_path

    testdir = f"{pdir}{indir}_t/"
    # out = [match_fun(sel_match, outdir=f"{testdir}{pair}/", mapdir=f"{pdir}prj/{indir}/")
    #        for pair, sel_match in sel_matches.items()]
    out = p_umap(match_fun, sel_matches.values(), [f"{pdir}prj/{indir}/" for x in sel_matches.keys()],
                 [f"{testdir}{x}/" for x in sel_matches.keys()], total=len(sel_matches))
    resid_dict = {k: v for x in out if x is not None for k, v in x.items()}
    # clean up and collect all images to the same folder
    for img in glob.glob(f"{testdir}*/img/*.png"):
        try:
            shutil.move(img, f"{testdir}{img.split('/')[-1]}")
            shutil.rmtree(f"{testdir}{os.path.join(*img.split('/')[:-1])}")
        except:
            print(f"# Issue cleaning up {img}.")

    # convert to df and save to file
    df = pd.DataFrame.from_dict(resid_dict).T
    print(df)
    df.to_csv(f"{pdir}tst_{indir}.csv")
