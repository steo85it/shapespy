import glob
import os.path
import shutil
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
#import meshio

from sfs.alignment.align_util import get_disparities
# executor is the submission interface (logs are dumped in the folder)
from asp.functions import image_align, set_asp, mapproject, bundle_adjust
from isis.functions import set_isis
from sfs.processing.sfs_pipeline_tools import rms

from sfs.config import SfsOpt

logging.basicConfig(level=logging.INFO)

def align_as_chain(pdir, imgs, indir, outdir, nmatches_rank=0, prev_adj=None, aligned=[]):

    opt = SfsOpt.get_instance()

    # prepare list of matching pairs
    csvfil = f"{pdir}stats/ba/*match*.csv"
    print(csvfil)
    print(glob.glob(csvfil))
    assert len(glob.glob(csvfil)) > 0
    
    df_list = []
    for filin in tqdm(glob.glob(csvfil)):
        df_list.append(pd.read_csv(filin, index_col=0))

    df = pd.concat(df_list)
    dfred = df.loc[:, ['img1', 'img2', 'median_x', 'median_y', 'nb_matches', 'nb_ba_matches']]
    print(dfred.index)
    imgs = [x for x in list(set(dfred.img1.tolist()+dfred.img2.tolist())) if x in imgs]

    if len(imgs) == 0:
        print("No images to process.")
        return

    os.makedirs(f"{pdir}{outdir}/", exist_ok=True)
    os.makedirs(f"{pdir}prj/{outdir}/", exist_ok=True)

    #aligned = []
    problem_imgs = []
    for i in range(len(imgs)):

        # print(dfred.loc[dfred.index.str.contains(imgs[ref_img_idx])])
        print("# Looking for a new pair...")
        print("- Aligned list:", aligned)
        
        not_yet_aligned = [x for x in imgs if x not in aligned]
        print("- Not yet aligned:", not_yet_aligned)
        print(f"* Status: {len(aligned)} aligned / {len(not_yet_aligned)} not yet.")
        
        # select the next reference image among the not aligned ones
        for img_to_align in not_yet_aligned:

            imgA = None
            imgB = None
            
            print(f"Tentative imgs to align: {img_to_align}")

            pairs_df = dfred.loc[dfred.index.str.contains(img_to_align)].sort_values(by='nb_ba_matches', ascending=False)
            # reduce to the available images
            pairs_df = pairs_df.loc[(pairs_df.img1.isin(imgs)) & (pairs_df.img2.isin(imgs))]
            print(pairs_df[['nb_matches', 'nb_ba_matches']].head())

            # are there elements/pairs in the selection?
            assert len(pairs_df) > 0

            min_matches = min(pairs_df.values[0, -2:])

            print(f"min_match gen {img_to_align}, {min_matches}")
            if min_matches < 100:
                # remove ref_img from list
                print(f"Not enough matches for {img_to_align} ({min_matches}). Skip.")
                imgs = [x for x in imgs if x != img_to_align]
                continue

            # take the option with most matches where not both images (but at least one) are already aligned
            if len(aligned) > 0:
                pairs_sel = pairs_df.loc[np.logical_xor(pairs_df.img1.isin(aligned), pairs_df.img2.isin(aligned))]
            else: # except at first iter or if no image has yet been aligned
                pairs_sel = pairs_df.copy()

            print(pairs_sel[['nb_matches', 'nb_ba_matches']].head())

            # check if reference overlaps with any aligned image
            if len(pairs_sel) == 0:
                print(f"{img_to_align} does not overlap already aligned images. Cycle to next image.")
                continue

            # try pairing with the n-th match ranked by nmatches
            for ranked in np.arange(nmatches_rank, -1, -1):
                print(ranked)
                imgA, imgB = pairs_sel.values[min(len(pairs_sel)-1, ranked), :2]
                min_matches_with_aligned = min(pairs_sel.values[min(len(pairs_sel)-1, ranked), -2:])
                # if enough matches, we have our pair, else return to better match
                if min_matches_with_aligned < 100:
                    print(f"Not enough matches for {img_to_align} at #{ranked} ({min_matches_with_aligned}). Go back.")
                    # just cycle, get back to that later
                    continue
                else:
                    print(f"We got enough matches for {imgA} and {imgB} (#{ranked}) ({min_matches_with_aligned}). "
                          f"Let's use this.")
                    break

            # check if we got enough matches (should be done on full sel list)
            if min_matches_with_aligned <= 100:
                # remove ref_img from list
                print(f"Not enough matches for {img_to_align} ({min_matches_with_aligned}). It is no good reference image.")
                imgs = [x for x in imgs if x != img_to_align]
                continue

            # if we got here, we really want to use that image
            break

        # we have a reference image! yuppie!
        print("Reference image, expected matches, imgB, imgA:", img_to_align, min_matches_with_aligned, imgA, imgB)
        assert min_matches_with_aligned > 100

        # definitely not elegant, but the idea is that we want to use the aligned image as reference
        # when i==0, ref_img should be used as reference
        if imgA == img_to_align:
            imgB_ = imgA
            imgA = imgB
            imgB = imgB_

        # copy mapprojected imgA to ia_f folder (first image only, then aligned images will be there automatically)
        if i == 0 or len(aligned) == 0:
            if os.path.islink(f"{pdir}prj/{outdir}/{imgA}_map.tif"):
                os.remove(f"{pdir}prj/{outdir}/{imgA}_map.tif")
            #if not os.path.exists(f"{pdir}prj/{outdir}/{imgA}_map.tif"): # this shouldn't happen at first iter
            os.symlink(f"{pdir}prj/{indir}/{imgA}_map.tif", f"{pdir}prj/{outdir}/{imgA}_map.tif")

            # generate fake adjust for ref_img (for subsequent iters, equal to 0 or to adjust from previous iter)
            if prev_adj == None:
                with open(f"{pdir}{outdir}/run-{imgA}.adjust", 'w', encoding='utf-8') as f:
                    f.write(f'0 0 0')
                    f.write(f'1 0 0 0')
            else:
                if False and prev_adj == "ba/run": # when starting from BA
                    if os.path.islink(f"{pdir}{indir}/run-{imgA}.adjust"):
                        os.remove(f"{pdir}{indir}/run-{imgA}.adjust")
                    os.symlink(f"{pdir}{indir}/run-{imgA}.cal.echo.ba.adjust", # when starting from BA (should be standard)
                               f"{pdir}{indir}/run-{imgA}.adjust")

                shutil.copy(f"{pdir}{indir}/run-{imgA}.adjust", # this should work thanks to the link at i=0, and then from IA results
                            f"{pdir}{outdir}/run-{imgA}.adjust")

        print(f"Currently processing {imgA}_{imgB} (aligning {imgB} to {imgA}).")

        os.makedirs(f"{pdir}{outdir}/", exist_ok=True)

        # ########################
        if True: # i > 1:
            print(f"{pdir}prj/{outdir}/{imgA}_map.tif")
            if not os.path.exists(f"{pdir}prj/{outdir}/{imgA}_map.tif") and \
               not os.path.islink(f"{pdir}prj/{outdir}/{imgA}_map.tif"):
                   print(f"Mapprojected reference image {imgA} does not exist. Check.")
                   exit()

            print(prioridem_full)
            image_align(pdir, images=[f"{pdir}prj/{outdir}/{imgA}_map.tif", f"{pdir}prj/{indir}/{imgB}_map.tif"],
                            output_prefix=f"{pdir}{outdir}/run-{imgA}-{imgB}",
                            inlier_threshold=5000,
                            ecef_transform_type="translation",
                            dem1=opt.prioridem_full, #f"{pdir}../LDEM_60S_120M.tif",
                            dem2=opt.prioridem_full, #f"{pdir}../LDEM_60S_120M.tif",
                            o=f"{pdir}{outdir}/run-{imgA}-{imgB}-align-intensity.tif",
                            stdout=f"{pdir}{outdir}/run-{imgA}-{imgB}.out")
            print(f"- Run image_align, written output to {pdir}{outdir}/run-{imgA}-{imgB}.out.")
            
            if True: #try:
                # read cartesian shift
                df = pd.read_csv(f"{pdir}{outdir}/run-{imgA}-{imgB}-ecef-transform.txt", header=None, index_col=None)
                df = df[0].str.split('\s+', expand=True)
                df = df.astype('float')
                # if identity, something went wrong
                assert np.max(np.abs(df.values - np.identity(4))) > 1.e-12
                xyz_ecef = df.iloc[:-1, -1].values

                # read image plan shift
                xy = pd.read_csv(f"{pdir}{outdir}/run-{imgA}-{imgB}-transform.txt",
                                 header=None, sep='\s+').iloc[:2, -1].values
                # xy = xy * 1. * targetmpp # units seem to be already meters... weird
                print(xyz_ecef)
                print(xy)
            #except:
            #    print(f"Issue with {imgA}_{imgB}. Quit.")
            #    break

            if prev_adj != None:
                bundle_adjust([f"{imgB}.cub"], opt.prioridem_full, dirnam=pdir,
                              use_csm=opt.use_csm,
                              num_passes=1, apply_initial_transform_only=True,
                              input_adjustments_prefix=prev_adj,
                              output_prefix=f"{outdir}/run",
                              parallel=False, stdout='tst_ba.log',
                              initial_transform=f"{pdir}{outdir}/run-{imgA}-{imgB}-ecef-transform.txt")
            else:
                bundle_adjust([f"{imgB}.cub"], opt.prioridem_full, dirnam=pdir,
                              use_csm=opt.use_csm,
                              num_passes=1, apply_initial_transform_only=True,
                              output_prefix=f"{outdir}/run",
                              parallel=False, stdout='tst_ba.log',
                              initial_transform=f"{pdir}{outdir}/run-{imgA}-{imgB}-ecef-transform.txt")


            if not os.path.exists(f"{pdir}{outdir}/run-{imgB}.adjust"):
                print(f"Issue with ba at {imgA}. Skip.")
                exit() # continue

            print(f"- Mapprojecting {imgB}")
            mapproject(from_=f"{pdir}{imgB}.cub", bundle_adjust_prefix=f"{outdir}/run",
                       to=f"{pdir}prj/{outdir}/{imgB}_map.tif", #tr=10,
                       dem=opt.prioridem_full, dirnam=pdir, use_csm=SfsOpt.use_csm,
                       stdout='tst_prj.log')
            print("Done!")
        ###########################
            # get disp pre-ia
            os.makedirs(f"{pdir}ip_test/", exist_ok=True)
            print(f"- Computing disparities....")
            outpng, pre_residuals = get_disparities(f"{pdir}prj/{indir}/{imgA}_map.tif",
                                                f"{pdir}prj/{indir}/{imgB}_map.tif",
                                                f"{pdir}ip_test/")
            if outpng != None:
                shutil.move(outpng, f"/{os.path.join(*outpng.split('/')[:-1])}/pre_{outpng.split('/')[-1]}")
            # get disp post-ia
            outpng, post_residuals = get_disparities(f"{pdir}prj/{outdir}/{imgA}_map.tif",
                                                f"{pdir}prj/{outdir}/{imgB}_map.tif",
                                                f"{pdir}ip_test/")
            if outpng != None:
                shutil.move(outpng, f"/{os.path.join(*outpng.split('/')[:-1])}/post_{outpng.split('/')[-1]}")

            # check if postfit residuals look better or at least close enough to previous ones
            # anyway dangerous if solution isn't converging...
            print("Pre/post IA median of match residuals:", round(np.median(pre_residuals), 2),
                                                            round(np.median(post_residuals), 2))
            print("Pre/post IA rms of match residuals:", round(rms(pre_residuals), 2),
                                                         round(rms(post_residuals), 2))

            if (abs(np.median(post_residuals)) - abs(np.median(pre_residuals)) > 0.01) | \
                (rms(post_residuals) - rms(pre_residuals) > 0.1): # maybe a warning is safer here?
                
                # anyway generate fake ba adjust file for next iter
                if prev_adj == None:
                    with open(f"{pdir}{outdir}/run-{img_to_align}.adjust", 'w', encoding='utf-8') as f:
                        f.write(f'0 0 0')
                        f.write(f'1 0 0 0')
                else:
                    shutil.copy(f"{pdir}{indir}/run-{img_to_align}.adjust",
                                f"{pdir}{outdir}/run-{img_to_align}.adjust")

                    #try:
                    #    shutil.copy(f"{pdir}{indir}/run-{img_to_align}.adjusted_state.json",
                    #            f"{pdir}{outdir}/run-{img_to_align}.adjusted_state.json")
                    #except:
                    #    print(f"- No CSM adjust available for {img_to_align}.")
                        
                # if alignment good enough, keep old values, else add to bad images
                if (abs(np.median(post_residuals)) >= 0.1) & (rms(post_residuals) >= 0.2):
                    # remove img_to_align from list of images to be aligned
                    imgs = [x for x in imgs if x != img_to_align]
                    problem_imgs.append(img_to_align)

                    print(f"Failed to align {img_to_align}. Move to 'bad' list and continue.")
                    continue
                else:
                    print(f"Failed to align {img_to_align}. Alignment is good enough, keep old values.")
            else:
                print(f"- Successfully aligned {imgB} to {imgA}!")
                
            print("Plot of residuals saved to:", outpng)

        # add both images to aligned images list
        print(aligned + [imgA, imgB])
        aligned = list(set(aligned + [imgA, imgB]))

        # if we already aligned all images, stop
        print(len(aligned), len(imgs))
        if len(aligned) == len(imgs):
            break

        print(f"# After (maybe) aligning {imgA} and {imgB}, this is the status:\n"
              f"- Aligned: \n {aligned}"
              f"- Bad: \n {problem_imgs}"
              f"- Yet to align: \n {[x for x in not_yet_aligned if x not in [imgA, imgB]]}") 

        
    # - among already aligned images, look for the furthest one (in the subsolar-longitude-ranked list) providing matches with the target image to align (should avoid dead-end series)
    # - compute offsets to align to that image (which is already aligned to the prime - all alignments are only computed on pairs, never on the whole set)
    # - repeat the same, but among all images in reverse order (sort of a modified iteration)
    # - check results of alignment (besides movies/clips, he is making an "avg_lit" mosaic ... not sure what that gives, need to try)

    # once all images are done, archive plots of match stats
    shutil.move(f"/{os.path.join(*outpng.split('/')[:-1])}", f"{pdir}match_stats_{outdir}")
    print("Plot of residuals saved to:", f"{pdir}match_stats_{outdir}")

    print("Done! Summary of results:")
    print("Aligned:", aligned)
    print("Bad images to retry:", list(set(problem_imgs)))
    # clean
    shutil.rmtree(f"{pdir}ip_test")

    return aligned


if __name__ == '__main__':

    pdir = "/explore/nobackup/people/sberton2/RING/code/sfs_helper/examples/Lunar_SP/CH3/proc/tile_0/"
    indir = f"ba/"
    outdir = f"ia/"
    imgs = glob.glob(f"{pdir}prj/{indir}M*E_map.tif")
    imgs = [f.split('/')[-1].split('_map')[0] for f in imgs]
    
    align_as_chain(imgs, indir, outdir, nmatches_rank=0)
