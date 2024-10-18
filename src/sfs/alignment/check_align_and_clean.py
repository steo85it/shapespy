import glob
import os.path

import numpy as np
import pandas as pd
from io import StringIO

from sfs.config import SfsOpt
from sfs.alignment.plot_dispmap_matches import plot_match_stats
from sfs.alignment.prepare_match_stats import prepare_match_stats

def find_bad_images(mapproj_match_offset_stats_path, threshold_85p, min_count=100):

    file_path = mapproj_match_offset_stats_path

    # Read the file line by line until the specified stop line
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if "# Percentiles of distances between matching pixels after mapprojecting onto DEM." in line:
                break
            if not line.startswith('#'):
                lines.append(line.strip())

    # Join the collected lines into a single string
    data = "\n".join(lines)

    # Read the data into a pandas DataFrame
    df = pd.read_csv(StringIO(data), delim_whitespace=True, header=None,
                     names=['image_path', '25%', '50%', '75%', '85%', '95%', 'count'])

    # Remove the extension from image names
    df['img'] = df['image_path'].apply(lambda x: os.path.basename(x).rsplit('.')[0])

    # Apply the filtering conditions
    filtered_df = df[df['count'] > min_count]
    filtered_df = filtered_df.sort_values(by='85%')
    print(filtered_df)

    thresholds_dict = filtered_df.quantile(0.95, numeric_only=True).to_dict()

    print("-Current status at iter: nimg_pre, nimg_filtered, thresholds")
    print(len(df), len(filtered_df))
    print(thresholds_dict)

    filtered_df = filtered_df[(filtered_df['85%'] < thresholds_dict['85%']) |  # mod for MDIS!!! uncomment!!
                              (filtered_df['85%'] < threshold_85p)]

    # Print or save the filtered DataFrame
    bad_images = df.loc[~df['img'].isin(filtered_df.img)]
    all_images = df.copy()

    if len(bad_images) == 0: #thresholds_dict['85%'] <= threshold_85p:
        print(f"- Done with BA+filtering, {thresholds_dict['85%']} <= {threshold_85p} and no bad images.")
        return None, all_images #df.copy()
    else:
        print(f"- Keep iterating BA+filtering, {thresholds_dict['85%']} > {threshold_85p} or num_bad_images > 0 ({len(bad_images)}).")
        return bad_images, all_images

def clean_subset_by_disparities(procdir, tileid, prefixes, matches_path, disparity_limit=1, to_exclude=[]):

    opt = SfsOpt.get_instance()

    # check that match files exist
    assert os.path.exists(glob.glob(matches_path)[0])

    # check both pre- and post-BA or IA
    df_disp = []
    for prefix in prefixes:
        outdir = f"{procdir}stats/{prefix}"
        outpng = f"{outdir}res_matches_{prefix[:-1]}.png"
        try:
            csvfil = f"{outdir}{prefix[:-1]}_match_stats_.csv"
            assert os.path.exists(csvfil)
        except:
            csvfil = prepare_match_stats(procdir, prefix, matches_path=matches_path)

        # extract statistics and plot
        piv_x, piv_y = plot_match_stats(csvfil=csvfil, outpng=outpng, maxc=20)

        dfl = {}
        for idx, piv in {'x': piv_x, 'y': piv_y}.items():
            dfl[idx] = pd.DataFrame(list(piv.index))
            if idx == 'x':
                dfl[idx].columns = ['imgid', 'subsol_lon', f'median_disp_{idx}', f'rms_disp_{idx}']
            else:
                dfl[idx].columns = [f'median_disp_{idx}', f'rms_disp_{idx}']
        df_disp = pd.concat(dfl.values(), axis=1)

    # remove bad images
    if len(df_disp) > 0:
        # remove images for which no match_file has been found
        allimgs = [x.split('/')[-1].split('_map')[0] for x in glob.glob(f"{procdir}prj/{prefix}/*_map.tif")]
        imgs_with_no_bamatches = [x for x in allimgs if x not in df_disp.imgid.values]
        to_exclude = to_exclude + imgs_with_no_bamatches
        print(f"- We removed {len(imgs_with_no_bamatches)} images with no BA matches.")
        # update variable and config file
        opt.set("imgs_to_remove", {f'{tileid}': to_exclude})
        opt.to_json(opt.get('config_file'))

        # remove all images aligning worse than a threshold
        # check how many are left
        # after the last adjustment, check for "problematic images"
        percent90_x = df_disp['median_disp_x'].quantile(0.90)
        percent90_y = df_disp['median_disp_y'].quantile(0.90)
        bad_disp = df_disp.loc[(df_disp.median_disp_x.abs() > np.min([percent90_x, disparity_limit])) |
                               (df_disp.median_disp_y.abs() > np.min([percent90_y, disparity_limit]))]
        percent_excluded = len(bad_disp) / len(df_disp) * 100.
        to_exclude = to_exclude + bad_disp.imgid.values.tolist()
        print(f"- Based on disparities, we removed {percent_excluded}% of images.")
        # update variable and config file
        opt.set("imgs_to_remove", {f'{tileid}': to_exclude})
        opt.to_json(SfsOpt.get('config_file'))
    else:
        print(f"* Input disparities list is empty. Quit.")
        exit()

    return bad_disp

if __name__ == '__main__':

    opt = SfsOpt.get_instance()
    procdir = "/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/Lunar_SP/CH3/proc/tile_0/"
    tileid = 0
    prefixes = [f'1mpp/'] #, 'ba/'] #, 'ia/']
    opt.set("local", False)
    opt.check_consistency()

    clean_subset_by_disparities(procdir, tileid, prefixes, matches_path=f"{procdir}ba/run-*-clean.match",
                                disparity_limit=1, to_exclude=[])
