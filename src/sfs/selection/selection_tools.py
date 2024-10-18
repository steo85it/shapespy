import glob
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sfs.config import SfsOpt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_metric(a, b):
    distX, distY = get_metric_extended(a, b)

    #if debug:
        # check if "quality factors" have comparable magnitudes
        # print(f"distXY: {distX}, {distY}, {np.sqrt(distX ** 2 + distY ** 2)}")

    # compute distance between 2 images as the quadratic distance between the 2 (in the phase angle vs geocov space)
    # weight X: angle difference, weight Y: covered pixels
    return np.sqrt(distX ** 2 + distY ** 2)


def get_metric_extended(a, b):
    aX = a[0]  # fig1 azimuth angle in deg
    aY = a[1:]  # fig1 list of covered pixels
    bX = b[0]  # fig2 azimuth angle in deg
    bY = b[1:]  # fig2 list of covered pixels

    assert(len(aY) == len(bY))
    npixels = len(aY)
    
    # check difference in phase angle
    angle = 180. - abs(abs(aX - bX) - 180.)
    # print(f"angs, ang_diff: {aX},{bX}, {angle}")
    distX = angle
    # check difference in geographical coverage (uses % coverage of "pixels" defined in cell_utils)
    distY = (aY - bY)
    distY = len([x for x in distY if x != 0])

    # normalize "distances"/"quality factors"
    distX /= 180.
    distY /= npixels
    
    return distX, distY


def optim_imgsel_shadows(idx, df_, num_pixels, nimg_to_select):

    # needs df with columns ['img_id', 'img_name', 'pixel', 'SUB_SOLAR_LONGITUDE/sollon'],
    # rows will be repeated since more pixels correspond to same img_id

    gb_ = df_.groupby(by='img_id')
    img_prop = []
    for group in gb_.groups:
        pix_per_img = gb_['pixel'].get_group(group).values
        sollon_per_img = gb_['SUB_SOLAR_LONGITUDE'].get_group(group).values[0]
        img_prop.append([sollon_per_img] + [x if x in pix_per_img else 0 for x in range(num_pixels)])

    img_idx = dict(zip(range(len(gb_.groups)), list(gb_.groups.keys())))
    X = img_prop  # first element of each row is sun_azimuth (deg), [1:] are instead the covered pixels

    n_clust = nimg_to_select

    kmedoids = KMedoids(n_clusters=n_clust, init='k-medoids++', max_iter=100000, metric=get_metric, random_state=idx).fit(
        X)  # method='pam',
    kmed = kmedoids
    # else: # use kmeans (scales better) --> can't define metric, not really useful
    #     kmedoids = KMeans(n_clusters=n_clust, init='k-means++', max_iter=100000, metric=get_metric, random_state=idx).fit(
    #         X)  # method='pam',
    #     kmed = kmedoids
    
    selection = df_.loc[df_.img_id.isin([img_idx[x] for x in kmed.medoid_indices_])]['img_name'].unique()

    return selection

def filter_imgsel(procdir, cumindex, **kwargs):

    prj_ba_imgs = [x.split('/')[-1].split('_map')[0] for x in
                   glob.glob(f"{procdir}prj/ba/*_map.tif")]  # does this get affected by imgs_to_remove somehow?
    print(f"- We removed {len(cumindex) - len(cumindex.loc[cumindex['img_name'].isin(prj_ba_imgs)])} images "
          f"previously excluded from bundle_adjusting.")
    cumindex = cumindex.loc[cumindex['img_name'].isin(prj_ba_imgs)]

    if 'not_aligned_images_path' in kwargs:
        # remove bad images from bundle_adjust
        print(kwargs['not_aligned_images_path'])
        if os.path.exists(kwargs['not_aligned_images_path']):
            not_aligned = pd.read_parquet(kwargs['not_aligned_images_path'])
            print(not_aligned)
            cumindex = cumindex.loc[~cumindex['img_name'].isin(not_aligned.img)]
            print(f'- We removed {len(not_aligned)} images which did not align well with the rest,'
                  f' we are left with {len(cumindex)} images.')

    return cumindex