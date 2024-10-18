import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from old_tools.imgsel4tile import cluster_sun_azi
from sfs.config import SfsOpt

def optim_imgsel_kmedoids(df_, pixels, min_overlay=0.2):
    """
    Select most balanced subset of images according to solar longitude and spatial coverage
    :param df_:
    :param pixels:
    :param min_overlay:
    :return:
    """
    from sklearn_extra.cluster import KMedoids

    opt = SfsOpt.get_instance()

    # get list of pixels covered by each image
    df_ = join_to_pixels_check_overlay(df_, pixels, min_overlay=min_overlay)
    print(f"- Removing images overlapping by <{min_overlay}%. We are left with {len(df_.groupby(by='img_id'))} images.")

    # check if any image intersects pixels in the current cell TODO check if this can be done outside
    if len(df_.dropna()) > 0:
        gb_ = df_.groupby(by='img_id')
        img_prop = []

        for group in gb_.groups:
            pix_per_img = gb_['pixel'].get_group(group).values
            sollon_per_img = gb_['SUB_SOLAR_LONGITUDE'].get_group(group).values[0]
            img_prop.append([sollon_per_img] + [x if x in pix_per_img else 0 for x in range(len(pixels))])

        img_idx = dict(zip(range(len(gb_.groups)), list(gb_.groups.keys())))
        X = img_prop # first element of each row is sun_azimuth (deg), [1:] are instead the covered pixels

        def get_metric(a, b):
            aX = a[0] # fig1 azimuth angle in deg
            aY = a[1:] # fig1 list of covered pixels
            bX = b[0] # fig2 azimuth angle in deg
            bY = b[1:] # fig2 list of covered pixels

            # check difference in phase angle
            angle = 180. - abs(abs(aX - bX) - 180.)
            # print(f"angs, ang_diff: {aX},{bX}, {angle}")
            distX = angle
            # check difference in geographical coverage (uses % coverage of "pixels" defined in cell_utils)
            distY = (aY - bY)
            distY = len([x for x in distY if x != 0]) / 100.

            if opt.debug:
                # check if "quality factors" have comparable magnitudes
                print(f"distXY: {distX}, {distY}, {np.sqrt(distX ** 2 + distY ** 2)}")

            # compute distance between 2 images as the quadratic distance between the 2 (in the phase angle vs geocov space)
            return np.sqrt(distX ** 2 + distY ** 2)

        n_clust = opt.nimg_to_select

        kmedoids = KMedoids(n_clusters=n_clust, init='k-medoids++', max_iter=200000, metric=get_metric).fit(X) # method='pam',
        kmed = kmedoids

        selection = df_.loc[df_.img_id.isin([img_idx[x] for x in kmed.medoid_indices_])]['img_name'].unique()
        print(f"Selected {len(selection)} img out of {len(df_['img_name'].unique())}.")

        return selection

    else:
        logging.error("# 0 images selected, check what happened at tile or relax min_surface covered by images. Exit.")
        exit()

def join_to_pixels_check_overlay(df_, pixels, min_overlay = 0.1):
    # intersect images with pixels grid
    # TODO (should probably go outside)
    df_ = gpd.sjoin(df_, pixels, how='left', op='intersects').rename({'index_right': 'pixel'}, axis=1)
    # remove images by overlap
    img_to_remove = [x for x, y in df_.groupby(by='img_id') if len(y) <= min_overlay * len(pixels)]
    df_ = df_.loc[~df_.img_id.isin(img_to_remove)]
    return df_

def GA_eval(df, max_imgs, **kwargs):
    tot_pixels = 10000
    max_resol = 400
    tot_sunlon_slots = 12

    sel_imgs = [int(x) for x,y in kwargs.items() if y]
    # print(len(kwargs),len(sel_imgs))

    df = df.loc[df.img_id.isin(sel_imgs)]
    # divide per number of sun lon slots (if less are present in selection, will be penalized)
    nb_cov_pix_per_lon = df.groupby(by='sol_lon')#
    max_img_per_pixel_per_lon = []
    for gr in nb_cov_pix_per_lon.groups:
        max_img_per_pixel_per_lon.append(nb_cov_pix_per_lon.get_group(gr).groupby('pixel')['img_id'].count().max())
    max_img_per_pixel_per_lon = np.max(max_img_per_pixel_per_lon)
    # print(max_img_per_pixel_per_lon)

    # nb_cov_pix_per_lon = nb_cov_pix_per_lon['pixel'].nunique().sum() / tot_sunlon_slots

    # img_per_pixel = df.groupby(by='pixel')['img_id'].count()
    # mean_img_per_pix = img_per_pixel.mean() / len(kwargs)
    # min_img_per_pix = img_per_pixel.min() / len(kwargs)

    mean_resol = df['HORIZONTAL_PIXEL_SCALE'].mean() / max_resol

    nb_lon_slots = df['sol_lon'].nunique() / tot_sunlon_slots

    # print(nb_cov_pix,mean_img_per_pix,mean_resol,nb_lon_slots)

    # print( 2*nb_cov_pix_per_lon, 2*max_img_per_pixel_per_lon, #0.*mean_img_per_pix + 0*min_img_per_pix - \
    #        3*mean_resol , 5*nb_lon_slots, 6*len(sel_imgs)/len(kwargs) )
    return (
                   # 1.e5*nb_cov_pix_per_lon
             - 1000 * max_img_per_pixel_per_lon #-2) # 1000*max_img_per_pixel_per_lon + #0.*mean_img_per_pix + 0*min_img_per_pix - \
           # -3*mean_resol
           #   + 1.e5*nb_lon_slots
           #   - np.exp(len(sel_imgs)-max_imgs) # /len(kwargs)
             )#/ 35

def check_GA_res(df, params):
    sel_imgs = [int(x) for x,y in params.items() if y]
    df = df.loc[df.img_id.isin(sel_imgs)]

    nb_cov_pix_per_lon = df.groupby(by='sol_lon')#
    max_img_per_pixel_per_lon = []
    for gr in nb_cov_pix_per_lon.groups:
        max_img_per_pixel_per_lon.append(nb_cov_pix_per_lon.get_group(gr).groupby('pixel')['img_id'].count().max())
    max_img_per_pixel_per_lon = np.max(max_img_per_pixel_per_lon)
    nb_cov_pix_per_lon = nb_cov_pix_per_lon['pixel'].nunique().sum() / 12

    img_per_pixel = df.groupby(by='pixel')['img_id'].count()
    mean_img_per_pix = img_per_pixel.mean()
    min_img_per_pix = img_per_pixel.min()

    mean_resol = df['HORIZONTAL_PIXEL_SCALE'].mean()
    nb_lon_slots = df['sol_lon'].nunique()

    return {"nb_cov_pix_per_lon":nb_cov_pix_per_lon, "max_img_per_pixel_per_lon": max_img_per_pixel_per_lon,
            "min_img_per_pix":min_img_per_pix, "mean_img_per_pix":mean_img_per_pix,
            "mean_resol":mean_resol, "nb_lon_slots":nb_lon_slots,
            "len(sel_imgs)":len(sel_imgs)}

def optim_imgsel2(df_,pixels):
    from evolutionary_search import maximize

    # print(df_)
    #print(df_['sol_lon'].unique())
    # intersect images with pixels grid
    df_ = gpd.sjoin(df_, pixels, how='left', op='intersects').rename({'index_right': 'pixel'}, axis=1).dropna().reset_index()
    # print(df_.columns)
    df_ = df_[['img_id','img_name','HORIZONTAL_PIXEL_SCALE','sol_lon','pixel']]
    #print(df_['sol_lon'].unique())
    #print(df_)
    # exit()
    df_.to_pickle("/home/sberton2/tmp/img21_df.pkl")
    # get unique images in cell
    img_in_cell = df_['img_id'].unique()
    # prepare arguments for maximize fct
    param_grid = {str(img): [True, False] for img in img_in_cell}
    args = {'df': df_, 'max_imgs': 25}
    # run GA
    best_params, best_score, score_results, _, _ = maximize(GA_eval, param_grid, args, verbose=True,population_size=50,
                                                             gene_mutation_prob=0.99, gene_crossover_prob=0.5,
                                                             tournament_size=3, generations_number=50, gene_type=None,
                                                             n_jobs=1)
    # print(check_GA_res(df_, best_params))
    # return list of img_id selected by algo
    return df_.loc[df_.img_id.isin([int(x) for x, y in best_params.items() if y])]['img_name'].values



def optim_imgsel(df_,columns,pixels,sample_size=20):
    """
    Select best images in df[columns] covering an area gridded with pixels

    Parameters
    ----------
    df_ df with columns
    columns dict {pixel,img_name,solar_longitude} - names of useful df_ columns for the algorithm TODO find a more flexible/clearer way
    pixels geodataframe with pixel locations

    Returns
    -------
    np array of selected image names
    """
    # intersect images with pixels grid
    df_ = gpd.sjoin(df_, pixels, how='left', op='intersects').rename({'index_right': 'pixel'}, axis=1)

    # check if any image intersects pixels in the current cell TODO check if this can be done outside
    if len(df_.dropna())>0:
        # pivot table
        piv_ = df_[columns.values()].pivot_table(index=columns['pixel'], columns=columns['img_name'], values=columns['sol_lon'])

        # cluster NAC images by sol_lon and pixels covered, then extract N best images (fails if 0 img in cell)
        # try:
        # cluster images by sun longitude
        percent_covered, kmedoids_inertia, select_per_tile, clustered_img = cluster_sun_azi(
            piv_.max(axis=0).to_frame().T, piv_, 0, nsamples=500, sample_size=sample_size)
        # get binary grid of images vs pixels
        img_per_pixel = piv_.values
        img_per_pixel[img_per_pixel > 0] = 1
        img_per_pixel = pd.DataFrame(img_per_pixel, columns=piv_.columns, index=piv_.index)
        # sample best images based on sun long clusters and pixels coverage
        img_list = cluster_to_imgsel(img_per_pixel, clustered_img, nsample_sel=2000, sample_size=sample_size)
        # imgs_sel_per_group.append(img_list.values)
        return img_list.values
    # except:
    #     print(f"Issue with tile / 0 images in cell")
    #     return []
    else:
        return []

def cluster_to_imgsel(img_per_pixel, clustered_img,nsample_sel=1000,sample_size=20):
    """
    Given a list of images clustered by a parameter, select and return the best sample (criteria inside) TODO create independent criteria to pass

    Parameters
    ----------
    img_per_pixel pd.df, binary grid of images vs pixels
    clustered_img list/dict of {cluster:[images list]}
    nsample_sel int, number of random samplings
    sample_size int, number of sampled images to select

    Returns
    -------
    list of selected "sample_size" images
    """
    # print(clustered_img)
    # get list of clustered images
    df_clusters_per_row = [clustered_img]

    # for each random sampling of images, check how many pixels end up containing at least images in 2 clusters
    df_clusters = pd.DataFrame(df_clusters_per_row)
    df = df_clusters.T
    # df = df.sample(10)
    best_qual = (0, 0)

    # selection criteria are here!! TODO improve selection criteria/extract them
    for isample in range(nsample_sel):
        # extract nsample_sel samples of size=sample_size
        sample = df.sample(min(sample_size,len(df)))
        # print(sample)
        # evaluate quality of sample
        qual = (sample.nunique() > 2).sum()
        if qual > best_qual[0]:
            best_qual = (qual, sample.index)

    # print(best_qual)

    # extract selected images from input df of images
    select_per_tile = pd.DataFrame(img_per_pixel.loc[:, list(best_qual[-1])].sum(axis=0)).T
    # print(select_per_tile)

    # some plotting and statistics...
    # check number of img per each pixel
    sel_img_stats = img_per_pixel.loc[:, select_per_tile.columns]
    if True:
        nimg_per_pix = sel_img_stats.sum(axis=1)
        print("How many pixels covered by at least one image?")
        print(nimg_per_pix.loc[nimg_per_pix >= 1].count(), "out of", len(nimg_per_pix), ", or ",
              np.round(nimg_per_pix.loc[nimg_per_pix >= 1].count() / len(nimg_per_pix) * 100., 1), "%")
        # plot histo
        plt.clf()
        plt.figure()
        nimg_per_pix.hist()
        plt.xlabel('# images per pixel')
        plt.ylabel('# pixels')
        plt.savefig(f"{opt.procroot}img_per_pixel_histo_{str(sample_size)}.png")
        print("How many pixels covered by each image?")
        print(sel_img_stats.sum(axis=0))

    return select_per_tile.columns
