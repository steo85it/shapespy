import glob
import logging

from scipy.stats import anderson
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from asp.functions import set_asp
from isis.functions import set_isis
from sfs.config import SfsOpt

logging.basicConfig(level=logging.INFO)

def get_stats(csvfil, img_sublist=None, index_path=None):

    aspdir = SfsOpt.get("aspdir")
    isisdir = SfsOpt.get("isisdir")
    isisdata = SfsOpt.get("isisdata")
    set_asp(aspdir)
    set_isis(isisdir, isisdata)

    if index_path == None:
        index_path = f"{SfsOpt.get('rootdir')}{SfsOpt.get('pds_index_name')}.parquet"

    df_list = []
    print(csvfil)
    for filin in tqdm(glob.glob(csvfil)):
        df_list.append(pd.read_csv(filin, index_col=0))

    df = pd.concat(df_list).iloc[:]
    print(df.columns)

    df_new = []
    def clean_update(row):

        idx, pair = row
        to_remove_xy = []
        for ax, res_str in {'x': pair.resx, 'y': pair.resy}.items():
            res = np.asarray(np.matrix(res_str))[0]
            # print(len(res), np.mean(res), np.median(res), np.std(res), shapiro(res).pvalue > 0.05)

            # IQR
            Q1 = np.percentile(res, 25, method='midpoint')
            Q3 = np.percentile(res, 75, method='midpoint')
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            # Create arrays of Boolean values indicating the outlier rows
            upper_array = np.where(res >= upper)[0]
            lower_array = np.where(res <= lower)[0]
            to_remove_xy.extend(upper_array.tolist()+lower_array.tolist())

        pair[f'bad_res'] = to_remove_xy
        # Removing the outliers
        for ax, res_str in {'x': pair.resx, 'y': pair.resy}.items():
            res = np.asarray(np.matrix(res_str))[0]
            res = np.delete(res, to_remove_xy)
            # print(len(res), np.mean(res), np.median(res), np.std(res), shapiro(res).pvalue > 0.05, anderson(res, 'norm'))
            pair[f'new_median_{ax}'] = np.median(res)
            pair[f'new_mean_{ax}'] = np.mean(res)
            pair[f'new_std_{ax}'] = np.std(res)
            # pair[f'shapiro_normal_{ax}'] = shapiro(res).pvalue > 0.05
            pair[f'anderson_normal_{ax}'] = anderson(res, 'norm').statistic < 0.719 # 5% confidence

        return pair

    # for idx, row in enumerate(df.iterrows()):
    #     # if idx>10:
    #     #     break
    #     pair = clean_update(row)
    #     # print(pair)
    #     df_new.append(pair)

        # f, axes = plt.subplots(1, 2)  # , figsize=(40, 15))
        # axes[0].hist(np.asarray(np.matrix(pair.resx))[0], density=True, label='data')
        # value = np.random.normal(loc=pair.mean_x, scale=pair.std_x, size=10000)
        # axes[0].hist(value, alpha=0.5, density=True, label='mean')
        # value = np.random.normal(loc=pair.median_x, scale=pair.std_x, size=10000)
        # axes[0].hist(value, alpha=0.5, density=True, label='median')
        # axes[0].legend()
        # axes[0].set_title(f"{np.round(pair.mean_x, 2)}, {np.round(pair.median_x, 2)}")
        #
        # #axes[0].set_xlim(-5, 5)
        # axes[1].hist(np.asarray(np.matrix(pair.resy))[0], density=True, label='data')
        # value = np.random.normal(loc=pair.mean_y, scale=pair.std_y, size=10000)
        # axes[1].hist(value, alpha=0.5, density=True, label='mean')
        # value = np.random.normal(loc=pair.median_y, scale=pair.std_y, size=10000)
        # axes[1].hist(value, alpha=0.5, density=True, label='median')
        # axes[1].legend()
        # axes[1].set_title(f"{np.round(pair.mean_y, 2)}, {np.round(pair.median_y, 2)}")
        #
        # plt.suptitle(idx)
        # #axes[1].set_xlim(-5, 5)
        # # plt.savefig(f"{procdir}stats/ba/res_matches_hist.png")
        # plt.show()

    # exit()
    #
    # df = df.iloc[:10]
    # df_new.append(p_umap(clean_update, df.iterrows(), total=len(df)))
    # print(df_new[0]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       )
    # # print(df_new[0]['resy'])
    # exit()
    # df_new = pd.concat(df_new, axis=0)
    # print(df_new)
    #
    # exit()

    dfred = df.loc[:, ['img1', 'img2', 'median_x', 'median_y', 'std_x', 'std_y', 'nb_matches', 'nb_ba_matches']]
    # exit()

    # remove rows with < 10 matches (unreliable)
    dfred = dfred.loc[dfred.nb_matches >= 10]

    if img_sublist != None:
        logging.info(f"- Reducing matches to sublist of images {img_sublist}.")
        subl_imgs = pd.read_csv(img_sublist).PRODUCT_ID.str.strip().values

        # only select rows with imgs contained in list
        dfred = dfred.loc[(dfred.img1.isin(subl_imgs)) & (dfred.img2.isin(subl_imgs))]
        try:
            assert len(subl_imgs) == len(dfred)
        except:
            logging.error("{len(subl_imgs)} != {len(dfred)}. Stats not present for some of the requested images. ")

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

    piv_x = dfred.pivot_table(index='img1', columns='img2', values='median_x')
    piv_x = template + piv_x
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
                             abs(piv_x - piv_x.median(axis=1)).mean(axis=1).round(2).values]).T

    piv_std_x = dfred.pivot_table(index='img1', columns='img2', values='std_x')
    piv_std_x = template + piv_std_x
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
                             abs(piv_std_x - piv_std_x.median(axis=1)).mean(axis=1).round(2).values]).T

    piv_y = dfred.pivot_table(index='img1', columns='img2', values='median_y')
    piv_y = template + piv_y

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
    # put all x corrections to the low triangle
    piv_nbmatchesl = np.tril(piv_nbmatches.values)
    piv_nbmatchesu = np.triu(piv_nbmatches.values)
    cmb_nbmatchesu = np.nan_to_num(piv_nbmatchesu) + np.nan_to_num(piv_nbmatchesl.T)
    cmb_nbmatchesl = cmb_nbmatchesu.T
    cmb_nbmatches = cmb_nbmatchesu + cmb_nbmatchesl
    piv_nbmatches[:] = np.where(cmb_nbmatches != 0., cmb_nbmatches, np.nan)
    piv_nbmatches = piv_nbmatches.loc[imgs, imgs]

    # print(piv_x)
    # fig, axes = plt.subplots()
    # for i in range(len(piv_x)):
    #     piv_x.iloc[i].hist(ax=axes)
    #     piv_std_x.iloc[i].hist(ax=axes)
    #     plt.show()
    #     # plt.cla()
    # exit()

    return piv_x, piv_y, piv_nbmatches


def plot_match_stats(csvfil, outpng, maxc, imglist=None):

    # extract statistics and build pivot table
    piv_x, piv_y, piv_nbmatches = get_stats(csvfil, img_sublist=imglist)

    if False:
        f, axes = plt.subplots(1, 3)  # , figsize=(40, 15))
        piv_x.hist(ax=axes[0])
        axes[0].set_xlim(-5, 5)
        piv_y.hist(ax=axes[1])
        axes[1].set_xlim(-5, 5)
        piv_nbmatches.hist(ax=axes[2])
        # axes[0].set_xlim(-,5)
        plt.savefig(f"{procdir}stats/ba/res_matches_hist.png")
        plt.show()
        plt.clf()

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
    # plt.show()

    return piv_x, piv_y


if __name__ == '__main__':
    sol = 'ba'
    # procdir = "/home/sberton2/Lavoro/projects/Lunar_SP/DM2/proc_test/tile_4/"
    procdir = "/explore/nobackup/people/sberton2/RING/code/sfs_helper/examples/Lunar_SP/IM1/proc/tile_0/"
    img_list = None  # f"{procdir}../tile_5/final_selection_5.csv"
    csvfil = f"{procdir}stats/{sol}/{sol}*match*.csv"
    outpng = f"{procdir}stats/{sol}res_matches_.png"
    maxc = 1
    # M1098567811LE, M1146576313LE
    SfsOpt.set('rootdir', f"{procdir}../../root/")
    # plot_match_stats(csvfil, outpng, maxc)
    get_stats(csvfil)

    # csvfil2 = f"{procdir}stats/ba2/ba2*match*.csv"
    # get_stats(csvfil2)

    exit()

# -------------------------------------

# av_imgs = np.unique(np.hstack([df.img1.unique(), df.img2.unique()]))
# 
# means = {}
# medians = {}
# stds = {}
# nb_matches = {}
# 
# print(len(av_imgs))
# for img in av_imgs:
#     # print(img)
#     tmpdf = df[(df["img1"] == img) | (df["img2"] == img)]
#     tmpdf = tmpdf.sort_values(by='median')[['median', 'mean', 'std', 'nb_matches', 'nb_ba_matches']]
#     # y = [y.split() for x in tmpdf for y in re.sub('[\[\]\\n]', '', x).split("\s+")]
#     # print([float(x) for yi in y for x in yi if x != "..."])
#     # tmpdf[['median', 'mean']].hist()
#     means[img] = tmpdf.loc[:, 'mean'].mean()
#     medians[img] = tmpdf.loc[:, 'median'].mean()
#     stds[img] = tmpdf.loc[:, 'std'].mean()
#     nb_matches[img] = tmpdf.loc[:, 'nb_matches'].mean()
# 
# # only works because dicts are aligned, I think... warning!!
# df = pd.DataFrame([medians, means, stds, nb_matches]).T
# df.columns = ['median', 'mean', 'std', 'nb_matches']
# # print(df)
# 
# print(df.sort_values(by="mean", ascending=False).head())
# print(df.sort_values(by="median", ascending=False).head())
# print(df.sort_values(by="std", ascending=False).head())
# print(df.sort_values(by="nb_matches", ascending=True).head())
# 
# df.hist()
# plt.show()
# 
# cdf = df.loc[df['median'] < 5]
# print(len(cdf))
# cdf.hist()
# plt.show()
# # df = cdf.copy()
# 
# df.plot.scatter(x='nb_matches', y='median')
# plt.show()
# 
# df.plot.scatter(x='median', y='mean')
# plt.show()
# 
# df.plot.scatter(x='median', y='std')
# plt.show()
# 
# 
# # print(df)
#
