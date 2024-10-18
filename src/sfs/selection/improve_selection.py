import pandas as pd
import numpy as np
import shutil

from sfs.config import SfsOpt


def improve_selection_sunlon(selection, all_merged, rough_selection, max_images_to_add=None):

    total_num_pixels = all_merged.pixel.max()
    # all_merged_pixels = all_merged.pixel.unique()

    # Bin subsolar longitudes
    bins = range(0, 361, 30)
    bin_labels = [i for i in range(0, 360, 30)]
    all_merged['sun_lon_bin'] = pd.cut(all_merged['SUB_SOLAR_LONGITUDE'], bins=bins, labels=bin_labels,
                                       include_lowest=True, right=False)
    tmp = all_merged.drop_duplicates(subset=['pixel', 'sun_lon_bin'])
    num_sunlon_per_pixel_all = tmp.groupby('pixel')['sun_lon_bin'].count()
    all_covered_pixels = num_sunlon_per_pixel_all[num_sunlon_per_pixel_all >= 3].index
    total_pixels_good_sunlon_cov = len(all_covered_pixels)

    at_least_1_all = len(num_sunlon_per_pixel_all[num_sunlon_per_pixel_all >= 1])
    at_least_2_all = len(num_sunlon_per_pixel_all[num_sunlon_per_pixel_all >= 2])
    at_least_3_all = len(num_sunlon_per_pixel_all[num_sunlon_per_pixel_all >= 3])
    print("all", at_least_1_all, at_least_2_all, at_least_3_all)

    # check number of pixels with at least 3 different solar longitude bins from selection
    selection_merged = all_merged.loc[all_merged.img_name.isin(selection.img_name)]
    sunlon_per_pixel_sel = selection_merged.drop_duplicates(subset=['pixel', 'sun_lon_bin'])
    tmp = selection_merged.drop_duplicates(subset=['pixel', 'sun_lon_bin'])
    num_sunlon_per_pixel_sel = tmp.groupby('pixel')['sun_lon_bin'].count()
    settled_pixels = num_sunlon_per_pixel_sel[num_sunlon_per_pixel_sel >= 3].index
    sel_pixels_good_sunlon_cov = len(settled_pixels)

    at_least_1 = len(num_sunlon_per_pixel_sel[num_sunlon_per_pixel_sel >= 1])
    at_least_2 = len(num_sunlon_per_pixel_sel[num_sunlon_per_pixel_sel >= 2])
    at_least_3 = len(num_sunlon_per_pixel_sel[num_sunlon_per_pixel_sel >= 3])
    print("sel", at_least_1, at_least_2, at_least_3)
    print("total/covered all/covered sel", total_num_pixels, total_pixels_good_sunlon_cov, sel_pixels_good_sunlon_cov)

    # check status of non-settled pixels
    tmp = sunlon_per_pixel_sel.loc[~sunlon_per_pixel_sel.pixel.isin(settled_pixels)]
    sunlon_per_pixel_sel = tmp[['pixel', 'sun_lon_bin', 'img_name']].pivot_table(index='pixel', columns='sun_lon_bin',
                                                                                 aggfunc='count', fill_value=0)
    print(len(sunlon_per_pixel_sel[sunlon_per_pixel_sel.sum(axis=1) >= 3]))

    additional_images = []
    if sel_pixels_good_sunlon_cov < total_pixels_good_sunlon_cov:
        # initial missing pixels
        missing_pixels = [x for x in all_covered_pixels if x not in settled_pixels]
        pixels_to_gain = len(missing_pixels)
        # print(pixels_to_gain, total_illuminated_pixels, 0.05 * total_illuminated_pixels)

        while pixels_to_gain > 0.001 * total_pixels_good_sunlon_cov:

            print(f"- {pixels_to_gain}/{total_pixels_good_sunlon_cov} "
                  f"({round(pixels_to_gain/total_pixels_good_sunlon_cov*100.,1)}%) "
                  f"still up for grabs. Continue.")
            images_in_missing_pixels = all_merged.loc[all_merged.pixel.isin(missing_pixels)]
            images_in_missing_pixels = images_in_missing_pixels[['pixel', 'img_name', 'sun_lon_bin']]
            already_selected = selection.img_name.to_list() + additional_images
            images_in_missing_pixels = images_in_missing_pixels.loc[~images_in_missing_pixels.img_name.isin(already_selected)]

            image_contribution_1 = {}
            image_contribution_2 = {}
            image_contribution_3 = {}
            for new_img in images_in_missing_pixels.img_name.unique():
                selection_merged = all_merged.loc[all_merged.img_name.isin(selection.img_name.to_list() + [new_img])]
                sunlon_per_pixel_new = selection_merged.drop_duplicates(subset=['pixel', 'sun_lon_bin'])
                tmp = sunlon_per_pixel_new.loc[~sunlon_per_pixel_new.pixel.isin(settled_pixels)]
                sunlon_per_pixel_new = tmp[['pixel', 'sun_lon_bin', 'img_name']].pivot_table(index='pixel',
                                                                                             columns='sun_lon_bin',
                                                                                             aggfunc='count', fill_value=0)
                image_contribution_1[new_img] = len(sunlon_per_pixel_new[sunlon_per_pixel_new.sum(axis=1) >= 1])
                image_contribution_2[new_img] = len(sunlon_per_pixel_new[sunlon_per_pixel_new.sum(axis=1) >= 2])
                image_contribution_3[new_img] = len(sunlon_per_pixel_new[sunlon_per_pixel_new.sum(axis=1) >= 3])

            # TODO improve the logic of this!! ^^
            # check which image delivers most pixels with 3+ sunlon
            num_pix_per_image = pd.Series(image_contribution_3).sort_values()
            # stop if not adding anything useful
            if num_pix_per_image.max() == 0:
                # else, check which image delivers most pixels with 2+ sunlon
                num_pix_per_image = pd.Series(image_contribution_2).sort_values()
                if num_pix_per_image.max() == 0:
                    # else, check which image delivers most pixels with 1+ sunlon
                    num_pix_per_image = pd.Series(image_contribution_1).sort_values()
                    if num_pix_per_image.max() == 0:
                        print(f"- Residual images don't add anything. Stop adding.")
                        break
                
            mbpb = num_pix_per_image[-1:].index.to_list()
            additional_images.extend(mbpb)
            print(f"- Adding {mbpb} to list of images...")

            selection_merged = all_merged.loc[all_merged.img_name.isin(selection.img_name.to_list() + additional_images)]
            tmp = selection_merged.drop_duplicates(subset=['pixel', 'sun_lon_bin'])
            num_sunlon_per_pixel_upd = tmp.groupby('pixel')['sun_lon_bin'].count()

            at_least_1 = len(num_sunlon_per_pixel_upd[num_sunlon_per_pixel_upd >= 1])
            at_least_2 = len(num_sunlon_per_pixel_upd[num_sunlon_per_pixel_upd >= 2])
            at_least_3 = len(num_sunlon_per_pixel_upd[num_sunlon_per_pixel_upd >= 3])
            print(f"sel 123: {at_least_1}, {at_least_2}, {at_least_3} "
                  f"(all123: {at_least_1_all}, {at_least_2_all}, {at_least_3_all})")

            # update metrics
            pixels_to_gain = total_pixels_good_sunlon_cov - at_least_3
            settled_pixels = num_sunlon_per_pixel_upd[num_sunlon_per_pixel_upd >= 3].index

            if max_images_to_add != None and len(additional_images) >= max_images_to_add:
                print(f"- Reached max additional images. Stop adding.")
                break

        print(f"- We exit the loop with {pixels_to_gain} pixels left uncovered. "
              f"We added {len(additional_images)} to the selected list {additional_images}.")


    final_selection = np.unique(selection.img_name.to_list() + additional_images)
    final_selection = rough_selection.loc[rough_selection.img_name.isin(final_selection)]

    selection_merged = all_merged.loc[all_merged.img_name.isin(final_selection.img_name)]
    tmp = selection_merged.drop_duplicates(subset=['pixel', 'sun_lon_bin'])
    num_sunlon_per_pixel_sel = tmp.groupby('pixel')['sun_lon_bin'].count()
    final_status = num_sunlon_per_pixel_sel[num_sunlon_per_pixel_sel >= 3].sort_values()
    print(final_status)
    print(len(final_status), len(all_covered_pixels))

    return final_selection


def improve_selection(selection, all_merged, rough_selection):

    total_num_pixels = all_merged.pixel.max()
    all_merged_pixels = all_merged.pixel.unique()

    total_illuminated_pixels = len(all_merged_pixels)
    selection_merged = all_merged.loc[all_merged.img_name.isin(selection.img_name)]

    selection_merged_pixels = selection_merged.pixel.unique()
    covered_pixels_selection = len(selection_merged_pixels)
    print("total/covered all/covered sel", total_num_pixels, total_illuminated_pixels, covered_pixels_selection)

    additional_images = []
    if covered_pixels_selection < total_illuminated_pixels:
        # initial missing pixels
        missing_pixels = [x for x in all_merged_pixels if x not in selection_merged_pixels]
        pixels_to_gain = len(missing_pixels)
        # print(pixels_to_gain, total_illuminated_pixels, 0.05 * total_illuminated_pixels)

        while pixels_to_gain > 0.005 * total_illuminated_pixels:
            print(f"- {pixels_to_gain}/{total_illuminated_pixels} "
                  f"({round(pixels_to_gain/total_illuminated_pixels*100.,1)}%) "
                  f"still up for grabs. Continue.")
            images_in_missing_pixels = all_merged.loc[all_merged.pixel.isin(missing_pixels)][['pixel', 'img_name']]
            num_pix_per_image = images_in_missing_pixels.groupby('img_name').count().sort_values(by='pixel')
            mbpb = num_pix_per_image[-1:].index.to_list()
            additional_images.extend(mbpb)
            print(f"- Adding {mbpb} to list of images...")
            gained_pixels = all_merged.loc[all_merged.img_name.isin(mbpb)]['pixel'].unique()
            gained_pixels = [x for x in gained_pixels if x in missing_pixels]
            # new list of missing pixels
            missing_pixels = [x for x in missing_pixels if x not in gained_pixels]
            pixels_to_gain = len(missing_pixels)
        print(f"- We exit the loop with {pixels_to_gain} pixels left uncovered. "
              f"We added {additional_images} to the selected list.")

    final_selection = np.unique(selection.img_name.to_list() + additional_images)
    final_selection = rough_selection.loc[rough_selection.img_name.isin(final_selection)]

    return final_selection

# print(sum(all_merged.groupby('pixel')['img_id'].count() > 3))
# print(sum(selection_merged.groupby('pixel')['img_id'].count() > 3))

if __name__ == '__main__':

    opt = SfsOpt.get_instance()

    all_merged = pd.read_parquet(f"{opt.rootdir}all_merged_0.parquet")
    all_merged = all_merged.drop_duplicates(subset=['pixel', 'img_name']).drop(columns=['band_data', 'geometry'])

    selection = pd.read_csv(f"{opt.rootdir}final_selection_0_sel0.csv", sep=',')
    rough_selection = pd.read_csv(f"{opt.rootdir}rough_selection_0.csv", sep=',')

    # additional_selection = improve_selection(selection=selection, all_merged=all_merged,
    #                                          rough_selection=rough_selection)
    final_selection = improve_selection_sunlon(selection=selection, all_merged=all_merged,
                                               rough_selection=rough_selection, max_images_to_add=15)
    print(final_selection)
