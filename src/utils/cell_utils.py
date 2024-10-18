import logging

import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sfs.config import SfsOpt

from utils.geopandas_utils import GeoSeries_to_GeoDataFrame
from utils.gridding import new_grid
from sfs.selection.optim_imgsel import optim_imgsel_kmedoids, join_to_pixels_check_overlay

def new_square_cell(center, size):
    """
    Generate square shape with side length around coordinates
    https://gis.stackexchange.com/questions/314949/creating-square-buffers-around-points-using-shapely
    Parameters
    ----------
    center: tuple
    size: float

    Returns
    -------
    GeoSeries, polygon
    """
    from shapely.geometry import Point

    opt = SfsOpt.get_instance()
    
    # Generate some sample data
    p1 = Point(center)
    points = gpd.GeoSeries([p1])

    # Buffer the points using a square cap style
    # Note cap_style: round = 1, flat = 2, square = 3
    buffer = points.buffer(size, cap_style=3)

    if opt.debug:
        print(buffer)
        # Plot the results
        fig, ax1 = plt.subplots()
        buffer.boundary.plot(ax=ax1, color='slategrey')
        points.plot(ax=ax1, color='red')
        plt.show()

    return buffer


def select_images(merged_cell, cell, crs_stereo, cells_to_process, min_overlay=0.1):

    opt = SfsOpt.get_instance()

    imgs_sel_per_group = []
    potent_img_per_group = []

    for idx, icell in tqdm(enumerate(merged_cell.groups), desc="img selection for cell "):

        if icell in cells_to_process:
            print("--> selecting images for cell #", icell)

            # select images belonging to the icell
            df_ = merged_cell.get_group(icell)
            df_ = gpd.overlay(df_, GeoSeries_to_GeoDataFrame(cell.loc[icell]), how='intersection')
                
            # Prepare "pixels" (sub-cells)
            # Dividing each cell in a 50x50 grid of "pixels" to check coverage by images
            pixels = new_grid(n_cells_per_side=opt.pixels_per_cell_per_side, bounds=cell.bounds.loc[icell].values)
            pixels = gpd.GeoDataFrame(pixels, columns=['geometry'],
                                      crs=crs_stereo)

            if opt.debug:
                print(pixels)
                ax = pixels.plot(facecolor="none", edgecolor='grey')
                plt.savefig(f"{opt.procroot}tst1{idx}.png")
                plt.show()
                # exit()

            # remove some specific image?
            try:
                if len(opt.imgs_to_remove[icell]) > 0:
                    df_ = df_.loc[~df_.img_name.isin(opt.imgs_to_remove[icell])]
            except:
                logging.warning(f"No key to remove specific images in imgs_to_remove for cell {icell}.")

            # remove images exceeding max set resolution on target area
            img_with_max_resol_per_cell = join_to_pixels_check_overlay(df_, pixels, min_overlay=min_overlay
                                                                       ).loc[df_['RESOLUTION'] <=
                                                                             opt.min_resolution]['img_name'].values
            # we are not really making any selection at this point anymore
            imgs_sel_per_group.append(img_with_max_resol_per_cell)
            potent_img_per_group.append(img_with_max_resol_per_cell)

        else:
            imgs_sel_per_group.append([])
            potent_img_per_group.append([])

    sel_per_tile = dict(zip(merged_cell.groups, imgs_sel_per_group))
    ioi_per_tile = dict(zip(merged_cell.groups, potent_img_per_group))

    return sel_per_tile, ioi_per_tile

if __name__ == '__main__':

    center = (159.45826, 97.18430)
    size = 1.
    new_square_cell(center, size)
