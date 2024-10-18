import glob
import logging
import os

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, mapping
import colorcet as cc
from sfs.config import SfsOpt
from utils.cell_utils import new_square_cell, select_images
from utils.geopandas_utils import GeoSeries_to_GeoDataFrame  # , to_square
from sfs.preprocessing.import_cumindex import pds3_to_df, downsize


def preliminary_selection(input_images, box_half_side_km, cells_to_process,
                          center=None, input_shp=None, input_dem=None):
    """

    :param input_images:
    :param box_half_side_km:
    :param cells_to_process:
    :param center:
    :param input_shp: shapefile of the region
    :param input_dem: Raster/Geotiff covering the region, x,y should be in km
    :return:
    """

    opt = SfsOpt.get_instance()

    if not center is None:
        cell = new_square_cell(center, box_half_side_km)
        cell.crs = opt.crs_stereo
        cell = GeoSeries_to_GeoDataFrame(cell)

    elif (center is None) & (input_dem is not None):
        #print(xr.open_dataset(input_dem).rio.crs)
        left, bottom, right, top = [x*1e-3 for x in xr.open_dataset(input_dem).rio.bounds()]  # TODO --> for km (rather reproject to km crs)
        if box_half_side_km != None:
            center = (np.mean([left, right]), np.mean([top, bottom]))  # we want km
            cell = new_square_cell(center, box_half_side_km)
        else:
            cell = gpd.GeoSeries(shapely.box(left, bottom, right, top))
            print(cell)
        cell.crs = opt.crs_stereo
        cell = GeoSeries_to_GeoDataFrame(cell)

    # elif (center is None) & (input_shp is not None):
    elif input_shp is not None:
        try:
            cell = gpd.read_file(input_shp).set_index('id')
        except:
            cell = gpd.read_file(input_shp).set_index('FID')

        if len(cell) > 1:
            cell = cell.loc[list(cells_to_process.keys())]
        # cell.crs = opt.crs_stereo

    elif (center is None) & (input_dem is None) & (input_shp is None):
        print("** Center, input_shp and input_dem are all None. Exit.")
        exit()

    else:
        print(f"** What is happening here? Check {center}, {input_dem}, {input_shp}. Exit.")
        exit()

    # reindex input cell to requested cell (weird fix ... but it works)
    cell['new_index'] = list(cells_to_process.keys())
    cell = cell.set_index('new_index')

    print(cell.values)
    print(gpd.GeoDataFrame(input_images))
    
    # only select images within the cell
    gdf_within = gpd.overlay(gpd.GeoDataFrame(input_images), cell, how='intersection')
    logging.info(f"Total/overlapping images: {len(input_images)}, {len(gdf_within)}")

    if opt.debug:
        print(gdf_within)
        save_to_path = f'{opt.rootdir}gdf_within.json'
        gpd.GeoSeries(gdf_within.geometry, crs=opt.crs_stereo).to_file(save_to_path)
        logging.debug(f"Saved gdf_within.json to {save_to_path}.")

    # assign solar longitude range
    gdf_within['sol_lon'] = (gdf_within['SUB_SOLAR_LONGITUDE'].values / opt.lon_step).astype(int) * opt.lon_step
    gdf_within.crs = opt.crs_stereo

    # assign to each cell
    merged = gpd.sjoin(gdf_within, cell, how='left', op='intersects').rename({'index_right': 'cell'}, axis=1)
    merged = merged.dropna(axis=0).astype({'cell': 'uint'})
    merged['img_name'] = merged['ORIGINAL_PRODUCT_ID'].values
    merged = merged.reset_index().rename({'index': 'img_id'}, axis=1)
    # print(merged)

    # prepare groups
    merged_cell = merged.groupby('cell')

    # recheck images overlapping with tile
    sel_per_tiles, pot_per_tiles = select_images(merged_cell=merged_cell, cell=cell,
                                                 crs_stereo=opt.crs_stereo,
                                                 cells_to_process=list(cells_to_process.keys()),
                                                 min_overlay=0.01)
    # remove empty cells from dict before generating output selection
    sel_per_tiles = {k: v for k, v in sel_per_tiles.items() if v is not None}
    print(sel_per_tiles)
    print(pot_per_tiles)
    # else:
    #     sel_per_tiles = {}
    #     pot_per_tiles = {}
    #     for idx, icell in enumerate(merged_cell.groups):
    #         sel_per_tiles[icell] = merged_cell.get_group(icell)
    #         pot_per_tiles[icell] = merged_cell.get_group(icell)

    # check solar longitude + resolution coverage
    selout_list = []
    for icell, imgs in sel_per_tiles.items():
        if icell in list(cells_to_process.keys()):

            # get df with all images, with first N imgs coming from selection, the rest just for bundle-adjustment
            sel_df = merged_cell.get_group(icell).loc[merged_cell.get_group(icell).img_name.isin(
                sel_per_tiles[icell])][['FILE_SPECIFICATION_NAME', 'img_name', 'sol_lon', 'START_TIME',
                                        'RESOLUTION', 'INCIDENCE_ANGLE', 'SUB_SOLAR_LONGITUDE']]
            pot_df = merged_cell.get_group(icell).loc[merged_cell.get_group(icell).img_name.isin(
                pot_per_tiles[icell])][['FILE_SPECIFICATION_NAME', 'img_name', 'sol_lon', 'START_TIME',
                                        'RESOLUTION', 'INCIDENCE_ANGLE', 'SUB_SOLAR_LONGITUDE']]

            # print(f"selection including additional images for bundle_adjust (cell {icell}):", len(pot_df))
            #
            # # remove sel_df IMG from pot_df
            # img_already_sel = pd.merge(pot_df, sel_df, how='inner', on=['img_name'])['img_name']
            # pot_df = pot_df[~pot_df['img_name'].isin(img_already_sel)]
            # print("images for bundle_adjust (after removing main selection):", len(pot_df))
            #
            # cos_sol_lon = np.cos(np.deg2rad(sel_df[['sol_lon']].values))
            # print("mean/std of cos(lon) of selection =", np.mean(cos_sol_lon), np.std(cos_sol_lon))

            # Only tot_df matters at this stage!!
            tot_df = np.vstack([sel_df.values, pot_df.values])
            tot_df = pd.DataFrame(tot_df, columns=sel_df.columns).drop_duplicates().drop('img_name', axis=1)

            # just print the actual image name (no path)
            tot_df.loc[:, 'img_name'] = (
                tot_df.apply(lambda x: x['FILE_SPECIFICATION_NAME'].split('/')[-1].split('.')[0], axis=1))

            # save start_time with MB format
            tot_df.loc[:, 'epoch_str'] = pd.to_datetime(tot_df['START_TIME'])
            tot_df['epoch_str'] = tot_df['epoch_str'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
            # template 2012-05-04T13:51:30.925

            if str(icell) in opt.imgs_to_remove:
                if len(opt.imgs_to_remove[str(icell)]) > 0:
                    tot_df_clean = tot_df.loc[~tot_df.img_name.isin(opt.imgs_to_remove[str(icell)])]
                    logging.info(f'- Removed {len(tot_df) - len(tot_df_clean)} based on "to_exclude" list.')
                    tot_df = tot_df_clean.copy()

            #selout_path = f"{opt.rootdir}lnac_{cells_to_process[icell]}.in"
            print(f"- Saving total selection of {len(tot_df)} images to {opt.rootroot}{opt.imglist_full}.")
            tot_df.to_csv(opt.rootroot+opt.imglist_full, index=None)

            # write to shapefile (if not already provided as input)
            if input_shp == None: # or not os.path.exists(f"{opt.rootdir}clip_{cells_to_process[icell]}.shp"):
                filename = f"{opt.rootdir}clip_{cells_to_process[icell]}.shp"
                cell['id'] = cell.index.values
                cell = cell.set_index('id')
                cell.to_file(filename, crs=opt.crs_stereo)
                logging.info(f"## Tile polygon ESRI shapefile written to {filename}")

            #merged = merged_cell.get_group(icell)
            #cut_ = gpd.overlay(merged, GeoSeries_to_GeoDataFrame(cell.loc[icell]), how='intersection')
            # prepare movie
            #gb_ = cut_.groupby('sol_lon')

            # plot final selection and coverage
            #fig, ax = plt.subplots()
            #if len(imgs) > opt.nimg_to_select:
            #    print("### Too many sel_imgs:", len(imgs), ". Probably some issue!!!")
            #selimg = cut_.loc[cut_['img_name'].isin(imgs)]

            #if opt.debug:
                # write to shapefile
                # Define a polygon feature geometry with one attribute
            #    schema = {
            #        'geometry': 'Polygon',
            #        'properties': {'id': 'int'}
            #    }
            #    filename = f"{opt.rootdir}cov_{cells_to_process[icell]}.shp"
                # print(cell.loc[icell].values[0])
            #    with fiona.open(filename, 'w', 'ESRI Shapefile', schema, crs=opt.crs_stereo) as c:
                    ## If there are multiple geometries, put the "for" loop here
            #        for img in selimg:
                        #print(img)
            #            c.write({
            #                'geometry': mapping(img.values[0]),
            #                'properties': {'id': 123},
            #                'crs': opt.crs_stereo
            #            })
            #    logging.info("## Tile polygon ESRI shapefile written to", filename)
            #    exit()

            #selimg.plot(column='sol_lon', cmap=cc.cm.colorwheel, alpha=0.2,
            #            edgecolor="grey", ax=ax, vmin=0, vmax=359)
            #scatter = ax.collections[-1]
            #plt.colorbar(scatter, ax=ax, extend='min')  # , vmin=150, vmax=350)
            #plt.title(f"Cell coverage with {len(imgs)} images.")
            #plt.ylabel("y, km from NP")
            #plt.xlabel("x, km from NP")
            #plt.savefig(f"{opt.rootdir}covtot_{cells_to_process[icell]}.png")

            selout_list.append(opt.rootroot+opt.imglist_full)

    return selout_list
            
# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    opt = SfsOpt.get_instance()
    
    # select list of sites
    sites = glob.glob(f"{opt.rootroot}../artemis_sites/*.tif")
    print(sites)

    # prepare some useful dirs
    os.makedirs(f'{opt.procroot}', exist_ok=True)
    os.makedirs(f'{opt.procroot}plt', exist_ok=True)

    # Read and prepare images
    # ----------------------------------

    # only needed once, to import index
    if opt.import_index:
        df = pds3_to_df(opt.rootdir, opt.pds_index_name)
    # just read pkl
    else:
        try:
            df = pd.read_parquet(opt.rootdir + opt.pds_index_name + '.parquet')
        except:
            df = pd.read_pickle(opt.rootdir + opt.pds_index_name + '.pkl')

    img_poly = ['UPPER_RIGHT_LATITUDE', 'UPPER_RIGHT_LONGITUDE', 'LOWER_RIGHT_LATITUDE',
                'LOWER_RIGHT_LONGITUDE', 'LOWER_LEFT_LATITUDE', 'LOWER_LEFT_LONGITUDE',
                'UPPER_LEFT_LATITUDE', 'UPPER_LEFT_LONGITUDE', 'CENTER_LATITUDE']

    # update lat_bounds w.r.t. where the blocks to recover are (else full grid takes too long...)
    df = downsize(df[:], filnam=opt.pds_index_name, lat_bounds=[-90, -70],
                  column_names=['VOLUME_ID', 'FILE_SPECIFICATION_NAME',
                                'START_TIME',
                                'SUB_SOLAR_LONGITUDE', 'RESOLUTION',
                                'EMISSION_ANGLE',
                                'DATA_QUALITY_ID', 'SLEW_ANGLE',
                                'ORIGINAL_PRODUCT_ID',
                                'INCIDENCE_ANGLE', 'IMAGE_LINES'
                                'NAC_LINE_EXPOSURE_DURATION'] + img_poly)

    print("post downsize", len(df))
    
    # normalize to lon=[-180,180]
    df[['UPPER_RIGHT_LONGITUDE', 'LOWER_LEFT_LONGITUDE', 'LOWER_RIGHT_LONGITUDE', 'UPPER_LEFT_LONGITUDE']] = \
        df[['UPPER_RIGHT_LONGITUDE', 'LOWER_LEFT_LONGITUDE', 'LOWER_RIGHT_LONGITUDE', 'UPPER_LEFT_LONGITUDE']].apply(
            lambda x: np.mod(x - 180.0, 360.0) - 180.0)
    # transform to (lon, lat) tuple demanded by shapely
    df['UR'] = df[['UPPER_RIGHT_LONGITUDE', 'UPPER_RIGHT_LATITUDE']].astype('float').apply(tuple, axis=1)
    df['LR'] = df[['LOWER_RIGHT_LONGITUDE', 'LOWER_RIGHT_LATITUDE']].astype('float').apply(tuple, axis=1)
    df['LL'] = df[['LOWER_LEFT_LONGITUDE', 'LOWER_LEFT_LATITUDE']].astype('float').apply(tuple, axis=1)
    df['UL'] = df[['UPPER_LEFT_LONGITUDE', 'UPPER_LEFT_LATITUDE']].astype('float').apply(tuple, axis=1)
    # generate geodf of polygons in lonlat and convert to stereo
    gpol = gpd.GeoDataFrame(df.loc[:, ['LL', 'LR', 'UR', 'UL']].apply(lambda x: Polygon(x), axis=1),
                            columns=['geometry'])
    gpol.crs = opt.crs_lonlat
    gpol_stereo = gpol.to_crs(opt.crs_stereo)
    # print(gpol_stereo)

    # add image polygons to other info from label
    input_images = pd.merge(df, gpol_stereo, left_index=True, right_index=True)
    print("post merge", len(input_images))

    
    # set selection criteria to get "good obs"
    select_criteria = ((input_images['RESOLUTION'] < opt.min_resolution) & (input_images['RESOLUTION'] > 0.)
                       & (input_images['SLEW_ANGLE'].abs() <= 5)
                       & (90. - input_images['INCIDENCE_ANGLE'] > 2.)
                       & (input_images['NAC_LINE_EXPOSURE_DURATION'] < 5e-3))

    if opt.debug:
        # plot histo of available images to set selection criteria
        axs = input_images[['RESOLUTION', 'SUB_SOLAR_LONGITUDE', 'DATA_QUALITY_ID', 'SLEW_ANGLE',
                            'FILE_SPECIFICATION_NAME', 'NAC_LINE_EXPOSURE_DURATION',
                            'INCIDENCE_ANGLE', 'EMISSION_ANGLE', 'IMAGE_LINES']].hist(figsize=(15, 10), layout=(3, 3))
        plt.tight_layout()
        filnam = f"{opt.procroot}img_hist.pdf"
        plt.savefig(filnam)
        logging.info(f"Histogram of selected images saved in {filnam}")

    # got subselection of images
    input_images = input_images.loc[select_criteria]
    if len(input_images) > 0:
        print(f"- We enter selection with {len(input_images)} images... going on!!!")
    else:
        print(f"- We found 0 images ... something is wrong. Stop.")
        exit()
        
    # loop through the sites
    for input_dem in sites:
        print(f"- Processing {input_dem.split('/')[-1]}...")
        boxid = input_dem.split('/')[-1].split('_')[0]
        preliminary_selection(input_dem, input_images=input_images, box_half_side_km=4, cells_to_process={0: boxid})
