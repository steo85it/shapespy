import logging
from matplotlib import pyplot as plt
import os
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point

from sfs.config import SfsOpt
from sfs.preprocessing.preprocessing import preprocess_mdis
from sfs.preprocessing.get_lnac_from_raster import preliminary_selection
from sfs.preprocessing.import_cumindex import downsize

def rough_imgsel_sites(tileid, boxcentersfil=None, input_shp=None, input_tif=None, lonlat=False, latrange=[-90,-60]):

    opt = SfsOpt.get_instance()

    # read file with center of cells and make all "odd" at meter level
    if boxcentersfil != None:
        box_list = pd.read_csv(boxcentersfil,
                               sep="\s+", header=None, index_col=0)
        box_list = box_list.iloc[tileid:tileid+1, :]
        if lonlat:
            print(box_list)
            lonc = box_list.iloc[:, 1].values
            latc = box_list.iloc[:, 0].values
            box_list = gpd.GeoSeries(Point([lonc, latc]), crs=opt.crs_lonlat)
            box_list = box_list.to_crs(opt.crs_stereo)
            box_list = pd.DataFrame(np.vstack([(x, y) for x, y in zip(box_list.x, box_list.y)]), columns=['xc', 'yc'])
        else:
            box_list.columns = ["x", "y"]
            box_list *= 1.e3
            box_list = box_list.astype('int')
            box_list['xc'] = np.where(box_list['x'] % 2 == 0, box_list['x']+1, box_list['x'])
            box_list['yc'] = np.where(box_list['y'] % 2 == 0, box_list['y']+1, box_list['y'])
            box_list /= 1.e3
            box_list = box_list.drop(['x', 'y'], axis='columns')
        print(f"- Got center coordinates from file.")
    elif input_shp != None:
        print(f"- Taking input area from {input_shp}.")
    elif input_tif != None:
        print(f"- Taking input area from {input_tif}.")
    else:
        print("** get_lnac_single_cell.py: Both boxcenterfil and input_shp are None. Exit.")
        exit()

    # print(f"- Getting images in a box around stereo coordinates {box_center} km from SP.")

    # prepare some useful dirs
    os.makedirs(f'{opt.procroot}', exist_ok=True)
    os.makedirs(f'{opt.procroot}plt', exist_ok=True)

    # Read and prepare images
    #----------------------------------
    df = pd.read_parquet(opt.rootdir + opt.pds_index_name + '.parquet')

    columns_subset = ['VOLUME_ID', 'FILE_SPECIFICATION_NAME',
                     'START_TIME',
                     'SUB_SOLAR_LONGITUDE', 'RESOLUTION',
                     'EMISSION_ANGLE',
                     'DATA_QUALITY_ID', 'SLEW_ANGLE',
                     'ORIGINAL_PRODUCT_ID',
                     'INCIDENCE_ANGLE',
                     'NAC_LINE_EXPOSURE_DURATION', 'CENTER_LATITUDE']

    if opt.calibrate == 'lrocnac':
        img_poly = ['UPPER_RIGHT_LATITUDE', 'UPPER_RIGHT_LONGITUDE', 'LOWER_RIGHT_LATITUDE',
                    'LOWER_RIGHT_LONGITUDE', 'LOWER_LEFT_LATITUDE', 'LOWER_LEFT_LONGITUDE',
                    'UPPER_LEFT_LATITUDE', 'UPPER_LEFT_LONGITUDE', 'DATA_QUALITY_ID']
        columns_subset = columns_subset + img_poly
    elif opt.calibrate[:4] == 'mdis':
        df = preprocess_mdis(mdis_index_df=df, geojson_file_path=opt.rootdir,
                             nacwac=str.upper(opt.calibrate[4:]), latrange=latrange)
        columns_subset = columns_subset + ['geometry']
    else:
        logging.error("** get_lnac_single_cell.py: Unknown calibrate option.")
        exit()

    # downsize by columns and central latitude
    df = downsize(df[:], filnam=opt.pds_index_name, lat_bounds=latrange, column_names=columns_subset)
    print("len post downsize", len(df))
    
    if opt.calibrate == 'lrocnac':
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
                                columns=['geometry'], crs=opt.crs_lonlat)
        # add image polygons to other info from label
        df = pd.merge(df, gpol, left_index=True, right_index=True)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        input_images = gpd.GeoDataFrame(df, geometry='geometry', crs=opt.crs_lonlat)
    elif opt.calibrate[:4] == 'mdis':
        input_images = gpd.GeoDataFrame(df, geometry='geometry', crs=opt.crs_lonlat)
    else:
        logging.error("** get_lnac_single_cell.py: Unknown calibrate option.")
        exit()

    # bring to stereo crs
    input_images = input_images.to_crs(opt.crs_stereo)

    # set selection criteria to get "good obs"
    select_criteria = ((input_images['RESOLUTION'] < opt.min_resolution) & (input_images['RESOLUTION'] > 0.)
                       & (input_images['SLEW_ANGLE'].abs() <= (5 if opt.calibrate == 'lrocnac' else 45))
                       & (input_images['DATA_QUALITY_ID'] == 0)
                       # & (90. - input_images['INCIDENCE_ANGLE'] > 0.) # incidence angle check that sometimes needs to be commented out
                       # & (input_images['NAC_LINE_EXPOSURE_DURATION'] < 5e-3)
                       )

    if opt.debug:
        # save shapefile of images
        print(input_images)
        input_images.geometry.to_file(f'{opt.rootdir}gpol_debug.geojson', driver='GeoJSON')

        # plot histo of available images to set selection criteria
        axs = input_images[['RESOLUTION', 'SUB_SOLAR_LONGITUDE', 'DATA_QUALITY_ID', 'SLEW_ANGLE',
                        'FILE_SPECIFICATION_NAME', 'NAC_LINE_EXPOSURE_DURATION',
                        'INCIDENCE_ANGLE', 'EMISSION_ANGLE']].hist(figsize=(15, 10), layout=(3, 3))
        plt.tight_layout()
        filnam = f"{opt.procroot}img_hist.pdf"
        plt.savefig(filnam)
        logging.info(f"Histogram of selected images saved in {filnam}")

    # got subselection of images
    input_images = input_images.loc[select_criteria]
    if len(input_images) > 0:
        logging.info(f"- We enter selection with {len(input_images)} images... going on!!!")
    else:
        logging.error("*** We ended up with 0 images. Something is wrong, stop!")
        exit()

    if opt.get('size_of_cells') != None:
        box_half_side_km = opt.get('size_of_cells')/2.
    else:
        box_half_side_km = None

    # loop through the sites
    if (input_shp == None) & (input_tif == None):
        for row in box_list.iterrows():
            box_center = row[1].values
            print(f"- Getting images in a {box_half_side_km*2.}x{box_half_side_km*2.} "
               f"box around stereo coordinates {box_center} km from SP "
               f"({opt.get('site')[:2]}{row[0]}).")
            selout_list = preliminary_selection(center=box_center, input_images=input_images,
                               box_half_side_km=box_half_side_km,
                               cells_to_process={tileid: f"{opt.get('site')[:2]}{tileid}"})
    elif input_shp != None:
        print(f"- Getting images in a "
              f"box defined by {input_shp}.")
        selout_list = preliminary_selection(input_shp=input_shp, input_images=input_images,
                          box_half_side_km=box_half_side_km,
                          cells_to_process={tileid: f"{opt.get('site')[:2]}{tileid}"})

    elif input_tif != None:
        selout_list = preliminary_selection(input_dem=input_tif, input_images=input_images,
                          box_half_side_km=box_half_side_km,
                          cells_to_process={tileid: f"{opt.get('site')[:2]}{tileid}"})
    else:
        print(f"* Missing input region. Stop.")
        exit()

    return selout_list
