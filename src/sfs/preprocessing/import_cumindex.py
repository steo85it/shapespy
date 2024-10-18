import logging
import re

import numpy as np
import pandas as pd

def pds3_to_df(rootdir,filnam,save_to=None,instrument='LROC'):

    try:
        lbl = rootdir + filnam + '.LBL'
        tab = rootdir + filnam + '.TAB'
        # open lbl file and extract "names" of all columns in tab file
        f = open(lbl)
    except:
        lbl = rootdir + filnam + '.lbl'
        tab = rootdir + filnam + '.tab'
        # open lbl file and extract "names" of all columns in tab file
        f = open(lbl)

    # since f is a generator, do not print/list it before this (else you'll get an empty list)
    if instrument == 'VOY':
        columns = [re.sub("\n", " ", x.split('=')[-1]).strip() for x in list(f) if 'OBJECT ' in x]
    else:
        columns = [re.sub("\"", " ", x.split('=')[-1]).strip() for x in list(f) if ' NAME ' in x]

    # print(len(columns))
    # additional column parsing for this stupid format
    if instrument == 'MDIS':
        print("- Reading MDIS index.")
        columns = columns[:-3] + [x + '_' + str(idx) for idx, x in enumerate(np.repeat(columns[-3:-2], 3))] + columns[
                                                                                                              -2:]
    elif instrument in ['LROC', 'VOY']:
        print("- Reading LROC or VOY index.")
        columns = columns[1:]

    else:
        print(f"* Unknown instrument {instrument}. Exit.")
        exit()

    print(columns)

    # import tab file to df, add columns from label
    df = pd.read_csv(tab, header=None, names=columns, index_col=False)

    # save to file
    if save_to != None:
        if save_to.split('.')[-1] == 'pkl':
            df.to_pickle(save_to)
        elif save_to.split('.')[-1] == 'parquet':
            df.to_parquet(save_to)
        else:
            print(f"* Unknown extension {save_to.split('.')[-1]}. "
                  f"Only .pkl and .parquet implemented. Exit.")
            exit()

    return df


def downsize(df, filnam, lat_bounds, lon_bounds=None, column_names=None):
    # print columns
    # print("Columns in imported df:")
    # print(df.columns)

    if column_names != None:
        ext_cols = column_names
        df = df.loc[:, ext_cols]
        try:
            df['nac_col'] = df['INSTRUMENT_ID'].values
        except:
            df['nac_col'] = df['ORIGINAL_PRODUCT_ID'].values
    elif filnam in 'CUMINDEX_MSGRMDS_1001':
        # select interesting columns
        # reticle_cols = [x for x in df.columns if ('RETICLE' in x) & ('ITUDE' in x)] # all N/A ...
        ext_cols = ['VOLUME_ID', 'PATH_NAME', 'FILE_NAME', 'HORIZONTAL_PIXEL_SCALE', 'SUB_SOLAR_LONGITUDE',
                    'INCIDENCE_ANGLE', 'EXPOSURE_DURATION', 'STANDARD_DEVIATION', 'SATURATED_PIXEL_COUNT',
                    'MISSING_PIXELS']  # +reticle_cols
        df['nac_col'] = df['INSTRUMENT_ID'].values
    elif filnam == 'INDEX_MESSDEM_1001':
        ext_cols = [x for x in df.columns if 'LATITUDE' in x]
        logging.debug(ext_cols)
    elif filnam == 'CUMINDEX_LROC':
        ext_cols = ['VOLUME_ID', 'FILE_SPECIFICATION_NAME', 'ORIGINAL_PRODUCT_ID', 'SUB_SOLAR_LONGITUDE', 'RESOLUTION',
                    'INCIDENCE_ANGLE', 'NAC_LINE_EXPOSURE_DURATION']  # +reticle_cols
        df['nac_col'] = df['ORIGINAL_PRODUCT_ID'].values
    else:
        ext_cols = df.columns

    pd.set_option('display.max_columns', None)
    
    # select images whose center latitude is within bounds (first rough selection)
    print(f"- pre downsize length: {len(df)}")

    if lon_bounds is not None:
        # Apply the filter
        selected_rows = (
                df['CENTER_LONGITUDE'].apply(lambda lon: is_within_longitude_range(lon, lon_bounds)) &
                (df['CENTER_LATITUDE'] <= lat_bounds[1]) &
                (df['CENTER_LATITUDE'] >= lat_bounds[0])
        )
    else:
        selected_rows = (df['CENTER_LATITUDE'] <= lat_bounds[1]) & (df['CENTER_LATITUDE'] >= lat_bounds[0])

    df = df.loc[selected_rows
                & (
                    df['nac_col'].str.contains('nac', case=False) |
                    df['nac_col'].str.contains('en', case=False) |
                    df['nac_col'].str.contains('ew', case=False)
                )
                , ext_cols]
    # print(df.sort_values(by='CENTER_LATITUDE'))
    print(f"- post downsize length: {len(df)}")
    assert len(df) > 0, "- No images remaining after downsize. Check longitudes range."

    return df.reset_index().rename({'index': 'orig_idx'}, axis=1)

def normalize_longitude_180(x):
# normalize to lon=[-180,180]

    return np.mod(x - 180.0, 360.0) - 180.0

def normalize_longitude_360(x):
    return np.mod(x, 360.0)


def is_within_longitude_range(lon, lon_bounds):
    lon = normalize_longitude_360(lon)
    lon_min = normalize_longitude_360(lon_bounds[0])
    lon_max = normalize_longitude_360(lon_bounds[1])

    # Check if lon_bounds span across the 0/360 boundary
    if lon_min <= lon_max:
        # No wraparound
        return lon_min <= lon <= lon_max
    else:
        # Wraparound case
        return lon >= lon_min or lon <= lon_max


if __name__ == '__main__':

    rootdir = '/home/sberton2/tmp/SP75/'
    filnam = 'CUMINDEX_MSGRMDS_1001' # 'INDEX_MESSDEM_1001' #
    import_data = False # True #

    # only needed once, to import index
    if import_data:
        df = pds3_to_df(rootdir,filnam)
    # just read pkl
    else:
        df = pd.read_pickle(rootdir+filnam+'.pkl')

    # print columns
    print("Columns in imported df:")
    print(df.columns)

    if filnam == 'CUMINDEX_MSGRMDS_1001':
        # select interesting columns
        reticle_cols = [x for x in df.columns if 'RETICLE' in x]
        ext_cols = ['INCIDENCE_ANGLE','SUB_SOLAR_AZIMUTH','HORIZONTAL_PIXEL_SCALE','INSTRUMENT_ID','CENTER_LATITUDE']
    elif filnam == 'INDEX_MESSDEM_1001':
        ext_cols = [x for x in df.columns if 'LATITUDE' in x]
        print(ext_cols)
    else:
        ext_cols = df.columns


    # print only interesting columns
    print(df.loc[:,ext_cols])
    pd.set_option('display.max_columns', None)

    # print(df.loc[:,reticle_cols])
    if filnam == 'INDEX_MESSDEM_1001':
        print(df.loc[df['MAXIMUM_LATITUDE']>80,['PRODUCT_ID']+ext_cols])
        print(df.loc[df['PRODUCT_ID'].str.contains('ASU'),['PRODUCT_ID','MAP_RESOLUTION', 'CENTER_LONGITUDE','WESTERNMOST_LONGITUDE', 'EASTERNMOST_LONGITUDE']+ext_cols].sort_values(by='MAXIMUM_LATITUDE'))
    elif filnam == 'CUMINDEX_MSGRMDS_1001':
        # reasonable images inside the ring
        selected_rows = (df['HORIZONTAL_PIXEL_SCALE'].between(0, 300))&(df['CENTER_LATITUDE']>=-90)&(df['CENTER_LATITUDE']<=-70)
        print(df.loc[selected_rows&(df['INSTRUMENT_ID']=='MDIS-WAC'),
                     ['PATH_NAME', 'FILE_NAME','CENTER_LONGITUDE','CENTER_LATITUDE','HORIZONTAL_PIXEL_SCALE']].sort_values(by="HORIZONTAL_PIXEL_SCALE"))

        # exit()

        import matplotlib.pyplot as plt
        fig = plt.figure()
        # print(df.loc[df['HORIZONTAL_PIXEL_SCALE'].between(0,500),['HORIZONTAL_PIXEL_SCALE','INSTRUMENT_ID']].sort_values(by='HORIZONTAL_PIXEL_SCALE'))
        # good to remove outliers
        # df['HORIZONTAL_PIXEL_SCALE'][df['HORIZONTAL_PIXEL_SCALE'].between(df['HORIZONTAL_PIXEL_SCALE'].quantile(.15), df['HORIZONTAL_PIXEL_SCALE'].quantile(.85))].hist()

        df.loc[selected_rows&(df['INSTRUMENT_ID']=='MDIS-WAC'), 'HORIZONTAL_PIXEL_SCALE'].hist(label='WAC')
        df.loc[selected_rows&(df['INSTRUMENT_ID']=='MDIS-NAC'), 'HORIZONTAL_PIXEL_SCALE'].hist(alpha=0.5,label='NAC')
        plt.legend()
        plt.xlabel('Resolution (mt/px)')
        plt.ylabel('Number of images')
        plt.savefig(rootdir+'histo_res_lat80N.png')
