import xarray as xr

import numpy as np
import geopandas as gpd

from sfs.config import SfsOpt

from asp.functions import set_asp
from isis.functions import set_isis

# aspdir = SfsOpt.get("aspdir")
# isisdir = SfsOpt.get("isisdir")
# isisdata = SfsOpt.get("isisdata")
# set_asp(aspdir)
# set_isis(isisdir, isisdata)

def rms(x):
    return np.linalg.norm(x) / np.sqrt(len(x.ravel()))

def get_tile_bounds(filin, tileid=None, extend=0.):

    if filin.split('.')[-1] == 'shp':
        # gdf = gpd.read_file(f"{opt.rootdir}clip_{tileid}.shp")
        gdf_ = gpd.read_file(filin)
        try:
            try:
                gdf = gdf_.loc[gdf_['FID'] == tileid]
            except:
                gdf = gdf_.loc[gdf_['id'] == tileid]
            #tile_side = np.sqrt(gdf.area * 1.e6)[tileid]
            tile_side = np.sqrt(gdf.area)[tileid]
        except:
            print("No index named FID or id found. Get full map.")
            gdf = gdf_.copy()
            #tile_side = np.sqrt(gdf.area * 1.e6)[0] # possible error
            tile_side = np.sqrt(gdf.area)[0] # possible error
        print(gdf)
        print(tile_side)
        minx, miny, maxx, maxy = gdf.total_bounds #* 1.e3

    elif filin.split('.')[-1] == 'tif':
        print("-Reading bounds from GeoTiff.")
        ds = xr.open_dataset(filin)  # already in meters
        minx, miny, maxx, maxy = [x for x in ds.rio.bounds()]
        if extend != 0.:
            assert maxy - miny == maxx - minx
            tile_side = maxy - miny
        print("minx, miny, maxx, maxy", minx, miny, maxx, maxy)

    else:
        print("** get_tile_bounds only accepts .shp or .tif input. Exit.")
        exit()

    # if extend != 0, add overlap
    if extend != 0.:
        minx = minx - tile_side * extend
        miny = miny - tile_side * extend
        maxx = maxx + tile_side * extend
        maxy = maxy + tile_side * extend

    return minx, miny, maxx, maxy


def outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return upper_limit, lower_limit
