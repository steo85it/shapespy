import os

import fiona
import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, mapping
import geopandas as gpd
from sfs.config import SfsOpt

SfsOpt.display()
SfsOpt.set('local', False)
SfsOpt.set('site', "DM2_5m")
if SfsOpt.get('local') == False:
    SfsOpt.set('rootroot', f"{os.environ['HOME']}/nobackup/RING/code/sfs_helper/examples/HLS/{SfsOpt.get('site')}/")
    SfsOpt.set('datadir', f'{os.environ["HOME"]}/nobackup/RING/data/LROC/')
else:
    SfsOpt.set('rootroot', f"{os.environ['HOME']}/Lavoro/projects/HLS/{SfsOpt.get('site')}/")
    # SfsOpt.set('datadir', f'{os.environ["HOME"]}/nobackup/RING/data/LROC/')

SfsOpt.set('rootdir', f"{SfsOpt.get('rootroot')}root/")
SfsOpt.set('procroot', f"{SfsOpt.get('rootroot')}proc/")

os.makedirs(SfsOpt.get('rootdir'), exist_ok=True)

config_path = f"{SfsOpt.get('rootdir')}config.json"
if os.path.exists(config_path):
    optold = SfsOpt.from_json(config_path)
    to_exclude = optold.get("imgs_to_remove")
    print(to_exclude)
else:
    to_exclude = {}
    print("# No previous config file found.")

SfsOpt.set("crs_moon_lonlat", "+proj=lonlat +units=m +a=1737.4e3 +b=1737.4e3 +no_defs")
SfsOpt.set("crs_stereo", '+proj=stere +lat_0=-90 +lon_0=0 +lat_ts=-90 +k=1 +x_0=0 +y_0=0 +units=km +a=1737.4e3 '
                         '+b=1737.4e3 +no_defs')
SfsOpt.set("crs_stereo_meters",
           'PROJCS["Moon (2015) - Sphere / Ocentric / South Polar", BASEGEOGCRS["Moon (2015) - Sphere / Ocentric", DATUM["Moon (2015) - Sphere", ELLIPSOID["Moon (2015) - Sphere",1737400,0, LENGTHUNIT["metre",1]]], PRIMEM["Reference Meridian",0, ANGLEUNIT["degree",0.0174532925199433]], ID["IAU",30100,2015]], CONVERSION["South Polar", METHOD["Polar Stereographic (variant A)", ID["EPSG",9810]], PARAMETER["Latitude of natural origin",-90, ANGLEUNIT["degree",0.0174532925199433], ID["EPSG",8801]], PARAMETER["Longitude of natural origin",0, ANGLEUNIT["degree",0.0174532925199433], ID["EPSG",8802]], PARAMETER["Scale factor at natural origin",1, SCALEUNIT["unity",1], ID["EPSG",8805]], PARAMETER["False easting",0, LENGTHUNIT["metre",1], ID["EPSG",8806]], PARAMETER["False northing",0, LENGTHUNIT["metre",1], ID["EPSG",8807]]], CS[Cartesian,2], AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]], AXIS["(N)",north, ORDER[2], LENGTHUNIT["metre",1]], ID["IAU",30135,2015]]')
SfsOpt.set("prioridem_full", f"/explore/nobackup/people/mkbarker/GCD/grid/20mpp/v4/public/final/LDEM_83S_10MPP_ADJ.TIF")

SfsOpt.set("size_of_cells", 2.)
SfsOpt.set("pixels_per_cell_per_side", 20)

SfsOpt.check_consistency()

# config_path = f"{procdir}config.json"
SfsOpt.to_json(config_path)

SfsOpt.to_json(SfsOpt.get('config_file'))

from subdivide_polygon import split_polygon
crs_stereo = SfsOpt.get("crs_stereo")
crs_moon_lonlat = SfsOpt.get("crs_moon_lonlat")
rootdir = SfsOpt.get('rootdir')

def split_region(box=None, shpin=None, gtiffin=None):

    if box != None:
        print("-Reading bounds from manual box bounds.")
        strip_box = box
        strip_poly = Polygon(strip_box)
        strip_box = gpd.GeoDataFrame([strip_poly], columns=['geometry'])
    elif shpin != None:
        print("-Reading bounds from Shapefile.")
        strip_poly = gpd.read_file(shpin)
        strip_box = gpd.GeoDataFrame(geometry=strip_poly.geometry)
    elif gtiffin != None:
        print("-Reading bounds from GeoTiff.")
        ds = xr.open_dataset(gtiffin)
        minx, miny, maxx, maxy = [x * 1.e-3 for x in ds.rio.bounds()]
        print("minx, miny, maxx, maxy (we need km, or at least the same "
              "unit as size_of_cells)", minx, miny, maxx, maxy)
        strip_box = [(minx, maxy), (minx, miny), (maxx, miny), (maxx, maxy)]
        strip_poly = Polygon(strip_box)
        strip_box = gpd.GeoDataFrame([strip_poly], columns=['geometry'])
    strip_box.crs = crs_stereo
    print(strip_box)

    # fig, ax = plt.subplots()
    # strip_box.plot(ax=ax)
    # plt.show()

    print(f"- Splitting region into cells of side {SfsOpt.get('size_of_cells')}")
    squares = split_polygon(strip_box.geometry.values[0], shape='square', thresh=1.,
                            side_length=SfsOpt.get("size_of_cells"))
    cell = gpd.GeoDataFrame(squares, columns=['geometry'],
                            crs=crs_stereo)

    print(f"- Writing tiles to shapefile {rootdir}tiles.shp.")
    cell.to_file(f"{rootdir}tiles.shp")
    print(gpd.read_file(f"{rootdir}tiles.shp"))

    # plot annotated cells over region
    fig, ax1 = plt.subplots(figsize=(50, 50))
    plt.rc('font', size=5)

    cell_plt = cell.copy()
    cell_plt.reset_index().plot(column='index', cmap='viridis', edgecolor="grey", ax=ax1)
    cell_plt['coords'] = cell_plt['geometry'].apply(lambda x: x.representative_point().coords[:])
    cell_plt['coords'] = [coords[0] for coords in cell_plt['coords']]
    for idx, row in cell_plt.reset_index().iterrows():
        plt.annotate(xy=row['coords'],  # s=row['index'],
                     horizontalalignment='center', text=f"{idx}")
    plt.xlabel("x distance from NP, km")
    plt.ylabel("y distance from NP, km")

    for idx, polcir in enumerate([-60., -70., -80.]):
        polar_ = pd.DataFrame(np.arange(0, 360, 1), columns=['longitude'])
        polar_['latitude'] = polcir
        polar_ = gpd.GeoDataFrame([Polygon(polar_.values)], columns=['geometry'])
        polar_.crs = crs_moon_lonlat
        polar_ = polar_.to_crs(crs_stereo)
        polar_.plot(ax=ax1, color=['red', 'blue', 'green'][idx], alpha=0.5)

        # generate shp file for analysis in QGIS
        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'}
        }
        filename = f"{rootdir}polar_{polcir}.shp"
        print(polar_)
        with fiona.open(filename, 'w', 'ESRI Shapefile', schema, crs=crs_stereo) as c:
            ## If there are multiple geometries, put the "for" loop here
            c.write({
                'geometry': mapping(polar_.iloc[0].values[0]),
                'properties': {'id': 123},
                'crs': crs_stereo
            })
        print(f"- Polar circle {polcir} saved to shp")
    # exit()

    strip_box.plot(ax=ax1, alpha=0.5)
    # gdf_within.plot(ax=ax1, alpha=0.1, color='cyan')
    # psr_gdf.plot(ax=ax1, alpha=0.5, color='black')
    plt.savefig(f"{rootdir}ring_tiles_tracks.png")  # plot grid with latitude rings
    plt.savefig(f"{rootdir}ring_tiles_tracks.pdf")  # plot grid with latitude rings
    # plt.show()
    # exit()

if __name__ == '__main__':

    split_region(gtiffin=f"/panfs/ccds02/nobackup/people/sberton2/RING/dems/DM2_final_adj_5mpp_surf.tif")
    # split_region(shpin="/home/sberton2/Scaricati/test.shp")

    # polygon = ((148950, 101400), (159895, 101400), (159895, 93800), (148950, 93800), (148950, 101400)) # DM2b4
    # polygon = ((x*1e-3, y*1e-3) for x,y in polygon)
    # print(polygon)
    # split_region(box=polygon)
