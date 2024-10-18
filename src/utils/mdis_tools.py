import glob
import os

import geopandas as gpd
import pandas as pd
from shapely.lib import unary_union
from tqdm import tqdm

crs_lonlat = "+proj=lonlat +units=m +a=2440.0e3 +b=2440.0e3 +no_defs"

def gml2gdf(gml_files, outdir):

    # Dictionary to store GeoDataFrames categorized by prefix and latitude bins
    categorized_gdfs = {'EN': {}, 'EW': {}}
    categories_names = {'EN': 'NAC', 'EW': 'WAC'}

    for f in tqdm(gml_files, total=len(gml_files)):
        filename = os.path.basename(f).split('.')[0]
        img_prefix = filename[:2]  # Assuming 'EN' or 'EW' prefix
        gdf = gpd.read_file(f)

        gdf['geometry'] = gdf['geometry'].apply(lambda geom:
                                                unary_union(geom) if geom.geom_type == 'MultiPolygon' else geom)

        # Calculate center latitude for each image
        gdf['center_latitude'] = gdf.geometry.centroid.y

        # Bin center latitudes by 5 degrees
        gdf['lat_bin'] = (gdf['center_latitude'] // 5 * 5).astype(int)
        gdf['img_name'] = filename

        # Categorize by prefix and latitude bin
        for lat_bin, subset_gdf in gdf.groupby('lat_bin'):
            if lat_bin not in categorized_gdfs[img_prefix]:
                categorized_gdfs[img_prefix][lat_bin] = subset_gdf
            else:
                categorized_gdfs[img_prefix][lat_bin] = pd.concat(
                    [categorized_gdfs[img_prefix][lat_bin], subset_gdf])

    # Save each category to separate GeoJSON files
    for prefix in categorized_gdfs:
        for lat_bin in categorized_gdfs[prefix]:
            output_file = os.path.join(
                outdir, f'MDIS_{categories_names[prefix]}_{lat_bin}_{lat_bin+5}.geojson')
            gdf = categorized_gdfs[prefix][lat_bin].set_index('img_name').drop(
                                                    columns=['fid', 'ID', 'center_latitude', 'lat_bin'])
            gdf.to_file(output_file, driver='GeoJSON')



if __name__ == '__main__':

    gml_files = (glob.glob("/home/sberton2/Downloads/mercury_sp*/out/*.gml")+
                 glob.glob("/home/sberton2/Downloads/out/*.gml"))
    gml2gdf(gml_files, outdir="/home/sberton2/Lavoro/code/geotools/examples/sfs/Mercury/MSP/root/")
