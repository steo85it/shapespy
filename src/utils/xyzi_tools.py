import argparse
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from shapely.vectorized import contains

from tqdm import tqdm

try:
    from sfs.config import SfsOpt
    opt = SfsOpt.get_instance()
    crs_stereo_meters = opt.crs_stereo_meters
except:
    crs_stereo_meters = ('+proj=stere +lat_0=-90 +lon_0=0 +lat_ts=-90 +k=1 +x_0=0 +y_0=0 +units=m '
                         '+a=1737400 +b=1737400 +no_defs')

def convert_binary_to_parquet(binary_file_path, parquet_file_path, chunk_size=10 ** 6):
    dtype = np.dtype([('X', 'float'), ('Y', 'float'), ('Z', 'float'), ('I', 'float')])

    with open(binary_file_path, 'rb') as f:
        writer = None
        while True:
            chunk = np.fromfile(f, dtype=dtype, count=chunk_size)
            if chunk.size == 0:
                break

            df = pd.DataFrame(chunk)[['X', 'Y', 'Z']]
            table = pa.Table.from_pandas(df)

            if writer is None:
                writer = pq.ParquetWriter(parquet_file_path, table.schema)

            writer.write_table(table)

        if writer is not None:
            writer.close()


def read_parquet_and_filter(parquet_file_path, polygon_path, chunk_size=10**6, total_len=None, precision=3):

    # set proj not to complain
    # os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "True"

    polygon = gpd.read_file(polygon_path, engine='pyogrio')
    polygon = polygon.to_crs(crs_stereo_meters).geometry.iloc[0]

    print(polygon)

    filtered_data = []

    parquet_file = pq.ParquetFile(parquet_file_path)
    
    for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), desc='crop xyzi',
                      total=int(total_len/chunk_size)):
        df = batch.to_pandas()[['X', 'Y', 'Z']]
        # Create a rounded points array for filtering
        rounded_x = df['X'].round(precision).values
        rounded_y = df['Y'].round(precision).values
        # Use shapely's vectorized contains function
        mask = contains(polygon, rounded_x, rounded_y)
        filtered_chunk = df[mask]
        filtered_data.append(filtered_chunk)

    filtered_df = pd.concat(filtered_data, ignore_index=True)
    return filtered_df


def process_geospatial_data(input_file_path, output_file_path, crop_to):

    # Check the input file type and convert to Parquet if necessary
    if input_file_path.lower().endswith('.xyzi'):
        parquet_file_path = os.path.splitext(output_file_path)[0] + '_tmp.parquet'
        convert_binary_to_parquet(input_file_path, parquet_file_path)
    elif input_file_path.endswith('.parquet'):
        parquet_file_path = input_file_path
    else:
        raise ValueError("Unsupported file format. Please provide a .xyzi or .parquet file.")

    # Read and possibly crop the Parquet file
    if crop_to:
        len_parquet_in = len(pd.read_parquet(parquet_file_path)) # slow
        gdf = read_parquet_and_filter(parquet_file_path, crop_to, chunk_size=10**8, total_len=len_parquet_in)
    else:
        gdf = gpd.read_parquet(parquet_file_path)

    # Save the resulting GeoDataFrame to the specified output file path
    parquet_file_path = os.path.splitext(output_file_path)[0] + '.parquet'
    gdf.to_parquet(output_file_path)
    print(f"Processed data saved to {output_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Process and crop geospatial data.")
    parser.add_argument('--input_file_path', type=str, required=True,
                        help="Path to the input point clouds (meters, either .xyzi or .parquet).")
    parser.add_argument('--output_file_path', type=str, required=True,
                        help="Path to save the output Parquet file.")
    parser.add_argument('--crop_to', type=str,
                        help="Path to shp file with coordinates to crop to. If not provided, no cropping will be done.")

    args = parser.parse_args()
    process_geospatial_data(args.input_file_path, args.output_file_path, args.crop_to)


if __name__ == "__main__":
    main()

