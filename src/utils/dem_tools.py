import xarray as xr
import rioxarray
from rasterio._io import Resampling


def smooth_dem(sfs_da, max_tr, min_tr, sfs_map_coarse_fine, interp_fun=Resampling.cubic_spline, shift=(None, None)):

    # if same in/out resolution, skip
    if max_tr == min_tr:
        sfs_da.rio.to_raster(sfs_map_coarse_fine)
        return sfs_map_coarse_fine
    
    assert max_tr > min_tr, 'max_tr must be greater than min_tr'

    if shift[0] is not None:
        sfs_da.coords['x'] = sfs_da.coords['x'] + shift[0]
        sfs_da.coords['y'] = sfs_da.coords['y'] + shift[1]

    # # downsample and upsample SFS:
    da = sfs_da.rio.reproject(dst_crs=sfs_da.rio.crs, resampling=interp_fun, resolution=max_tr)
    da = da.rio.reproject(dst_crs=da.rio.crs, resampling=interp_fun, resolution=min_tr)
    da.rio.to_raster(sfs_map_coarse_fine)
    
    return sfs_map_coarse_fine
