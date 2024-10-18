import glob
import logging
import os
import shutil
from functools import partial
from tqdm import tqdm

import numpy as np
from xrspatial import slope
from p_tqdm import p_umap

import xarray as xr
from matplotlib import pyplot as plt
import pandas as pd
from asp.functions import mapproject, dem_mosaic, set_asp, hillshade, gdal_translate
from isis.functions import set_isis
from sfs.alignment.align_from_render import render_adjusted
from sfs.config import SfsOpt
from shadowspy.image_util import read_img_properties
from utils.mitsuta import angrange
from rasterio.enums import Resampling
from sfs.postprocessing.postpro_validation import postpro_val, align_mapproj

def products_overview(list_of_prods, outpng, nrows=4, ncols=3):

    # Create a figure with 4 rows and 3 columns of subplots
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))

    for i, ax in tqdm(enumerate(axes.flatten()), total=len(list_of_prods)):

        if i >= len(list_of_prods):
            break
        prod_name = list_of_prods[i].split('/')[-1].split('.')[0]
        # Load the raster xarray
        raster = xr.open_dataarray(list_of_prods[i])
        # Plot the raster using xarray's plot method
        im = raster.plot(ax=ax, add_colorbar=False, robust=True)  # Set robust=True for automatic outlier clipping
        # Create a colorbar for each subplot
        plt.colorbar(im, ax=ax)
        # Set the title for each subplot
        ax.set_title(f'{prod_name}')
        # Set x and y labels
        ax.set_xlabel('X (m)') # was originally km -- a mistake?
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')

    # Adjust layout
    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig(outpng)  # Replace with your desired file path
    # Close the plot to free up memory
    plt.close(fig)


def postpro(tileid, sel, final_selection, parallel=True, shadow_threshold=0.002, use_rough_sel=False, use_priori_dem=False):
    """
    reproject aligned images and get max illumination mosaic

    Parameters
    ----------
    tileid int
    sel int
    final_selection
    """

    opt = SfsOpt.get_instance()

    set_asp(opt.aspdir)
    set_isis(opt.isisdir, opt.isisdata)

    procdir = f"{opt.procroot}tile_{tileid}/"
    seldir = f"{procdir}sel_{sel}/"
    sfsdir = opt.get('sfsdir')

    prodsdir = f"{seldir}products/" # changed filepath
    os.makedirs(prodsdir, exist_ok=True)
    siteid = f"{opt.get('site')[:2]}{tileid:02d}"

    quicklook = f"{prodsdir}quicklook/"
    os.makedirs(f"{prodsdir}quicklook", exist_ok=True)

    # prepare map of vertical differences
    def vdiff_map():
        print("- Read ldem and compute vertical differences.")

        if opt.calibrate == 'lrocnac':
            ldec = "/explore/nobackup/people/mkbarker/GCD/grid/20mpp/v4/newLDEMs_10032022/ldem_75s_final_adj_30mpp_ldec.tif"
        else:
            ldec = None
        ldem = f"{procdir}ldem_{tileid}_{opt.targetmpp}mpp.tif"

        raster1_ds = xr.open_dataset(ldem)
        # add exception for weird DEMs
        try:
            print(raster1_ds.band_data)
        except:
            raster1_ds['band_data'] = raster1_ds['z']
        
        raster2_ds = xr.open_dataset(f"{sfsdir}run-DEM-nodata-final.tif", masked=True)
        raster2_ds.band_data.rio.to_raster(f"{seldir}sfsdem_nodata_{tileid}.tif")
        raster2_ds = raster2_ds.rio.reproject_match(raster1_ds, Resampling=Resampling.cubic_spline)
        # print("- both xarrays have been matched to same resolution and grid")

        diff = raster2_ds - raster1_ds
        # print(f"- differences computed {ta} vs {tb}")

        # crop boundary issues
        diff = diff.dropna(dim='x', how='all')
        diff = diff.dropna(dim='y', how='all')
        #
        diff.band_data.rio.to_raster(f"{seldir}vdiff_ldem_{tileid}_sel{sel}.tif")
        #
        fig, axes = plt.subplots(1, 2)
        diff.band_data.plot(ax=axes[0], robust=True)
        xr.plot.hist(diff.band_data, ax=axes[1])
        plt.savefig(f"{quicklook}vdiff_ldem_{tileid}.png")
        # plt.show()
        plt.clf()

        # apply ldec mask
        if ldec != None:
            mask = xr.open_dataset(ldec)
            mask = mask.rio.reproject_match(diff, Resampling=Resampling.cubic_spline)

            if opt.debug:
                mask.z.plot(robust=True)
                plt.show()
            diff_masked = diff.where(mask.z > 0)

            diff_masked.band_data.rio.to_raster(f"{seldir}vdiff_ldem_masked_{tileid}_sel{sel}.tif")

            fig, axes = plt.subplots(1, 2)
            diff_masked.band_data.plot(ax=axes[0], robust=True)
            xr.plot.hist(diff_masked.band_data, ax=axes[1])
            plt.savefig(f"{quicklook}vdiff_ldem_masked_{tileid}_sel{sel}.png")
            # plt.show()
            plt.clf()

            print("std full/masked", diff.band_data.std(), diff_masked.band_data.std())


    def diff_to_nac():

        render_post_dir = f"{seldir}render_post/"
        os.makedirs(render_post_dir, exist_ok=True)

        # parallel render and compare to NAC
        measproj_paths = [f"{seldir}prj/ba_align/{img}_map.tif" for img in final_selection.img_name.values]

        # get simulated image
        compproj_paths = render_adjusted(measproj_paths, pdir=seldir, dem_path=f"{sfsdir}run-DEM-final.tif",
                                         indir=render_post_dir)
        # compproj_paths = glob.glob(f'{procdir}render/out/M1*E_*.tif')

        diffs = []
        pixel_differences_rms = {}
        for idx, img in tqdm(enumerate(final_selection.img_name.values), total=len(final_selection)):

            # meas = f"{procdir}sfs_ia/run-{img}-meas-intensity.tif"
            # comp = f"{procdir}sfs_ia/run-{img}-comp-intensity.tif"
            meas = [x for x in measproj_paths if (img in x) and ("_sub" not in x)]
            comp = [x for x in compproj_paths if (img in x) and ("_sub" not in x)]

            assert (len(meas) == 1) and (len(comp) == 1)
            meas = meas[0]
            comp = comp[0]

            meas_ds = xr.open_dataset(meas)
            comp_ds = xr.open_dataset(comp)

            # save raster with both computed and measured?
            meascomp_ds = xr.merge([meas_ds.rename({'band_data': 'meas_rad'}),
                                    comp_ds.rename({'band_data': 'comp_rad'}).where(comp_ds.band_data > 0)])
            meascomp_ds = meascomp_ds.fillna(-99999)
            meascomp_ds.sel({'band': 1}).rio.to_raster(f"{render_post_dir}run-{img}-meascomp-intensity.tif")

            diff = meas_ds - comp_ds
            diff_rms = np.sqrt(np.nanmean(diff.band_data.values)**2 + np.nanstd(diff.band_data.values)**2)
            pixel_differences_rms[img] = diff_rms

            # maybe useful for aggregated map of pixel differences
            # diff_mean = xr.concat(diffs, dim='band').mean(dim='band')
            # diff_std = xr.concat(diffs, dim='band').std(dim='band')
            # diff_rms = (diff_mean.map(np.square) + diff_std.map(np.square)).map(np.sqrt)

            # save figure and raster of differences
            diff.band_data.plot(robust=True)
            plt.savefig(f"{render_post_dir}render_vs_nac_{img}.png")
            plt.clf()
            diff.band_data.rio.to_raster(f"{render_post_dir}render_vs_nac_{img}.tif")
            # move to new postpro folder
            # shutil.move(f"{procdir}sfs_ia/", f"{procdir}sfs_ia_post/")

            # get list of pixel differences among pairs
            diffs.append(diff)

        logging.info(f"- {len(measproj_paths)} images rendered and comparisons saved to {render_post_dir}.")

        return pixel_differences_rms, diffs


    num_threads = 1
    def aligned_maxlit():
        # reproject aligned images and produce max_lit mosaic
        def load_project_img(idxrow, bundle_adjust_prefix=None):
            idx, row = idxrow
            img = row.img_name

            opt = SfsOpt.get_instance()
            
            set_asp(opt.aspdir)
            set_isis(opt.isisdir, opt.isisdata)

            if use_priori_dem:
                #dem = f"{procdir}/ldem_{tileid}.tif"
                dem = f"{procdir}/ldem_{tileid}_fullA3.tif"
            else:
                dem = f"{sfsdir}run-DEM-final.tif"

            print(f"- Mapprojecting {img}.IMG...")
            # print(bundle_adjust_prefix, os.path.exists(prj_img_path), prj_img_path)

            if bundle_adjust_prefix == None:
                prj_img_path = f"{procdir}prj/{opt.targetmpp}mpp/"
                os.makedirs(prj_img_path, exist_ok=True)
                prj_img_path += f"{img}_map.tif"

                if not os.path.exists(prj_img_path):
                    mapproject(from_=f"{procdir}{img}.cub", to=prj_img_path,
                               dem=dem, dirnam=procdir,
                               threads=num_threads,
                               stdout=f"{procdir}prj/{opt.targetmpp}mpp/tmp_{img}.log")
            else:
                prj_img_path = f"{seldir}prj/{bundle_adjust_prefix}/"
                os.makedirs(prj_img_path, exist_ok=True)
                prj_img_path += f"{img}_map.tif"

                if not os.path.exists(prj_img_path):
                    mapproject(from_=f"{procdir}{img}.cub", to=prj_img_path,
                               bundle_adjust_prefix=f"{seldir}{bundle_adjust_prefix}/run",
                               dem=dem,
                               dirnam=procdir,
                               threads=num_threads,
                               stdout=f"{seldir}prj/{bundle_adjust_prefix}/tmp_{img}.log")

            # check if mapproj image has been correctly generated
            if not os.path.exists(prj_img_path):
                logging.error(f"** Issue with tile_{tileid}:{img} projection. "
                      f"Check {seldir}prj/{bundle_adjust_prefix}/tmp_{img}.log and relaunch postprocessing.")
                # exit()

            return prj_img_path

        dem = f"{procdir}/ldem_{tileid}_fullA3.tif"
        gdal_translate(procdir, filin=f"/explore/nobackup/people/sberton2/RING/dems/A3_sites/ldem5mpp_tile_{tileid}.tif",
                       filout=dem,
                       tr=f"{opt.targetmpp} {opt.targetmpp}", r='cubicspline')

        if parallel:
            p_umap(partial(load_project_img, bundle_adjust_prefix=f'ba_align'), final_selection.iterrows(), total=len(final_selection))
            # p_umap(partial(load_project_img, bundle_adjust_prefix='ba'), final_selection.iterrows(), total=len(final_selection))
        else:
            for idxrow in final_selection.iterrows():
                load_project_img(idxrow, bundle_adjust_prefix=f'ba_align')
                # load_project_img(idxrow, bundle_adjust_prefix='ba')

        # check if all selected images have been mapprojected
        assert (len([f"{procdir}{x}.IMG" for x in final_selection.img_name.values]) ==
                len([f"{seldir}prj/ba_align/{x}_map.tif" for x in final_selection.img_name.values]))
                #len([f"{procdir}prj/ba/{x}_map.tif" for x in final_selection.img_name.values]))

        dem_mosaic(imgs=[f"{seldir}prj/ba_align/{x}_map.tif" for x in final_selection.img_name.values
                         #if f"{procdir}prj/ba_align/{x}_map.tif" in glob.glob(f"{procdir}prj/ba_align/M*_map.tif")
                         ], dirnam=seldir,
                   max=None, output_prefix=f"max_lit_aligned_{tileid}_sel{sel}.tif")
        #dem_mosaic(imgs=[f"{procdir}prj/ba_align/{x}_map.tif" for x in final_selection.img_name.values], dirnam=procdir,
        #           max=None, output_prefix=f"max_lit_aligned_{tileid}.tif")
        #dem_mosaic(imgs=[f"{procdir}prj/ba/{x}_map.tif" for x in final_selection.img_name.values], dirnam=seldir,
        #           max=None, output_prefix=f"max_lit_aligned_{tileid}_sel{sel}.tif")

    def eval_dem_from_render(shadow_threshold, use_rough_sel=False, use_priori_dem=False):
        """Uses bundle-adjusted aligned NAC images from procdir/prj/ba_align/ to make native resolution post products. Includes binary mask,
        number of images, azimuth range, azimuth standard deviation, and minimum resolution plots.
        """
        # shadow_threshold = 0.002

        opt = SfsOpt.get_instance()
        
        if use_rough_sel:
            meas_path = glob.glob(f"{procdir}prj/ba/*_map.tif") # ROUGH SELECTION (and not aligned to DEM)
        else:
            final_sel = f"{procdir}final_selection_{tileid}_sel{sel}.csv"
            imgs = pd.read_csv(final_sel, sep=',').img_name.values
            meas_path = [f"{seldir}prj/ba_align/{x}_map.tif" for x in imgs]
            
        if use_priori_dem:
            sfsdem = xr.open_dataarray(f"{procdir}ldem_{tileid}.tif")
        else:
            sfsdem = xr.open_dataarray(f"{sfsdir}run-DEM-final.tif")

        dem_crs = sfsdem.rio.crs
            
        # load all NACs
        #meas_path = meas_path[:5]
        #print(meas_path)
        meas = {}
        for i in tqdm(range(len(meas_path))):
            imgid = meas_path[i].split('/')[-1].split('_map')[0]

            # TWEAK TO GET FINAL SEL ONLY
            #if not imgid in final_selection.img_name.values:
            #    continue
            ###
            
            tmp = xr.open_dataarray(meas_path[i]).rio.reproject_match(sfsdem, Resampling=Resampling.cubic_spline, nodata=np.nan)
            if use_rough_sel:
                tmp = tmp.coarsen(x=2, boundary='trim').mean().coarsen(y=2, boundary='trim').mean()
            cond = (tmp <= shadow_threshold) | (np.isnan(tmp))
            meas[imgid] = xr.where(cond, 0., 1.).astype('uint8')
            assert meas[imgid].rio.crs == dem_crs

        # stack along xarray new dimension
        meas_stack = xr.concat(meas.values(), pd.Index(meas.keys(), name='img'))
        del(meas)
        print(meas_stack)

        # (copy to and ) read INDEX info to dict
        try:
            cumindex = pd.read_parquet(f"{opt.get('rootdir')}{opt.get('pds_index_name')}.parquet")
        except:
            cumindex = pd.read_pickle(f"{opt.get('rootdir')}{opt.get('pds_index_name')}.pkl")

        dict_cumindex = read_img_properties(meas_stack['img'].values,
                                            cumindex, columns=['PRODUCT_ID', 'SUB_SOLAR_LONGITUDE', opt.resolution_name])
        dict_cumindex = dict_cumindex.set_index('PRODUCT_ID').to_dict()

        # remove 0 or negative values (could modify for threshold)
        #meas_stack = meas_stack.where(meas_stack > shadow_threshold).isel(band=0)
        #meas_stack = meas_stack.isel(band=0)
        #print(meas_stack)

        # nimg product
        count = meas_stack.sum(dim='img').astype('uint8')
        print(count)
        #count.plot()
        #plt.title('Number of images')
        #plt.savefig(f"{quicklook}hires_nimg_{tileid}_sel{sel}.png")
        #plt.clf()
        filout = f"{seldir}hires_nimg_{tileid}_sel{sel}.tif"
        count.rio.to_raster(filout, compress='zstd')
        logging.info(f"- Saved nimg to {filout}.")
        
        # best resolution plot
        meas_stack_bestres = meas_stack.copy().astype('uint16')
        #print(meas_stack_bestres)
        for imgid in meas_stack.img.values:
            meas_stack_bestres.loc[{'img': imgid}] = xr.where(meas_stack_bestres.loc[{'img': imgid}]==0., # np.isnan(meas_stack_bestres.loc[{'img': imgid}]),
                                                              999, #meas_stack_bestres.loc[{'img': imgid}],
                                                              dict_cumindex[opt.resolution_name][imgid]*100.)
        #print(meas_stack_bestres)
        bestres = meas_stack_bestres.min(dim='img')
        del(meas_stack_bestres)
        print(bestres)
        bestres = bestres.astype('float32')
        bestres = xr.where(np.isclose(bestres,999.), np.nan, bestres)
        bestres /= 100.
        # print(bestres)
        
        #bestres.plot(robust=True)
        #plt.title("Best Resolution")
        #plt.savefig(f"{quicklook}hires_best_resol_{tileid}_sel{sel}.png")
        #plt.clf()
        filout = f"{seldir}hires_best_resol_{tileid}_sel{sel}.tif"
        bestres.rio.to_raster(filout, compress='zstd')
        logging.info(f"- Saved best resolution to {filout}.")
        del(bestres)

        # binary mask plot
        #meas_stack_bin = meas_stack.copy()
        #for imgid in meas_stack.img.values:
        #    cond = (meas_stack_bin.loc[{'img': imgid}] <= shadow_threshold) | (np.isnan(meas_stack_bin.loc[{'img': imgid}]))
        #    meas_stack_bin.loc[{'img': imgid}] = xr.where(cond, 0., 1.)
        #meas_bin = meas_stack_bin.max(dim='img').astype('uint8')
        #print(meas_bin)
        #meas_bin.plot()
        #plt.title("Binary Mask")
        #plt.savefig(f"{quicklook}binary_mask_{tileid}_sel{sel}.png")
        #plt.clf()
        #filout = f"{seldir}binary_mask_{tileid}_sel{sel}.tif"
        #meas_bin.rio.to_raster(filout)
        #logging.info(f"- Saved mask to {filout}.")
        #del(meas_stack_bin, meas_bin)
        
        # Define a function to assign the wanted values to each combination
        def mask_function(img, index):
            return np.where(np.isnan(img), 0, 2 ** index)

        # Applying the mask function along the 'img' axis
        img_combos = xr.apply_ufunc(
            mask_function,
            meas_stack,
            xr.DataArray(np.arange(meas_stack.sizes['img']), dims="img"),
            input_core_dims=[["img"], ["img"]],
            output_core_dims=[["img"]],
            vectorize=True,
            dask="parallelized"
        ).sum(dim="img")#.astype('uint8') # want to only take unique values and convert to "index" in a list of those
        mapping_series = pd.DataFrame(img_combos.data.ravel(), columns=['orig'])
        mapping_series = mapping_series.drop_duplicates()
        mapping_series = mapping_series.reset_index(drop=True)
        mapping_series = mapping_series.reset_index().set_index('orig')['index']
        # arrange to map on xr
        img_combos_df = img_combos.to_dataframe(name='values').reset_index()
        img_combos_df['mapped_values'] = img_combos_df['values'].map(mapping_series)
        mapped_data = img_combos_df.set_index(['y', 'x'])['mapped_values'].to_xarray()
        mapped_data = mapped_data.astype('uint32', order='C')

        #mapped_data.plot()
        #plt.title('Combinations of images (2^i, i=1..nimg)')
        #plt.savefig(f"{seldir}hires_img_combo_{tileid}_sel{sel}.png")
        #plt.clf()

        mapped_data.rio.write_crs(dem_crs, inplace=True)
        print(mapped_data)
        filout = f"{seldir}hires_img_combo_{tileid}_sel{sel}.tif"
        mapped_data.rio.to_raster(filout)
        logging.info(f"- Saved img combos to {filout}.")
        

        # replace with subsolar longitudes and compute range of available angles
        meas_stack_ssl = meas_stack.copy().astype('int16')
        for imgid in meas_stack.img.values:
            meas_stack_ssl.loc[{'img': imgid}] = xr.where(meas_stack_ssl.loc[{'img': imgid}]==0.,
                                                          -1., #meas_stack_ssl.loc[{'img': imgid}],
                                                  dict_cumindex['SUB_SOLAR_LONGITUDE'][imgid])
        meas_stack_ssl.rio.write_crs(dem_crs, inplace=True)
        print(meas_stack_ssl)

        def use_angrange(da, axis):
            #print("in:", da)
            da = np.where(da == -1, np.nan, da)
            try:
                res = np.apply_along_axis(angrange, axis=axis, arr=da)
                #print("res:", res)
                return res
            except:
                print(da)
                print(np.apply_along_axis(angrange, axis=axis, arr=da))
                exit()
            
        # azimuth range product
        angrange_ds = xr.apply_ufunc(use_angrange, meas_stack_ssl, input_core_dims=[['img']],
                                     kwargs={"axis": -1}, vectorize=True, dask="parallelized")
        angrange_ds.rio.write_crs(dem_crs, inplace=True)
        print(angrange_ds)
        
        #angrange_ds.plot(robust=True)
        #plt.title('Range of available illumination angles (deg)')
        #plt.savefig(f"{quicklook}hires_azi_range_{tileid}_sel{sel}.png")
        #plt.clf()
        filout = f"{seldir}hires_azi_range_{tileid}_sel{sel}.tif"
        angrange_ds.rio.to_raster(filout)
        logging.info(f"- Saved solar angles to {filout}.")

        # azimuth std product
        azi_std = meas_stack_ssl.where(meas_stack_ssl != -1).std(dim='img')
        #azi_std.plot(robust=True)
        #plt.title('Sun Azimuth (stdev, deg)')
        #plt.savefig(f"{quicklook}hires_azi_std_{tileid}_sel{sel}.png")
        #plt.clf()
        filout = f"{seldir}hires_azi_std_{tileid}_sel{sel}.tif"
        azi_std.rio.to_raster(filout)
        logging.info(f"- Saved solar angles std to {filout}.")

        del(meas_stack_ssl)


    def prepare_products(prodsdir, use_priori_dem=False):

        # Product ID	Product #	Name	Unit	Parameters	Filename example
        # GLDELEV	GLD01	Elevation	m		SITEID_GLDELEV_002
        # GLDMASK	GLD02	Data Mask	-	altimetry point count, or number of images	SITEID_GLDMASK_002
        # GLDSIGM	GLD03	Elevation Uncertainty	m		SITEID_GLDISGM_002
        # GLDDIFF	GLD04	Elevation Difference to Reference	m	vs. LOLA DEM	SITEID_GLDDIFF_002
        # GLDSLOP	GLD05	Slope	deg	computed from GLD01	SITEID_GLDSLOP_002
        # GLDHILL	GLD06	Hillshade	-	default: 45/315	SITEID_GLDHILL_002_315_45
        # GLDOMOS	GLD07	Orthomosaic	RADF		SITEID_GLDOMOS_002
        # GLDAZRN	GLD14	Range of Azimuth of Input Images	deg		SITEID_GLDAZRN_002
        # GLDAZSD	GLD15	Std.Dev. of Azimuth of Input Images			SITEID_GLDAZSD_002
        # GLDBRES	GLD16	Best Resolution of Input Images	m		SITEID_GLDBRES_002
        # GLDICMB	GLD16	Combinations of images      		SITEID_GLDICMB_002

        siteid = f"{opt.get('site')[:2]}{tileid:02d}"

        # create products dir
        os.makedirs(prodsdir, exist_ok=True)

        # compute slope from DEM
        if use_priori_dem:
            refdem_path = f"{procdir}ldem_{tileid}.tif"
        else:
            refdem_path = f"{sfsdir}run-DEM-final.tif"

        dem_ds = xr.open_dataset(refdem_path)
        slope_da = slope(dem_ds.isel(band=0).band_data, name=f"{siteid}_GLDSLOP_001")
        slope_da.rio.to_raster(f"{prodsdir}{siteid}_GLDSLOP_001.tif")

        # compute hillshade (only save band=1)
        # use xrspatial.hillshade as alternative (cleaner, not 100% sure how equivalent)
        azi = 315
        elev = 45
        hillshade(dem=refdem_path, dirnam=seldir,
                  azi=azi, elev=elev,
                  o=f"{prodsdir}{siteid}_GLDHILL_001_{azi}_{elev}.tif")
        hs_ds = xr.open_dataset(f"{prodsdir}{siteid}_GLDHILL_001_{azi}_{elev}.tif")
        hs_ds.isel(band=0).band_data.rio.to_raster(f"{prodsdir}{siteid}_GLDHILL_001_{azi}_{elev}.tif")

        # copy or move already available products
        shutil.copy(refdem_path, f"{prodsdir}{siteid}_GLDELEV_001.tif")
        if not use_priori_dem:
            shutil.copy(f"{sfsdir}run-error-height-error.tif", f"{prodsdir}{siteid}_GLDSIGM_001.tif")
            shutil.move(f"{seldir}vdiff_ldem_{tileid}_sel{sel}.tif", f"{prodsdir}{siteid}_GLDDIFF_001.tif")
            shutil.move(f"{seldir}max_lit_aligned_{tileid}_sel{sel}.tif", f"{prodsdir}{siteid}_GLDOMOS_001.tif")
        else:
            shutil.copy(f"{procdir}max_lit_ba_{tileid}.tif", f"{prodsdir}{siteid}_GLDOMOS_001.tif")
        shutil.move(f"{seldir}hires_nimg_{tileid}_sel{sel}.tif", f"{prodsdir}{siteid}_GLDMASK_001.tif")
        shutil.move(f"{seldir}hires_azi_range_{tileid}_sel{sel}.tif", f"{prodsdir}{siteid}_GLDAZRN_001.tif")
        shutil.move(f"{seldir}hires_azi_std_{tileid}_sel{sel}.tif", f"{prodsdir}{siteid}_GLDAZSD_001.tif")
        shutil.move(f"{seldir}hires_img_combo_{tileid}_sel{sel}.tif", f"{prodsdir}{siteid}_GLDICMB_001.tif")
        shutil.move(f"{seldir}hires_best_resol_{tileid}_sel{sel}.tif", f"{prodsdir}{siteid}_GLDBRES_001.tif")


    #if not use_rough_sel:
    # run post-processing validation script
    #    postpro_val(tileid, sel, final_selection, comp_cut=0.04, align=True, render=True, diff=True, products=True)
    #exit()

    # test
    postpro_val(tileid, sel, final_selection, comp_cut=0.04, align=True, render=True, diff=True, products=True)
    exit()
        
    # run postprocessing steps
    if not use_rough_sel:
        vdiff_map()
        #pixel_differences_rms, pixel_differences_xr = diff_to_nac()
        aligned_maxlit()
        print('starting postpro validation')
    else:
        prodsdir = f"{prodsdir}all_imgs/"
        
    eval_dem_from_render(shadow_threshold, use_rough_sel, use_priori_dem)

    prepare_products(prodsdir, use_priori_dem)

    list_of_prods = glob.glob(f"{prodsdir}{siteid}_*_001.tif") + glob.glob(
        f"{prodsdir}{siteid}_GLDHILL_001_*_??.tif")
    list_of_prods = sorted(list_of_prods, key=lambda f: os.path.basename(f))
    print(list_of_prods)
    products_overview(list_of_prods, f'{prodsdir}{siteid}_products_001.png', )

    if not use_rough_sel:
        # run post-processing validation script
        postpro_val(tileid, sel, final_selection, comp_cut=0.04, align=True, render=True, diff=True, products=True)
        #shutil.copy(f'{prodsdir}{siteid}_products_001.png', f'{prodsdir}{siteid}_{sel}_products_001.png')




