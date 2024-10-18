import glob
import os
import logging
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import xarray as xr
from p_tqdm import p_umap
from functools import partial
from rasterio.enums import Resampling
from sfs.alignment.align_from_render import render_adjusted, align_img, project_img
from sfs.preprocessing.preprocessing import load_calibrate_project_img
from isis.functions import set_isis
from asp.functions import mapproject, bundle_adjust, set_asp
from tqdm import tqdm
from sfs.config import SfsOpt
from time import time


# func to render both selected and unselected, then difference rendering to NAC, save them as tifs
# func to read those and then make plots
# edit align_unselected to be good stand-alone

def align_mapproj(tile, sel, seldir, dem_path, prj_img_path, filtered_imgs):
        """Aligns the group of images that were not selected in the final selection step, but
        were eligible for selection using the run-ecef-transform.txt file from the selected images. 
        Args:
            tile (int): tile ID number
            sel (int): image selection number
            seldir (str): full path to selection directory
            dem_path (str): full path to final sfs DEM product
            prj_img_path (str): full path to directory where the aligned map-projected images will be generated
            filtered_imgs (array): contains the image names
        """
        #os.makedirs(f"{seldir}ba_align/", exist_ok=True)
        #shutil.copy(f"{seldir}/ba_align/run-ecef-transform.txt", f"{seldir}ba_align_unselected/run-ecef-transform.txt")

        opt = SfsOpt.get_instance()
        
        assert os.path.exists(f"{seldir}/ba_align/run-ecef-transform.txt"), \
                f"{seldir}/ba_align/run-ecef-transform.txt does not exist. Exit."

        # to be improved... TODO
        procdir = f"{seldir}../"
        for index,row in filtered_imgs.iterrows(): #in enumerate(filtered_imgs):
                img = row.img_name

                # update json
                load_calibrate_project_img((index,row), tile, dem_path, bundle_adjust_prefix=None, project_imgs=False)
                if not os.path.isfile(f"{procdir}{img}.cal.echo.red{opt.targetmpp}.cub"):
                        logging.error(f"- {img}.cal.echo.red{opt.targetmpp}.cub does not exist. Check.")
                # apply registration to DEM to adjust files
                align_img(img, tile, sel, seldir, dem_path) #, prj_img_path)

        # mapproject to new DEM
        filtered_imgs_names = filtered_imgs.img_name.values
        p_umap(partial(project_img, tile=tile, sel=sel, seldir=seldir,
                        prioridem_path=dem_path, ba_prefix="ba_align" , prj_img_path=prj_img_path),
                        filtered_imgs_names, desc='mapproj ba_align', total=len(filtered_imgs_names))
                    
                    
def diff_to_nac(tileid, sel, seldir, selection, sfsdir, ba_align_path, compproj_paths, dem_path, comp_cut, render_post_dir):
    """Differences the renders and the NAC images a given groups of images (selected or unselected).

    Args:
        seldir (str): full path to selection directory
        selection (DataFrame): contains information on the images (selected or unselected group)
        sfsdir (str): full path to sfs directory 
        ba_align_path (str): full path to directory containing the images aligned to the a priori DEM (selected or unselected group)
        compproj_paths (str): list of full paths rendered images (selected or unselected group)
        render_post_dir (str): full path to directory where selected and unselected post-processing renders will be generated

    Returns:
        (list): list of str of saved NAC/render tif difference files
    """
    os.makedirs(render_post_dir, exist_ok=True)
    opt = SfsOpt.get_instance()
    
    sfsdem = xr.open_dataset(dem_path)

    # parallel render and compare to NAC
    measproj_paths = [f"{ba_align_path}/{img}_map.tif" for img in selection.img_name.values]
    
    diffs_paths = []
    for idx, img in tqdm(enumerate(selection.img_name.values), total=len(selection), desc='Differencing NAC and renders'):
    
        if os.path.exists(f"{render_post_dir}render_vs_nac_{img}.png") and os.path.exists(f"{render_post_dir}render_vs_nac_{img}.tif"):
            logging.info(f'{img} difference files already exist. Skip.')
            continue
        
        meas = [x for x in measproj_paths if (img in x) and ("_sub" not in x)]
        comp = [x for x in compproj_paths if x is not None and (img in x) and ("_sub" not in x)]

        try:
            assert (len(meas) == 1) and (len(comp) == 1)
            meas = meas[0]
            comp = comp[0]
        except:
            logging.warning(f"- Missing rendered or NAC image {img}. Continue.") # likely triggered if rendering failed for some images, became None in comproj_paths
            continue

        meas_ds = xr.open_dataset(meas)
        comp_ds = xr.open_dataset(comp)

        # save raster with both computed and measured?
        if opt.debug:
            meascomp_ds = xr.merge([meas_ds.rename({'band_data': 'meas_rad'}),
                                        comp_ds.rename({'band_data': 'comp_rad'}).where(comp_ds.band_data > 0)])
            meascomp_ds = meascomp_ds.fillna(-99999)
            meascomp_ds.sel({'band': 1}).rio.to_raster(f"{render_post_dir}run-{img}-meascomp-intensity.tif")

        # remove problematic areas from the rendered image (get threshold value from render-real pixel scatterplot)
        bin_mask = xr.open_dataset(f'{seldir}binary_mask_{tileid}_sel{sel}.tif')
        comp_ds = comp_ds.rio.reproject_match(sfsdem, Resampling=Resampling.bilinear, nodata=np.nan)
        comp_ds = comp_ds.where(bin_mask==1)
        comp_ds = comp_ds.where(comp_ds<=comp_cut)
        meas_ds = meas_ds.rio.reproject_match(sfsdem, Resampling=Resampling.bilinear, nodata=np.nan)
        
        diff = meas_ds - comp_ds
        diff = diff.rio.write_crs(sfsdem.rio.crs, inplace=True)
        diff = diff.rio.reproject_match(sfsdem, Resampling=Resampling.bilinear, nodata=np.nan)

        # maybe useful for aggregated map of pixel differences
        # diff_mean = xr.concat(diffs, dim='band').mean(dim='band')
        # diff_std = xr.concat(diffs, dim='band').std(dim='band')
        # diff_rms = (diff_mean.map(np.square) + diff_std.map(np.square)).map(np.sqrt)

        # save figure and raster of differences
        diff.band_data.plot(robust=True)
        plt.savefig(f"{render_post_dir}render_vs_nac_{img}.png")
        plt.clf()
        diff.band_data.rio.to_raster(f"{render_post_dir}render_vs_nac_{img}.tif")
        diffs_paths.append(f"{render_post_dir}render_vs_nac_{img}.tif")

    logging.info(f"- {len(measproj_paths)} images rendered and comparisons saved to {render_post_dir}.")

    # returning just the paths instead of the list of xarrays
    return diffs_paths


def make_products(tileid, sel, seldir, siteid, prodsdir, diff_paths_sel, diff_paths_unsel, sel_paths, unsel_paths, dem_path):
    """Makes data products from the NAC/rendered image differences computed in diff_to_nac(). 
    Args:
        seldir (str): full path to selection directory
        siteid (str): site name
        prodsdir (str): full path to directory where products will be generated
        diff_paths_sel (list of str): list of paths to the NAC/rendered image differences
        of the selected images
        diff_paths_unsel (list of str): list of paths to the NAC/rendered image differences
        of the unselected images
        sel_paths (list of str)
        unsel_paths (list of str)
        dem_path (str)
    """
    # load in difference data from tif files for selected and unselected images
    sfsdem = xr.open_dataset(dem_path)
    # def _preprocess(x):
    #     #x = x.rio.write_crs(sfsdem.rio.crs, inplace=True)
    #     return x.rio.write_crs(sfsdem.rio.crs, inplace=True)
    # partial_func = partial(_preprocess)
    sel_diff = {}
    unsel_diff = {}
    for i in tqdm(diff_paths_sel, total=len(diff_paths_sel), desc='Opening selected image differences'):
        imgid = i.split('/')[-1].split('_')[3].split('.tif')[0]
        img = xr.open_dataarray(i).rio.write_crs(sfsdem.rio.crs, inplace=True)
        sel_diff[imgid] = img.rio.reproject_match(sfsdem, Resampling=Resampling.bilinear,
                                        nodata=np.nan).astype('uint8')
        print(i)
    # start = time()
    # sel_diff_stack = xr.open_mfdataset(diff_paths_sel, chunks='auto', )#preprocess=partial_func)
    # print(f'time is {time() - start}.')
    # print(sel_diff_stack)
    print('opening unselected image differences')
    for i in tqdm(diff_paths_unsel, total=len(diff_paths_unsel), desc='Opening unselected image differences'):
        imgid = i.split('/')[-1].split('_')[3].split('.tif')[0]
        img = xr.open_dataarray(i).rio.write_crs(sfsdem.rio.crs, inplace=True)
        unsel_diff[imgid] = img.rio.reproject_match(sfsdem, Resampling=Resampling.bilinear,
                                        nodata=np.nan).astype('uint8')
        print(i)
    # start = time()
    # unsel_diff_stack = xr.open_mfdataset(diff_paths_unsel, chunks='auto')
    # print(f"time is {time() - start}.")
    # print(unsel_diff_stack)
    # concatenate into xarray datasets for plotting and delete dictionaries
    # sel_diff_stack = xr.concat(sel_diff.values(), pd.Index(sel_diff.keys(), name='img')).astype('uint8')
    # unsel_diff_stack = xr.concat(unsel_diff.values(), pd.Index(unsel_diff.keys(), name='img')).astype('uint8')
    # rough_diff_stack = xr.concat((sel_diff_stack, unsel_diff_stack), dim='img').astype('uint8')
    # print(sel_diff_stack)
    # print(unsel_diff_stack)
    # print(rough_diff_stack)
    # del(sel_diff)
    # del(unsel_diff)
    
    # plot differences between image pairs
    stack = xr.concat(sel_diff.values(), pd.Index(sel_diff.keys(), name='img')).astype('uint8')
    stack = stack.values[~np.isnan(stack)].astype('uint8')
    counts, bins = np.histogram(stack, bins=100)
    plt.stairs(counts/len(stack), bins, label='selected')
    stack = xr.concat(unsel_diff.values(), pd.Index(unsel_diff.keys(), name='img')).astype('uint8')
    stack = stack.values[~np.isnan(stack)].astype('uint8')
    counts, bins = np.histogram(stack, bins=100)
    plt.stairs(counts/len(stack), bins, label='unselected')
    stack = xr.concat(sel_diff.values(), unsel_diff.values(), pd.Index(sel_diff.keys(), name='img'), pd.Index(unsel_diff.keys(), name='img')).astype('uint8')
    stack = stack.values[~np.isnan(stack)].astype('uint8')
    counts, bins = np.histogram(stack, bins=100)
    plt.stairs(counts/len(stack), bins, label='rough selection')
    plt.ylim(0,1)
    plt.xlim(0,0.06)
    plt.xlabel('Differences')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.title("Image Pair Differences")
    plt.savefig(f'{prodsdir}img_pair_diff.png')
    plt.clf()
    exit()
    
    # plot medians of image pair differences
    bin_edges = np.linspace(rough_diff_stack.min(), rough_diff_stack.max(), num=200)
    med = sel_diff_stack.median(dim=['x','y'], skipna=True).astype('uint8')
    data = med.values[~np.isnan(med)].astype('uint8')
    counts, bins = np.histogram(data, bins=bin_edges)
    plt.stairs(counts/len(data), bins, label='selected')
    med = unsel_diff_stack.median(dim=['x','y'], skipna=True)
    data = med.values[~np.isnan(med)].astype('uint8')
    counts, bins = np.histogram(data, bins=bin_edges)
    plt.stairs(counts/len(data), bins, label='unselected')
    med = rough_diff_stack.median(dim=['x','y'], skipna=True).astype('uint8')
    data = med.values[~np.isnan(med)].astype('uint8')
    counts, bins = np.histogram(data, bins=bin_edges)
    plt.stairs(counts/len(data), bins, label='rough selection')
    plt.xlim(0,0.03)
    plt.ylim(0,1)
    plt.xlabel('Medians of Image Pair Differences')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.title('Medians')
    plt.savefig(f'{prodsdir}pair_diff_medians.png')
    plt.clf()
    
    # plot standard deviations of image pair differences
    bin_edges = np.linspace(rough_diff_stack.min(), rough_diff_stack.max(), num=200)
    med = sel_diff_stack.std(dim=['x','y'], skipna=True).astype('uint8')
    data = med.values[~np.isnan(med)].astype('uint8')
    counts, bins = np.histogram(data, bins=bin_edges)
    plt.stairs(counts/len(data), bins, label='selected')
    med = unsel_diff_stack.std(dim=['x','y'], skipna=True).astype('uint8')
    data = med.values[~np.isnan(med)].astype('uint8')
    counts, bins = np.histogram(data, bins=bin_edges)
    plt.stairs(counts/len(data), bins, label='unselected')
    med = rough_diff_stack.std(dim=['x','y'], skipna=True).astype('uint8')
    data = med.values[~np.isnan(med)].astype('uint8')
    counts, bins = np.histogram(data, bins=bin_edges)
    plt.stairs(counts/len(data), bins, label='rough selection')
    plt.xlim(0,0.03)
    plt.ylim(0,1)
    plt.xlabel('Standard Deviations')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.title('Standard Deviations of Image Pair Differences')
    plt.savefig(f'{prodsdir}pair_diff_std.png')
    plt.clf()
    
    # plot RMS of image pairs
    sel_nimgs = sel_diff_stack.count(dim='img').astype('uint8')
    unsel_nimgs = unsel_diff_stack.count(dim='img').astype('uint8')
    rough_nimgs = rough_diff_stack.count(dim='img').astype('uint8')

    sel_rms = np.sqrt(1/sel_nimgs * (np.power(sel_diff_stack, 2).sum(dim='img')))  
    unsel_rms = np.sqrt(1/unsel_nimgs * (np.power(unsel_diff_stack, 2).sum(dim='img'))) 
    rough_rms = np.sqrt(1/rough_nimgs * (np.power(rough_diff_stack, 2).sum(dim='img')))

    bin_edges = np.linspace(rough_rms.min(), rough_rms.max(), num=100)
    data = sel_rms.values[~np.isnan(sel_rms)].astype('uint8')
    counts, bins = np.histogram(data, bins=bin_edges)
    plt.stairs(counts/len(data), bins, label='selected')
    data = unsel_rms.values[~np.isnan(unsel_rms)].astype('uint8')
    counts, bins = np.histogram(data, bins=bin_edges)
    plt.stairs(counts/len(data), bins, label='unselected')
    data = rough_rms.values[~np.isnan(rough_rms)].astype('uint8')
    counts, bins = np.histogram(data, bins=bin_edges)
    plt.stairs(counts/len(data), bins, label='rough selection')
    plt.ylim(0,1)
    plt.xlim(0,0.05)
    plt.xlabel('RMS')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.title('RMS of NAC and Rendered Image Pairs')
    plt.savefig(f'{prodsdir}rms_pairs.png')
    plt.clf()

    # plot RMS as a map
    fig, axes = plt.subplots(1, 3, figsize=(26, 6))
    sel_rms.plot(ax=axes[0])
    axes[0].set_title('Selected Images')
    unsel_rms.plot(ax=axes[1])
    axes[1].set_title("Unselected Images")
    rough_rms.plot(ax=axes[2])
    axes[2].set_title("Rough Selection Images")
    plt.suptitle(f'RMS Maps')
    plt.savefig(f'{prodsdir}rms_maps.png')
    plt.clf()

    # plot median absolute error 
    sel_mae = abs(sel_diff_stack).median(dim='img').astype('uint8')
    unsel_mae = abs(unsel_diff_stack).median(dim='img').astype('uint8')
    rough_mae = abs(rough_diff_stack).median(dim='img').astype('uint8')
    data = sel_mae.values[~np.isnan(sel_mae)].astype('uint8')
    counts, bins = np.histogram(data, bins=100)
    plt.figure(figsize=(8,6))
    plt.stairs(counts/len(data), bins, label='selected')
    data = unsel_mae.values[~np.isnan(unsel_mae)].astype('uint8')
    counts, bins = np.histogram(data, bins=100)
    plt.stairs(counts/len(data), bins, label='unselected')
    data = rough_mae.values[~np.isnan(rough_mae)].astype('uint8')
    counts, bins = np.histogram(data, bins=100)
    plt.stairs(counts/len(data), bins, label='rough selection')
    plt.ylim(0,1)
    plt.xlim(0,0.05)
    plt.xlabel('Median Absolute Error')
    plt.ylabel('Normalized Frequency')    
    plt.legend()
    plt.title('Median Absolute Error Between NAC and Rendered Images')
    plt.savefig(f'{prodsdir}med_abs_error.png')
    plt.clf()
    
    # plot 2d histo of RMS vs slope
    slope = xr.load_dataset(f"{seldir}/products/{siteid[:2]}{tileid}{sel}_GLDSLOP_001.tif")
    slope_flat = slope.to_array().values.ravel()
    slope_data = np.nan_to_num(slope_flat)
    sel_rms_flat = sel_rms.values.ravel()
    rms_data = np.nan_to_num(sel_rms_flat)
    plt.figure(figsize=(8,6))
    plt.hist2d(slope_data, rms_data, bins=100, norm = colors.LogNorm())
    plt.xlabel('Slope')
    plt.ylabel('Selected Images RMS')
    plt.colorbar(label='Counts')
    plt.title('RMS Vs. Slope in NAC and Rendered Image Pairs')
    plt.savefig(f'{prodsdir}rms_slope_2dhisto.png')
    plt.clf()
    
    del(sel_diff_stack)
    del(unsel_diff_stack)
    del(rough_diff_stack)
    
    ####### load in NAC and renders to do percent error plot ########
    sel_NAC = {}
    unsel_NAC = {}
    for i in tqdm(sel_paths, total=len(sel_paths), desc='selected NAC images'):
        imgid = i.split('/')[-1].split('_')[3].split('.tif')[0]
        sel_NAC[imgid] = xr.load_dataarray(sel_paths[i]).rio.reproject_match(sfsdem, Resampling=Resampling.bilinear,
                                        nodata=np.nan)
        assert sel_NAC[imgid].rio.crs == sfsdem.rio.crs
        
    for i in tqdm(unsel_paths, total=len(unsel_paths), desc='unselected'):
        imgid = i.split('/')[-1].split('_')[3].split('.tif')[0]
        unsel_NAC[imgid] = xr.load_dataarray(unsel_paths[i]).rio.reproject_match(sfsdem, Resampling=Resampling.bilinear,
                                        nodata=np.nan)
        assert unsel_NAC[imgid].rio.crs == sfsdem.rio.crs
    # stack into dataset
    sel_NAC_stack = xr.concat(sel_NAC.values(), pd.Index(sel_NAC.keys(), name='img'))
    unsel_NAC_stack = xr.concat(unsel_NAC.values(), pd.Index(unsel_NAC.keys(), name='img'))
    ###############
    
    # plot percent error 
    # make individual NAC and render stacks for percent error plot
    # percent error = rms / median flux * 100
    plt.figure(figsize=(8,6))
    
    sel_per_err = sel_rms/(sel_NAC_stack.median(dim=['img'],skipna = True))*100
    data = sel_per_err.values[~np.isnan(sel_per_err)]
    counts, bins = np.histogram(data, bins=50)
    plt.stairs(counts/len(data), bins, label='selected')
    del(sel_NAC_stack)
    del(sel_per_err)
    
    unsel_per_err = unsel_rms/(unsel_NAC_stack.median(dim=['img'],skipna = True))*100
    data = unsel_per_err.values[~np.isnan(unsel_per_err)]
    counts, bins = np.histogram(data, bins=50)
    plt.stairs(counts/len(data), bins, label='unselected')
    del(unsel_NAC_stack)
    del(unsel_per_err)
    
    rough_per_err = rough_rms/(rough_NAC_stack.median(dim=['img'],skipna = True))*100
    data = rough_per_err.values[~np.isnan(rough_per_err)]
    counts, bins = np.histogram(data, bins=50)
    plt.stairs(counts/len(data), bins, label='rough selection')
    del(rough_NAC_stack)
    del(rough_per_err)
    
    plt.ylim(0,1)
    plt.xlim(0,200)
    plt.xlabel('Percent Error')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.title('RMS Percent Error')
    plt.savefig(f'{prodsdir}rms_percent_error.png')
    plt.clf()
    
def postpro_val(tileid, sel, final_selection, comp_cut, exclude=None, align=True, render=True, diff=True, products=True): 
    """Analyzes accuracy of sfs rendering by comparing real images to simulated images from both the
    selected and unselected image sets.

    Args:
        tileid (int): tile ID number
        sel (int): image selection number
        final_selection (DataFrame): contains information on the images in the final selection
        align (bool): default is True, if False it will not align the unselected images
        render (bool): default is True, if False it will not render the images
        diff (bool): default is True, if False it will not difference rendered and NAC images
    """

    opt = SfsOpt.get_instance()
    set_asp(opt.aspdir)
    set_isis(opt.isisdir, opt.isisdata)

    # set up file paths
    procdir = f"{opt.procroot}tile_{tileid}/"
    seldir = f"{procdir}sel_{sel}/"
    sfsdir = opt.get('sfsdir')
    render_post_dir = f"{seldir}render_post/"
    ba_align_path = f"{seldir}prj/ba_align/"
    siteid = f"{opt.get('site')}"
    dem_path = f'{seldir}products/{siteid[:2]}{str(tileid).zfill(2)}_GLDELEV_001.tif' 
    
    # get img paths for selected and unselected, remove poorly aligned and exclude list images from unselected
    rough_selection = pd.read_csv(f"{procdir}filtered_selection_{siteid}_{tileid}.csv", sep=',')
    print(f'rough selection: {procdir}filtered_selection_{siteid}_{tileid}.csv')
    print(f"Filtered list contains {len(rough_selection)} images.")
    good_imgs = rough_selection.copy() #.loc[~rough_selection['img_name'].isin(final_selection['img_name'])] 
    if os.path.exists(f"{procdir}not_aligned_{siteid}_{tileid}.parquet"):
            not_aligned = pd.read_parquet(f"{procdir}not_aligned_{siteid}_{tileid}.parquet")
            good_imgs = good_imgs.loc[~good_imgs['img_name'].isin(not_aligned.img)]
            print(f"We removed {len(not_aligned)} images and ended with {len(good_imgs)} to be mapproj and rendered.")
    else:
        print(f"# No {procdir}not_aligned_{siteid}_{tileid}.parquet found. Weird but continue.")

    # good_imgs = good_imgs.loc[good_imgs['img_name'].isin(['M1307104410RE'])]
        
    # update with exclude list 
    # if exclude == None:
    #     exclude = []
    # else:
    #     print(f'Excluding {exclude} from the exclude list.')
    # good_imgs = good_imgs.loc[good_imgs.img_name.isin(exclude) == False]
    # unsel_ids = good_imgs['img_name'].reset_index(drop=True)

    # make alignment and plot directories
    prj_img_path = f"{seldir}prj/sfsdem/"
    os.makedirs(prj_img_path, exist_ok=True)
    prodsdir = f'{render_post_dir}plots/'
    os.makedirs(prodsdir, exist_ok=True)
    # os.makedirs(f'{seldir}ba_align_unselected/', exist_ok=True)

    print(f"- mapproj aligned images to SFSDEM with crs=")
    dem_crs_test = xr.open_dataarray(dem_path).rio.crs
    print(dem_crs_test)
    
    # align and mapproj all images to sfs dem by applying transform from selection
    start = time()
    if align:
            align_mapproj(tileid, sel, seldir, dem_path, prj_img_path, good_imgs)
    logging.info(f"- Finished aligning all images after {time() - start} seconds.")

    exit()
    
    # get measured image paths for rendering
    mapproj_paths = glob.glob(f"{prj_img_path}*_map.tif")
    #sel_paths = f"{ba_align_path}" + final_selection['img_name'].astype(str) + "_map.tif"
    
    start = time()
    if render:
        #selcomp_paths = render_adjusted(sel_paths, pdir=seldir, dem_path=dem_path, indir=f"{render_post_dir}selected/") #selected
        comp_paths = render_adjusted(mapproj_paths, pdir=seldir, dem_path=dem_path, indir=f"{render_post_dir}all/") #all
    else:
        #selcomp_paths = glob.glob(f"{render_post_dir}selected/out/M1*E_*.tif") 
        comp_paths = glob.glob(f"{render_post_dir}all/out/M1*E_*.tif")
    logging.info(f"- Finished rendering all the images after {time() - start} seconds.")
    
    start = time()
    if diff: 
        #diff_paths_sel = diff_to_nac(tileid, sel, seldir, final_selection, sfsdir, ba_align_path, selcomp_paths, dem_path, comp_cut, render_post_dir=f"{render_post_dir}selected/")
        diff_paths = diff_to_nac(tileid, sel, seldir, sfsdir, prj_img_path, comp_paths, dem_path, comp_cut, render_post_dir=f"{render_post_dir}all/")
    else: # gets files
        #diff_paths_sel = glob.glob(f"{render_post_dir}selected/render_vs_nac_*.tif")
        diff_paths = glob.glob(f"{render_post_dir}all/render_vs_nac_*.tif")
    logging.info(f'- Finished differencing NAC and rendered images after {time() - start} seconds.')

    # make rms plots using the NAC to render differences
    start = time()
    if products:
        make_products(tileid, sel, seldir, siteid, prodsdir, diff_paths_sel, diff_paths_unsel, sel_paths, unsel_paths, dem_path)
    logging.info(f'- Finished making postprocessing validation products after {time() - start} seconds.')
    

