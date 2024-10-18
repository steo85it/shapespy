import logging

from sfs.selection.selection_tools import filter_imgsel

logging.basicConfig(level=logging.INFO)

import argparse
import os
import shutil
import sys
import glob
from importlib import resources
from time import time
import pandas as pd
import yaml

from sfs.preprocessing.import_cumindex import pds3_to_df
from sfs.config import SfsOpt

from sfs.alignment.align_from_render import align_from_render
from sfs.alignment.analyze_matches import analyze_matches_plot_stats
from sfs.alignment.ba_align_res_stat import plot_img_alignments
from sfs.postprocessing.postpro import postpro
from sfs.postprocessing.check_alignment import check_alignment
from sfs.preprocessing.get_lnac_single_cell import rough_imgsel_sites
from sfs.preprocessing.preprocessing import preprocess, ba_and_mapproj, mapproj_maxlit
from sfs.processing.processing import sfs_on_tile
from sfs.selection.images_selection import select
from sfs.selection.improve_selection import improve_selection, improve_selection_sunlon
from utils.download_img import verify_and_download
from utils.yaml_utils import load_config_yaml, load_config_yaml_with_tileid
from asp.functions import dem_mosaic, set_asp
from sfs.alignment.check_align_and_clean import find_bad_images


class Site:
    def __init__(self, **kwargs):

        self.siteid = None
        self.tileid = None
        self.sel = None
        self.steps_to_run = None

        self.sfs_opt = None

        self.procdir = None
        self.seldir = None

        self.config_file = None

        self.procdir = None
        self.rootdir = None
        self.steps_to_run = None

        self.latest_selection = pd.DataFrame()  # TODO update at every selection step

        self.sfs_opt = SfsOpt.get_instance()

        self.setup_config(**kwargs)
        self.setup_dirs(**kwargs)

        if self.sfs_opt.steps_to_run is None:
            self.steps_to_run = '111111111'
        else:
            self.steps_to_run = self.sfs_opt.steps_to_run
        self.configure_steps(self.steps_to_run)

    def setup_config(self, **kwargs):

        # Load default configuration
        with resources.open_text('sfs', 'default_config.yaml') as f:
            default_config = yaml.safe_load(f)
        # Initialize or update the singleton instance
        self.sfs_opt.update_config(**default_config)

        # Update configuration with kwargs from object initialization
        kwargs_config = {k: v for k, v in kwargs.items() if v is not None}
        self.sfs_opt.update_config(**kwargs_config)

        # Handling command-line arguments to override
        parser = argparse.ArgumentParser(description="Dynamic Configuration for SfsOpt")
        for key in self.sfs_opt.__dict__.keys():
            parser.add_argument(f"--{key}", type=type(getattr(self.sfs_opt, key)), help=f"Set {key}")
        parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython",
                            default="1")  # added to make jupyter notebook run
        args = parser.parse_args()

        # Optionally load user configuration if specified
        if args.config_file is not None:
            self.config_file = args.config_file

            if args.tileid is not None:
                user_config = load_config_yaml_with_tileid(self.config_file, args)
            else:
                user_config = load_config_yaml(self.config_file)
            self.sfs_opt.update_config(**user_config)

        # Update configuration with command-line arguments
        cli_config = {k: v for k, v in vars(args).items() if v is not None}
        self.sfs_opt.update_config(**cli_config)

        # Display final configuration
        self.sfs_opt.display()

    def setup_dirs(self, **kwargs):

        if self.sfs_opt.rootroot is not None:
            self.rootroot = self.sfs_opt.rootroot
        else:
            logging.warning('No rootroot specified, using current directory')
            self.rootroot = f"{os.getcwd()}/"

        self.cwd = os.getcwd()
        self.siteid = self.sfs_opt.site
        self.tileid = self.sfs_opt.tileid
        self.sel = self.sfs_opt.sel
        self.rootdir = f"{self.rootroot}root/"
        self.sfs_opt.set('rootdir', self.rootdir)
        self.procroot = f"{self.rootroot}proc/"
        self.sfs_opt.set('procroot', self.procroot)

        self.procdir = f"{self.procroot}tile_{self.tileid}/"
        self.seldir = f"{self.procdir}sel_{self.sel}/"

        try:
            _ = self.sfs_opt.get("imgs_to_remove")[f"{self.tileid}"]
        except:  # if not properly defined
            self.sfs_opt.set('imgs_to_remove', {f"{self.tileid}": []})

        # Create necessary directories
        os.makedirs(self.rootdir, exist_ok=True)
        os.makedirs(self.procdir, exist_ok=True)
        os.makedirs(self.seldir, exist_ok=True)

        if self.sfs_opt.get('sfsdir') is None:
            self.sfs_opt.set('sfsdir', f"{self.procdir}sfs{self.sfs_opt.get('targetmpp')}_"
                                       f"sel{self.sel}_"
                                       f"{self.sfs_opt.get('sfs_smoothness_weight')}_"
                                       f"{self.sfs_opt.get('sfs_initial_dem_constraint_weight')}/")

        # check and update config files
        if self.sfs_opt.config_ba_path == "":
            with resources.path('sfs.preprocessing', 'default_config_ba.yaml') as yaml_path:
                self.sfs_opt.config_ba_path = str(yaml_path)

        # save to file
        self.sfs_opt.to_yaml(f"{self.procdir}final_config_{self.siteid}.yaml")

    def init_pipeline(self, **kwargs):

        # import and save index
        save_index_to = f"{self.rootdir}{self.sfs_opt.get('pds_index_name')}.parquet"
        print(save_index_to)
        if not os.path.exists(save_index_to):
            pds3_to_df(self.sfs_opt.get('pds_index_path'), self.sfs_opt.get('pds_index_name'),
                       save_to=save_index_to)
        else:
            assert os.path.exists(save_index_to)

    def configure_steps(self, steps_to_run):
        list_of_steps = ['rough_selection', 'verify_download', 'preprocess',
                         'clean_dataset', 'refine_align', 'final_selection',
                         'align_to_dem', 'sfs', 'postpro']
        # Set or update steps based on steps_to_run input
        if len(steps_to_run) == len(list_of_steps):
            steps_dict = dict(zip(list_of_steps, [bool(int(s)) for s in steps_to_run]))
            self.steps_to_run = steps_dict
        else:
            raise ValueError(f"Input string {steps_to_run} does not match the expected length of {len(list_of_steps)}.")

    def run_pipeline(self):

        siteid = self.siteid
        tileid = self.tileid
        sel = self.sel

        procdir = self.procdir
        seldir = self.seldir
        filtered_selection_path = f"{procdir}filtered_selection_{siteid}_{tileid}.csv"
        all_aligned_images_path = f"{procdir}all_aligned_{siteid}_{tileid}.parquet"
        not_aligned_images_path = f"{procdir}not_aligned_{siteid}_{tileid}.parquet"
        outstats = f"{procdir}check_ba/stats_{siteid}_{tileid}_exp.parquet"
        final_selection_path = f"{procdir}final_selection_{tileid}_sel{sel}.csv"

        logging.info("- Running ", self.steps_to_run)

        self.init_pipeline()

        if self.steps_to_run['rough_selection']:
            self._rough_selection()
        else:
            self.selout_path = self.sfs_opt.rootroot + self.sfs_opt.get('imglist_full')
        # update path to shapefile's path (if not defined in input)
        if self.sfs_opt.get('input_shp') == None:
            self.sfs_opt.set('input_shp', self.rootdir + f"clip_{self.siteid[:2]}{self.tileid}.shp")

        if self.steps_to_run['verify_download']:
            self._verify_download(self.selout_path)

        tmp_selection = pd.read_csv(f"{self.selout_path}")
        if self.steps_to_run['preprocess']:
            # print("!!! CAUTION, SERIAL!!!")
            filtered_tmp_selection = self._preprocess(tmp_selection, filtered_selection_path) #, parallel=False) #, with_ba=False)
        else:
            filtered_tmp_selection = pd.read_csv(filtered_selection_path)

        if self.steps_to_run['clean_dataset']:
            prefix = 'ba_iter0'
            bad_images, all_images = self._clean_dataset(mapproj_match_offset_stats_path=
                                                         f"{self.procdir}{prefix}/"
                                                         f"run-mapproj_match_offset_stats.txt",
                                                         threshold_85p=self.sfs_opt.get('threshold_bad_image'),
                                                         min_count=self.sfs_opt.min_count)
            if bad_images is not None:
                bad_images.to_parquet(not_aligned_images_path)
            all_images.to_parquet(all_aligned_images_path)

        if self.steps_to_run['refine_align']:
            self._refine_align(filtered_tmp_selection, not_aligned_images_path, outstats)

        if self.steps_to_run['final_selection']:
            final_selection = self._final_selection(not_aligned_images_path=not_aligned_images_path,
                                                    filtered_selection_path=filtered_selection_path,
                                                    final_selection_path=final_selection_path,
                                                    max_images_to_add=100)
        else:
            try:
                final_selection = pd.read_csv(final_selection_path)
            except:
                logging.error(f"* No final selection found. Exit.")
                exit()

        # choose
        os.chdir(seldir)
        self.latest_selection = final_selection

        if self.steps_to_run['align_to_dem']:

            if self.sel == 0:
                # prepare the transform file to register all BA images to the apriori DEM
                self._align_to_dem(base_resolution=self.sfs_opt.get('rendering_resolution'), new_rendering=True)
            else:
                reference_transform = f"{procdir}sel_0/ba_align/run-ecef-transform.txt"
                # simply apply the existing transform from sel0 (assumes that it exists)
                self._align_to_dem(base_resolution=self.sfs_opt.get('rendering_resolution'),
                                   use_existing_transform=reference_transform,
                                   new_rendering=False)


        if self.steps_to_run['sfs']:

            if self.sfs_opt.resample_images:
                cumindex = pd.read_csv(final_selection_path)

                logging.info('- Linking .cub/.json back to resampled cameras')
                for imgp in glob.glob(f"{procdir}*.IMG"):
                    img = os.path.basename(imgp).split('.IMG')[0]

                    if os.path.isfile(f"{procdir}{img}.cub") or os.path.islink(f"{procdir}{img}.cub"):
                        os.remove(f"{procdir}{img}.cub")
                    os.symlink(f"{procdir}{img}.cal.echo.red{self.sfs_opt.get('targetmpp')}.cub", f"{procdir}{img}.cub")

                    if self.sfs_opt.use_csm:

                        if os.path.isfile(f"{procdir}{img}.json") or os.path.islink(f"{procdir}{img}.json"):
                            os.remove(f"{procdir}{img}.json")
                        os.symlink(f"{procdir}{img}.model_state_red.json", f"{procdir}{img}.json")

            elif True:
                final_selection = pd.read_csv("/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/sfs/Mercury/MSP/proc/tile_0/filtered_selection_MSP_0.csv")
                bad_images = pd.read_parquet("/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/sfs/Mercury/MSP/proc/tile_0/not_aligned_MSP_0.parquet")
                # remove bad images
                self.latest_selection = final_selection.loc[
                    ~final_selection.img_name.isin(bad_images['img'])]
                print(f"We end up selecting {len(self.latest_selection)} from {len(final_selection)}.")
                print(self.latest_selection)
                for img in self.latest_selection.img_name.values:
                    if os.path.isfile(f"{procdir}sel_0/ba_align/run-{img}.adjust") or os.path.islink(f"{procdir}sel_0/ba_align/run-{img}.adjust"):
                        os.remove(f"{procdir}sel_0/ba_align/run-{img}.adjust")
                    os.symlink(f"/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/sfs/Mercury/MSP/proc/tile_0/ba/run-{img}.adjust",
                               f"{procdir}sel_0/ba_align/run-{img}.adjust")

                    if True:
                        if os.path.isfile(f"{procdir}{img}.cub") or os.path.islink(f"{procdir}{img}.cub"):
                            os.remove(f"{procdir}{img}.cub")
                        os.symlink(f"/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/sfs/Mercury/MSP/proc/tile_0/{img}.cal.cub",
                                   f"{procdir}{img}.cub")

                        if os.path.isfile(f"{procdir}{img}.cal.echo.red{self.sfs_opt.targetmpp}.cub") or os.path.islink(f"{procdir}{img}.cal.echo.red{self.sfs_opt.targetmpp}.cub"):
                            os.remove(f"{procdir}{img}.cal.echo.red{self.sfs_opt.targetmpp}.cub")
                        os.symlink(f"/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/sfs/Mercury/MSP/proc/tile_0/{img}.cal.echo.cub",
                                   f"{procdir}{img}.cal.echo.red{self.sfs_opt.targetmpp}.cub")

                        if os.path.isfile(f"{procdir}{img}.cal.cub") or os.path.islink(f"{procdir}{img}.cal.cub"):
                            os.remove(f"{procdir}{img}.cal.cub")
                        os.symlink(f"/panfs/ccds02/nobackup/people/sberton2/RING/code/sfs_helper/examples/sfs/Mercury/MSP/proc/tile_0/{img}.cal.cub",
                                   f"{procdir}{img}.cal.cub")

            # run sfs
            self._run_sfs()

        if self.steps_to_run['postpro']:
            self._postpro()

    def _rough_selection(self):
        start = time()
        self.selout_path = rough_imgsel_sites(self.tileid,
                                              boxcentersfil=self.sfs_opt.get('boxcentersfil'),
                                              lonlat=self.sfs_opt.get('lonlat'),
                                              input_shp=self.sfs_opt.get('input_shp'),
                                              input_tif=self.sfs_opt.get('input_tif'),
                                              latrange=self.sfs_opt.get('latrange'))[0]

        logging.info(f"- Rough selection completed after {time() - start} seconds.")
        logging.info(f"- Rough selection saved to {self.selout_path}.")

    def _verify_download(self, selout_path):
        start = time()
        tmp_selection = pd.read_csv(f"{selout_path}")
        print(self.sfs_opt.get('source_url'))
        verify_and_download(self.tileid, tmp_selection, exist_replace=False,
                            source_url=self.sfs_opt.get('source_url'), )
        logging.info(f"- Downloaded after {time() - start} seconds.")

    def _preprocess(self, tmp_selection, filtered_selection_path, with_ba=True, parallel=True):
        start = time()

        #tmp_selection = tmp_selection[:50]
        filtered_tmp_selection = preprocess(self.tileid, tmp_selection, with_ba=with_ba, parallel=parallel)  # !!!!
        #filtered_tmp_selection = pd.read_csv(filtered_selection_path)

        os.chdir(self.cwd)  # some ASP function seem to change dir...

        # update prj links to bundle-adjusted mapproj images
        for img in filtered_tmp_selection.img_name.values:
            if os.path.islink(f"{self.procdir}prj/{img}_map.tif"):
                os.remove(f"{self.procdir}prj/{img}_map.tif")
            os.symlink(f"{self.procdir}prj/ba_iter0/{img}_map.tif", f"{self.procdir}prj/{img}_map.tif")

        # save filtered selection to file
        filtered_tmp_selection.to_csv(filtered_selection_path)

        # symlink latest BA solution to ba folders
        dirs_to_move = {f"{self.procdir}ba_iter0": f"{self.procdir}ba",
                        f"{self.procdir}prj/ba_iter0": f"{self.procdir}prj/ba"}
        for dir_from, dir_to in dirs_to_move.items():
            if os.path.islink(dir_to):
                os.remove(dir_to)
            elif os.path.exists(dir_to):
                logging.warning(f"## {dir_to} dir exists and is not link. Maybe move it to {dir_from}?")
                shutil.rmtree(dir_to)
                #exit()
            os.symlink(dir_from, dir_to, target_is_directory=True)

        logging.info(f"- Preprocessing finalized after {time() - start} seconds.")

        return filtered_tmp_selection

    def _clean_dataset(self, mapproj_match_offset_stats_path, threshold_85p, min_count):
        start = time()

        bad_images, all_images = find_bad_images(mapproj_match_offset_stats_path,
                                                 threshold_85p, min_count)
        if bad_images is not None:
            logging.info(f"- Removed {len(bad_images)} bad images after {time() - start} seconds.")
            return bad_images, all_images
        else:
            logging.info(f"- Removed 0 bad images after {time() - start} seconds.")
            return None, all_images

    def _update_symlinks(self, filtered_tmp_selection, prefix):
        for img in filtered_tmp_selection.img_name.values:
            symlink_path = f"{self.procdir}prj/{img}_map.tif"
            new_symlink_target = f"{self.procdir}prj/{prefix}/{img}_map.tif"
            if os.path.islink(symlink_path):
                os.remove(symlink_path)
            os.symlink(new_symlink_target, symlink_path)

    def _refine_align(self, filtered_tmp_selection, not_aligned_images_path, outstats):
        start = time()

        set_asp(self.sfs_opt.aspdir)

        # mod for MSP tiles
        if False:
            if os.path.islink(f'{self.procdir}ba_iter0'):
                os.remove(f'{self.procdir}ba_iter0')
            os.symlink(f"{self.procroot}tile_0/ba_iter0", f'{self.procdir}ba_iter0', target_is_directory=True)
            if os.path.islink(f'{self.procdir}not_aligned_MSP_{self.tileid}.parquet'):
                os.remove(f'{self.procdir}not_aligned_MSP_{self.tileid}.parquet')
            os.symlink(f"{self.procroot}tile_0/ba_iter0/not_aligned_MSP_0.parquet", f'{self.procdir}not_aligned_MSP_{self.tileid}.parquet')

            if True:
             for img in filtered_tmp_selection.img_name.values:

                #if os.path.exists(f'{self.procdir}prj/ba_iter2/{img}_map.tif'):
                #    os.symlink(f"{self.procroot}tile_0/prj/ba_iter8/{img}_map.tif", f'{self.procdir}prj/ba_test/{img}_map.tif')

                adjfil = f'{self.procdir}ba_iter0/run-{img}.adjust'
                if not os.path.exists(adjfil):
                    # Open the file in write mode
                    with open(adjfil, "w") as file:
                        # Write the first line
                        file.write("0 0 0\n")
                        # Write the second line
                        file.write("1 0 0 0\n")
                    print(f"- Generated empty adjust file for {img}. Continue.")
            print("Done with MSP-specific tasks.")
            # exit()

        if False:
            # select only images in sunbin
            filtered_tmp_selection = filtered_tmp_selection.loc[
                filtered_tmp_selection.sol_lon.isin([90])]
            print(filtered_tmp_selection[['img_name', 'sol_lon']])

        ###### END mod

        iteration = 0

        if iteration == 0:
            shutil.copy(not_aligned_images_path, f'{self.procdir}ba_iter0/{os.path.basename(not_aligned_images_path)}')

        bad_images = pd.read_parquet(not_aligned_images_path)

        while len(bad_images) > 0:

            # increase iteration number
            iteration += 1

            # if already computed, skip
            not_aligned_next_iter = f'{self.procdir}ba_iter{iteration}/{os.path.basename(not_aligned_images_path)}'
            if os.path.exists(not_aligned_next_iter):
                print(f"{not_aligned_next_iter} already exists. Continue.")
                total_bad_images = []
                for not_aligned_images in [f'{self.procdir}ba_iter{x}/{os.path.basename(not_aligned_images_path)}'
                                           for x in range(0, iteration)]:
                    total_bad_images.append(pd.read_parquet(not_aligned_images))
                bad_images = pd.concat(total_bad_images)
                continue
            
            print(f'bad_images at iter #{iteration}')
            print(bad_images[['img', '85%', 'count']].sort_values(by='85%'))

            # remove bad images
            filtered_tmp_selection = filtered_tmp_selection.loc[
                ~filtered_tmp_selection.img_name.isin(bad_images['img'])]

            print(f"- We removed {len(bad_images)} bad images, we kept {len(filtered_tmp_selection)} selected ones. "
                  f"Re-running BA.")
            if len(filtered_tmp_selection) > 0:
                print(filtered_tmp_selection)
            else:
                print("** No images left after removing bad ones. Exit.")
                exit()
                
            # and produce max_lit mosaic
            #if not os.path.exists(f"{self.procdir}max_lit_ba_iter{iteration-1}_filter_{self.tileid}.tif"):
            #  dem_mosaic(imgs=[f"{self.procdir}prj/ba_iter{iteration-1}/{x}_map.tif"
            #                 for x in filtered_tmp_selection.img_name.values],
            #             dirnam=self.procdir, tr=self.sfs_opt.get('targetmpp'),
            #           max=None, output_prefix=f"max_lit_ba_iter{iteration-1}_filter_{self.tileid}.tif")

            # Run bundle adjustment and map projection
            current_prefix = f'ba_iter{iteration}'

            ba_and_mapproj(self.tileid, cumindex=filtered_tmp_selection,
                           use_mapproject=False,  # needs to be false when passing .adjust info
                           prioridem_path=f"{self.procdir}ldem_{self.tileid}.tif",
                           bundle_adjust_prefix=current_prefix,
                           input_adjustments_prefix=f'ba_iter{iteration - 1}/run',
                           clean_match_files_prefix=f'ba_iter{iteration - 1}/run'
                           )

            # Update symlinks
            self._update_symlinks(filtered_tmp_selection, current_prefix)

            # Analyze results and determine bad images
            validation_prefix = current_prefix  # place-holder for compatibility
            bad_images, all_images = self._clean_dataset(mapproj_match_offset_stats_path=
                                                         f"{self.procdir}{validation_prefix}/"
                                                         f"run-mapproj_match_offset_stats.txt",
                                                         threshold_85p=self.sfs_opt.get('threshold_bad_image'),
                                                         min_count=self.sfs_opt.min_count)

            if not os.path.exists(f'{self.procdir}{validation_prefix}/'):
                os.makedirs(f'{self.procdir}{validation_prefix}/')

            if bad_images is not None:
                bad_images.to_parquet(f'{self.procdir}{validation_prefix}/{os.path.basename(not_aligned_images_path)}')
                print(f"- New run found {len(bad_images)} bad images.")

            else:
                print("- Done iterating, no bad images found.")
                bad_images = []

            os.chdir(self.cwd)

        #iteration = 2
        #print(f"## Starting from here at iter {iteration}.")
            
        # Copy final result to the reference "ba" files
        final_prefix = f'ba_iter{iteration}'
        dirs_to_move = {f"{self.procdir}{final_prefix}": f"{self.procdir}ba",
                        f"{self.procdir}prj/{final_prefix}": f"{self.procdir}prj/ba"}

        # Concatenate bad_images lists
        total_bad_images = []
        for not_aligned_images in [f'{self.procdir}ba_iter{x}/{os.path.basename(not_aligned_images_path)}'
                                   for x in range(0, iteration)]:
            total_bad_images.append(pd.read_parquet(not_aligned_images))

        total_bad_images_path = f'{self.procdir}{os.path.basename(not_aligned_images_path)}'
        if len(total_bad_images) > 0:
            total_bad_images = pd.concat(total_bad_images)
            #total_bad_images = pd.DataFrame(columns=not_aligned_images.columns)
            total_bad_images.to_parquet(total_bad_images_path)
        else:
            if os.path.isfile(total_bad_images_path):
                os.remove(f'{self.procdir}{os.path.basename(not_aligned_images_path)}')
            os.symlink(f'{self.procdir}ba_iter0/{os.path.basename(not_aligned_images_path)}', total_bad_images_path)

        print(f"total bad images {len(total_bad_images)}/{len(filtered_tmp_selection)}",
              total_bad_images)
        filtered_tmp_selection = filtered_tmp_selection.loc[
                ~filtered_tmp_selection.img_name.isin(total_bad_images['img'])]
        print(f"good images {len(filtered_tmp_selection)}")

        
        for dir_from, dir_to in dirs_to_move.items():
            if os.path.islink(dir_to):
                os.remove(dir_to)
            elif os.path.exists(dir_to):
                logging.warning(f"## {dir_to} dir exists and is not link. Maybe move it to {dir_from}?")
                exit()
            os.symlink(dir_from, dir_to, target_is_directory=True)

        # get bundle-adjusted mapproj images and maxlit
        mapproj_path = mapproj_maxlit(self.tileid, cumindex=filtered_tmp_selection,
                                      prioridem_path=f"{self.procdir}ldem_{self.tileid}.tif",
                                      bundle_adjust_prefix=final_prefix,
                                      input_adjustments_prefix=f'ba_iter{iteration}/run',
                                      # parallel=False,
                                      )
            
        logging.info(f"- Refined alignments after {time() - start} seconds.")

    def _final_selection(self, final_selection_path, filtered_selection_path, max_images_to_add=35, **kwargs):
        start = time()

        preliminary_selection_paths = select(self.tileid,
                                             num_sel=5,
                                             frac_diff=0.5,
                                             exclude=self.sfs_opt.get("imgs_to_remove")[f"{self.tileid}"],
                                             **kwargs)
        #else:
        #    preliminary_selection_paths = [f"{self.procdir}final_selection_{self.tileid}_sel{x}.csv" for x in range(20)]
            
        # refine selection
        all_merged = pd.read_parquet(f"{self.procdir}all_merged_{self.tileid}.parquet")
        selection = pd.read_csv(preliminary_selection_paths[self.sel], sep=',')
        # selection = pd.read_csv(f'{self.procdir}final_selection_0_sel{self.sel}.csv')
        rough_selection = pd.read_csv(filtered_selection_path, sep=',')

        # only include images that have been bundle_adjusted
        rough_selection = filter_imgsel(self.procdir, rough_selection, **kwargs)

        additional_selection = improve_selection(selection=selection, all_merged=all_merged,
                                                 rough_selection=rough_selection, )
        final_selection = improve_selection_sunlon(selection=additional_selection, all_merged=all_merged,
                                                   rough_selection=rough_selection, max_images_to_add=max_images_to_add)
        final_selection.to_csv(final_selection_path)

        logging.info(f"- Final selection completed after {time() - start} seconds.")

        return final_selection

    def _align_to_dem(self, **kwargs):
        start = time()
        align_from_render(self.tileid, self.sel,
                          self.latest_selection.loc[:, "img_name"].values,
                          **kwargs)

        for img in self.latest_selection.loc[:, "img_name"].values:
            adjustfile = self.seldir+f"ba_align/run-{img}.adjust"
            assert os.path.exists(adjustfile), f"* {adjustfile} not found. Stop."

        logging.info(f"- Alignment to DEM finalized after {time() - start} seconds.")

    def _run_sfs(self):
        start = time()
        sfs_on_tile(self.tileid, self.sel, self.latest_selection)
        logging.info(f"- Sfs finalized after {time() - start} seconds.")

    def _postpro(self):
        start = time()
        postpro(self.tileid, self.sel, self.latest_selection, use_rough_sel=False, use_priori_dem=False)

        xyzi = f"{self.procdir}{os.path.basename(self.sfs_opt.xyzi_full).split('.XYZI')[0]}_XYZI_crop.parquet"
        prodsdir = f"{self.seldir}products/"
        siteid = f"{self.sfs_opt.get('site')[:2]}{self.tileid:02d}"

        check_alignment(self.tileid, self.sel, self.sfs_opt.xyzi_full, xyzi,
                        self.sfs_opt.input_shp,
                        #priormap=f"{self.procdir}ldem_{self.tileid}.tif",
                        priormap=f"{self.procdir}ldem_{self.tileid}.tif",
                        sfsmap=f"{prodsdir}{siteid}_GLDELEV_001.tif",
                        mask_shadows=f"{prodsdir}{siteid}_GLDOMOS_001.tif",
                        shadow_threshold=0.005,
                        target_resolution=self.sfs_opt.get('targetmpp'), default_smoothing_resol=5)
        logging.info(f"- Postpro finalized after {time() - start} seconds.")

    def initialize(self, init_fun):
        # Any initialization logic goes here
        pass

    def finalize(self, final_fun):
        # Any finalization logic goes here
        pass
