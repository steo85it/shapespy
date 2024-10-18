import os
import logging

from asp.functions import gdal_translate, sfs, set_asp
from isis.functions import reduce as isis_reduce, set_isis
from sfs.config import SfsOpt
from sfs.preprocessing.preprocessing import load_project_img, load_calibrate_project_img

# aspdir = SfsOpt.get("aspdir")
# isisdir = SfsOpt.get("isisdir")
# isisdata = SfsOpt.get("isisdata")


def sfs_on_tile(tileid, sel, cumindex, **kwargs):

    opt = SfsOpt.get_instance()

    set_asp(opt.aspdir)
    set_isis(opt.isisdir, opt.isisdata)
    
    procdir = f"{opt.procroot}tile_{tileid}/"
    seldir = f"{procdir}sel_{sel}/"

    input_imgs = [x for x in cumindex.loc[:, 'img_name'].values]
    #input_imgs = [x for x in cumindex.loc[:, 'img_name'].values if x not in ['M1245317931RE']] #,'M1121385267RE','M1141648573RE','M1121392383LE']]
    print(input_imgs)

    prioridem_path = f"{procdir}ldem_{tileid}.tif"
    # prioridem_path = f"{procdir}sfs/run-DEM-final.tif"

    for index, row in cumindex.iterrows():
        img = row.img_name

        load_calibrate_project_img((index,row), tileid, prioridem_path, bundle_adjust_prefix=None, project_imgs=False)

        if not os.path.isfile(f"{procdir}{img}.cal.echo.red{opt.targetmpp}.cub"):
            logging.error(f"- {img}.cal.echo.red{opt.targetmpp}.cub does not exist. Check.")
            exit()
        #    if False and opt.use_csm:
        #        os.symlink(f"{img}.cal.echo.cub", f"{img}.cal.echo.red{opt.targetmpp}.cub")
        #    elif False: #se:
        #        MPP = row['RESOLUTION']
        #        redfact = max(1, int(opt.targetmpp / MPP))
        #        print("- Generating reduced camera: ", redfact, opt.targetmpp, MPP)                
        #        isis_reduce(from_=f"{img}.cal.echo.cub", to=f"{img}.cal.echo.red{opt.targetmpp}.cub", algorithm="average",
        #                dirnam=procdir, sscale=redfact, lscale=redfact)

        # # symlink for (reduced) cub
        # if os.path.islink(f"{procdir}{img}.cub"):
        #     os.remove(f"{procdir}{img}.cub")
        # os.symlink(f"{procdir}{img}.cal.echo.red{opt.targetmpp}.cub", f"{procdir}{img}.cub")

        # symlink for adjust (NOT if getting alignment from align_dem_with_hs)
        # if os.path.exists(f"{procdir}ba/run-{img}.adjust"):
        #    os.remove(f"{procdir}ba/run-{img}.adjust")
        # os.symlink(f"{procdir}ba/run-{img}.cal.echo.ba.adjust", f"{procdir}ba/run-{img}.adjust")

    # Run sfs at desired resolution and with full selection of images
    gdal_translate(procdir, filin=prioridem_path,
                   filout=f"{procdir}ldem_{tileid}_{opt.targetmpp}mpp.tif",
                   tr=f"{opt.targetmpp} {opt.targetmpp}", r='cubicspline')

    out_prefix = opt.sfsdir

    if True:
        sfs(input_imgs[:], dem=f"{procdir}ldem_{tileid}_{opt.targetmpp}mpp.tif",
        parallel=True,
            out_prefix=f'{out_prefix}run', in_ext='.cub', dirnam=procdir,
            reflectance_type=1, smoothness_weight=opt.sfs_smoothness_weight,
            initial_dem_constraint_weight=opt.sfs_initial_dem_constraint_weight,
            max_iterations=15,
            model_shadows=None,  # use_approx_camera_models=None,
            bundle_adjust_prefix=f"{seldir}ba_align/run", save_dem_with_nodata=None,
            crop_input_images=None,
            # save_sparingly=None, # will not save intermediate iteration files
            shadow_threshold=opt.shadow_threshold,
            robust_threshold=0.005, # shadow_threshold was set to 0.004 for NPA
            nodes_list=opt.nodes_list,
            # image_exposures_prefix=f'{out_prefix}/run',              #uses previous exposures (crashes if not available)
            allow_borderline_data=None,
            blending_dist=100, min_blend_size=20, # should smooth blending weights around nodata blobs larger than min_blend_size
            # tile_size=700, padding=100,  # test with larger tiles
            resume=None,                                             #will reuse previous sfs tiles if available
            use_csm=opt.use_csm,
            tif_compress='Deflate'
            )



    if not os.path.exists(f'{out_prefix}run-DEM-final.tif'): # had {procdir} in filepath but was causing to fail
        print("** sfs did non produce the expected output. Stop and check.")
        exit()

    if True:
        # estimate height errors
        sfs(input_imgs, dem=f'{out_prefix}run-DEM-final.tif',
            estimate_height_errors=None,
            out_prefix=f'{out_prefix}run-error',
            parallel=True,
            nodes_list=opt.nodes_list,
            dirnam=procdir, in_ext='.cub',
            resume=None,
            image_exposures_prefix=f'{out_prefix}/run', # crashes if not available (it should be from the sfs run)
            use_csm=opt.use_csm,
            tif_compress='Deflate')
