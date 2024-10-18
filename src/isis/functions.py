import os
from utils.slurm_tools import call_script


def set_isis(dir, data=None):
    global isisdir
    isisdir = dir
    os.environ["ISISROOT"] = isisdir

    if data != None:
        os.environ["ISISDATA"] = data


def calibrate_mdis(imgnam, dirnam):
    # calibrate images using isis routines
    call_script(dirnam=dirnam, script=f'{isisdir}bin/mdis2isis',
                args=[f"from={imgnam}.IMG", f"to={imgnam}.cub"], is_slurm=False)
    call_script(dirnam=dirnam, script=f'{isisdir}bin/spiceinit',
                args=[f"from={imgnam}.cub", "spksmithed=true", "cksmithed=true"], is_slurm=False)
#                args=[f"from={imgnam}.cub", "spksmithed=false", "cksmithed=false"], is_slurm=False)                
#                args=[f"from={imgnam}.cub", "spk=/home/emazaric/nobackup/MESSENGER/data/spk/msgr_20040803_20150430_od431sc_0.bsp", "cksmithed=true"], is_slurm=False)
    call_script(dirnam=dirnam, script=f'{isisdir}bin/mdiscal',
                args=[f"from={imgnam}.cub", f"to={imgnam}.cal.cub"], is_slurm=False)
    os.remove(f"{dirnam}{imgnam}.cub")


def calibrate_lrocnac(imgnam, dirnam):
    # calibrate images using isis routines
    call_script(dirnam=dirnam, script=f'{isisdir}bin/lronac2isis',
                args=[f"from={imgnam}.IMG", f"to={imgnam}.cub"], is_slurm=False)
    call_script(dirnam=dirnam, script=f'{isisdir}bin/spiceinit',
                args=[f"from={imgnam}.cub", "spksmithed=true", "cksmithed=true", "web=false"], is_slurm=False)
    call_script(dirnam=dirnam, script=f'{isisdir}bin/lronaccal',
                args=[f"from={imgnam}.cub", f"to={imgnam}.cal.cub"], is_slurm=False)
    call_script(dirnam=dirnam, script=f'{isisdir}bin/lronacecho',
                args=[f"from={imgnam}.cal.cub", f"to={imgnam}.cal.echo.cub"], is_slurm=False)
    os.remove(f"{dirnam}{imgnam}.cal.cub")
    os.remove(f"{dirnam}{imgnam}.cub")


def calibrate_gllssi(imgnam, dirnam, smithed=False):

    # calibrate images using isis routines
    call_script(dirnam=dirnam, script=f'{isisdir}bin/gllssi2isis',
                args=[f"from={imgnam}.lbl", f"to={imgnam}.cub"], is_slurm=False)
    if smithed:
        call_script(dirnam=dirnam, script=f'{isisdir}bin/spiceinit',
                    args=[f"from={imgnam}.cub", "spksmithed=true", "cksmithed=true", "web=false"], is_slurm=False)
    else:
        call_script(dirnam=dirnam, script=f'{isisdir}bin/spiceinit',
                    args=[f"from={imgnam}.cub", "spksmithed=false", "cksmithed=false", "web=false"], is_slurm=False)
    call_script(dirnam=dirnam, script=f'{isisdir}bin/gllssical',
                args=[f"from={imgnam}.cub", f"to={imgnam}.cal.cub"], is_slurm=False)
    os.remove(f"{imgnam}.cub")


def calibrate_voy(imgnam, dirnam, smithed=False):

    # calibrate images using isis routines
    call_script(dirnam=dirnam, script=f'{isisdir}bin/voy2isis',
                args=[f"from={imgnam}.imq", f"to={imgnam}.cub"], is_slurm=False)
    if smithed:
        call_script(dirnam=dirnam, script=f'{isisdir}bin/spiceinit',
                    args=[f"from={imgnam}.cub", "spksmithed=true", "cksmithed=true", "web=false"], is_slurm=False)
    else:
        call_script(dirnam=dirnam, script=f'{isisdir}bin/spiceinit',
                    args=[f"from={imgnam}.cub", "spksmithed=false", "cksmithed=false", "web=false"], is_slurm=False)
    call_script(dirnam=dirnam, script=f'{isisdir}bin/voycal',
                args=[f"from={imgnam}.cub", f"to={imgnam}.cal.cub"], is_slurm=False)
    os.remove(f"{imgnam}.cub")

def phocube(dirnam, **kwargs):
    print(kwargs)

    args = []
    for key, value in kwargs.items():
        if key == "from_":
            args = args + [f"from={value}"]
        else:
            args = args + [f"{key}={value}"]

    print(args)
    # get photometry info as cube layers
    call_script(dirnam, script=f'{isisdir}bin/phocube',
                args=args, is_slurm=False)

def findfeatures(dirnam, **kwargs):
    print(kwargs)

    args = []
    for key, value in kwargs.items():
        if key == "from_":
            args = args + [f"from={value}"]
        else:
            args = args + [f"{key}={value}"]

    print(args)
    # get map-projected images with asp
    call_script(dirnam, script=f'{isisdir}bin/findfeatures',
                args=args, is_slurm=False)

def maptrim(dirnam, **kwargs):
    print(kwargs)

    args = []  # f"from={imgnam}.cal.echo.cub", f"to={imgnam}.cal.echo.red.cub", f"algorithm={algorithm}"]
    for key, value in kwargs.items():
        if key == "from_":
            args = args + [f"from={value}"]
        else:
            args = args + [f"{key}={value}"]

    print(args)
    # get map-projected images with asp
    call_script(dirnam, script=f'{isisdir}bin/maptrim',
                args=args, is_slurm=False)

def dstripe(dirnam, **kwargs):
    print(kwargs)

    args = []  # f"from={imgnam}.cal.echo.cub", f"to={imgnam}.cal.echo.red.cub", f"algorithm={algorithm}"]
    for key, value in kwargs.items():
        if key == "from_":
            args = args + [f"from={value}"]
        else:
            args = args + [f"{key}={value}"]

    print(args)
    # get map-projected images with asp
    call_script(dirnam, script=f'{isisdir}bin/dstripe',
                args=args, is_slurm=False)

def reduce(dirnam, **kwargs):
    print(kwargs)

    args = []  # f"from={imgnam}.cal.echo.cub", f"to={imgnam}.cal.echo.red.cub", f"algorithm={algorithm}"]
    for key, value in kwargs.items():
        if key == "from_":
            args = args + [f"from={value}"]
        else:
            args = args + [f"{key}={value}"]

    print(args)
    # get map-projected images with asp
    call_script(dirnam, script=f'{isisdir}bin/reduce',
                args=args, is_slurm=False)

def cam2map(dirnam, **kwargs):
    print(kwargs)

    args = []  # f"from={imgnam}.cal.echo.cub", f"to={imgnam}.cal.echo.red.cub", f"algorithm={algorithm}"]
    for key, value in kwargs.items():
        if key == "from_":
            args = args + [f"from={value}"]
        else:
            args = args + [f"{key}={value}"]

    print(args)
    # get map-projected images with asp
    call_script(dirnam, script=f'{isisdir}bin/cam2map',
                args=args, is_slurm=False)
