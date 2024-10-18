#!/usr/bin/python

import os, sys
import json
import ale

from asp.functions import cam_test, set_asp


def gen_csm_frame(prefix):
    if prefix.lower().endswith(".cub") or prefix.lower().endswith(".img") \
        or prefix.lower().endswith(".lbl"):
        # Wipe extension
        prefix = os.path.splitext(prefix)[0]

    print("Prefix is: " + prefix)

    cub_file = prefix + '.cub'
    img_file = prefix + '.IMG'

    kernels = ale.util.generate_kernels_from_cube(cub_file, expand = True)

    usgscsm_str = ale.loads(img_file, props={'kernels': kernels},
                            formatter='ale', verbose = False)

    csm_isd = prefix + '.json'
    print("Writing: " + csm_isd)
    with open(csm_isd, 'w') as isd_file:
        isd_file.write(usgscsm_str)

def gen_csm_line(cub_file):
    #!/usr/bin/python

    # Get the input cub
    #cub_file = sys.argv[1]

    # Form the output cub
    #isd_file = os.path.splitext(cub_file)[0] + '.json'
    isd_file = cub_file.split('.')[0] + '.json'

    print("Reading: " + cub_file)
    usgscsm_str = ale.loads(cub_file)

    print("Writing: " + isd_file)
    with open(isd_file, 'w') as isd_file:
        isd_file.write(usgscsm_str)

if __name__ == '__main__':

    prefix = sys.argv[1]
    dirnam = sys.argv[2]

    set_asp("/home/sberton2/nobackup/.local/opt/StereoPipeline-3.3.0-Linux/")
    # set_asp("/home/sberton2/.local/share/StereoPipeline-3.3.0-Linux/")

    gen_csm_line(prefix+'.cal.cub')
    cam_test(dirnam, prefix, sample_rate=100)
    gen_csm_frame(prefix)
    cam_test(dirnam, prefix, sample_rate=100)
