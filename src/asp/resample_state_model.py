import json
import numpy as np

from asp.functions import usgscsm_cam_test, set_asp, cam_test
from isis.functions import set_isis


def scale_json_parameters(input_path, output_path, scale_factor):
    """
    Scales specific parameters in a JSON file and writes the modified content to a new file.

    Args:
    - input_path (str): Path to the input JSON file.
    - output_path (str): Path to save the modified JSON file.
    - scale_factor (float): The factor by which to scale the specified parameters.

    Returns:
    - None
    """

    with open(input_path, 'r') as json_data:
        header = json_data.readline()
        data = json.load(json_data)

    # Apply scale_factor to the specified parameters
    data['m_intTimes'] = [value * scale_factor for value in data['m_intTimes']]
    data['m_nLines'] = data['m_nLines'] / scale_factor
    data['m_nSamples'] = data['m_nSamples'] / scale_factor
    data['m_detectorLineOrigin'] = data['m_detectorLineOrigin'] / scale_factor
    data['m_detectorSampleOrigin'] = data['m_detectorSampleOrigin'] / scale_factor
    data['m_focalLength'] = data['m_focalLength'] / scale_factor
    data['m_opticalDistCoeffs'] = [coeff * (scale_factor ** 2) for coeff in data['m_opticalDistCoeffs']]

    # Save the modified JSON to a new file
    with open(output_path, 'w') as file:
        file.write(header)
        json.dump(data, file, indent=2)

if __name__ == '__main__':

    # Example usage
    imgid = 'M129674288LE'
    procdir = '/home/sberton2/Lavoro/code/geotools/examples/sfs/Moon/A3CLS/proc/tile_6/'
    camera_path = f"{procdir}{imgid}.cub"
    model_state_path = f'{procdir}run-{imgid}.model_state.json'
    output_path = f'{procdir}run-{imgid}.model_state_red.json'
    scale_factor = 10

    aspdir = "/home/sberton2/.local/share/StereoPipeline-3.3.0-Linux/"
    isisdir = ("/home/sberton2/.local/share/spack/opt/spack/linux-linuxmint21-skylake/gcc-11.4.0/"
               "miniconda3-24.3.0-ynid3mehhgxzwhiqvdroz7r4mn26rwky/envs/isis")
    isisdata = "/home/sberton2/Downloads/ISISDATA/"

    set_asp(aspdir)
    set_isis(isisdir, isisdata)

    usgscsm_cam_test(input_camera_model=camera_path, output_model_state=model_state_path,
                     dirnam=procdir)
    cam_test(procdir, imgid, cams=[camera_path, model_state_path], sample_rate=100)

    scale_json_parameters(model_state_path, output_path, scale_factor)
