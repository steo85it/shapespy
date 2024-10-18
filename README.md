# SFS processing helper functions and ASP+ISIS wrappers
[![DOI](https://zenodo.org/badge/319753409.svg)]()

Python suite of helper functions for Shape-from-Shading processing, including wrappers to the Ames Stereo Pipeline (ASP) 
and USGS Integrated Software for Imagers and Spectrometers (ISIS). 
 
## Disclaimer

This is scientific code in ongoing development: using it might be tricky, reading it can cause 
 headaches and results need to be thoroughly checked and taken with an healthy degree of mistrust!
Use it at your own risk and responsibility. 
 ## Installation ##

### Set up a virtual environment and clone the repository ###

Make a new directory and clone this repository to it. Then, inside the
directory that you've just created, run `python -m venv env`. This will
create a "virtual environment", which is a useful tool that Python
provides for managing dependencies for projects. The new directory
"contains" the virtual environment.

### Activate the virtual environment ###

To activate the virtual environment, run `source env/bin/activate` from
the directory containing it. Make sure to do this before doing
anything else below.

### Getting the dependencies ###

Install the rest of the dependencies by running `pip install -r
requirements.txt`.

### Installing this package ###

Finally, run:
``` shell
pip install .
```
To install the package in editable mode, so that changes in this
directory propagate immediately, run:
``` shell
pip install -e .
```
To test the installation, from the project directory, run:
``` shell
python setup.py test
```
