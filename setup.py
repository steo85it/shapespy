import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='geotools',
    version='0.2.0',
    author='Stefano Bertone',
    author_email='stefano.bertone@umd.edu',
    description='A suite of geospatial tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/steo85it/geotools',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"), # where="./geotools"),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
    ],
    extras_require={
        'dev': ['pytest', 'check-manifest'],
        'test': ['coverage'],
    },
    include_package_data=True,
    package_data={
        # If any package contains data files, include them here
        # 'mypackage': ['data/*.dat'],
    },
    entry_points={
        'console_scripts': [
            # If you have any scripts you want to be installed as executables
            # 'script-name = mypackage.module:function',
        ],
    },
)

