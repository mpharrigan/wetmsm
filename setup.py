from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    name='WetMSM',
    version="0.2",
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize("wetmsm/*.pyx"),
    packages=find_packages(),
    zip_safe=False
)
