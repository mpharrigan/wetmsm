from setuptools import setup, find_packages
from Cython.Build import cythonize


setup(
    name='WetMSM',
    version="0.2",
    ext_modules=cythonize("wetmsm/*.pyx"),
    packages=find_packages(),
    zip_safe=False
)
