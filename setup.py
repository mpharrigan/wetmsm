from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    name='WetMSM',
    version="0.3",
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize("wetmsm/*.pyx"),
    packages=find_packages(),
    zip_safe=False,
    entry_points={'msmbuilder.commands': [
        'wetmsm1 = wetmsm.shells:SolventShellsFeaturizerCommand',
        'wetmsm2 = wetmsm.shells:SolventShellsAssignerCommand'
    ]}
)
