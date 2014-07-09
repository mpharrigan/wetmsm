from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(
        name='WetMSM',
        #ext_modules = cythonize("cysingmodel/_ising.pyx"),
        packages=['wetmsm']
)
