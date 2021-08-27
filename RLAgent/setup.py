import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import pandas

# setup(ext_modules=(cythonize(Extension("FirstAgent", ["Agent.pyx"]))),
#       include_dirs=[numpy.get_include()],
#       requires=['numpy', 'Cython'])

setup(ext_modules=(cythonize(Extension("Agent", ["Agent.pyx"]))),
      include_dirs=[numpy.get_include()],
      requires=['numpy', 'Cython'])