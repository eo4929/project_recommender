'''
from distutils.core import setup
from Cython.Build import cythonize

setup( ext_modules = cythonize("similarity.pyx") )  
'''
from setuptools import setup, find_packages, Extension
from codecs import open
from os import path

import numpy as np

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

here = path.abspath(path.dirname(__file__))

cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension(
        'similarity',
        ['similarity' + ext],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'matrixFactorization',
        ['matrixFactorization' + ext],
        include_dirs=[np.get_include()]),
    Extension('optimizing_baseline',
              ['optimizing_baseline' + ext],
              include_dirs=[np.get_include()]),
]

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = extensions

setup(
    name='Assignment-recommender',
    author='Dae Young',
    author_email='eo4929@naver.com',
    keywords='recommender system',
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)