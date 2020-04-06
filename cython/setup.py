from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize('cpython_demo.py'))
