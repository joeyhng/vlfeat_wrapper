from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


ext_modules = [
Extension("gmm", ["gmm.pyx"], libraries=["vl", "m"], include_dirs=[np.get_include()], language='c'),
Extension("kmeans", ["kmeans.pyx"], libraries=["vl", "m"], include_dirs=[np.get_include()], language='c'),
Extension("fisher", ["fisher.pyx"], libraries=["vl", "m"], include_dirs=[np.get_include()], language='c'),
]

setup(name = 'vl_wrap', 
        cmdclass = {"build_Ext": build_ext},
        ext_modules = cythonize(ext_modules))

'''
Extension("gmm_wrap", ["gmm_wrap.pyx"], libraries=["vl", "m"], include_dirs=[np.get_include()], language='c'),
Extension("fisher_wrap", ["fisher_wrap.pyx"], libraries=["vl", "m"], include_dirs=[np.get_include()], language='c'),
'''
