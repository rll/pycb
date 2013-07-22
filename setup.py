from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension(
        "c_pycb",
        ["src/c_pycb.pyx"],
        language="c",
        include_dirs=[numpy.get_include()],
    )
]

setup(
  name = 'pycb',
  version='0.0.1',
  cmdclass = {'build_ext': build_ext},
  ext_package = 'pycb',
  ext_modules = ext_modules,
  packages= ['pycb'],
)
