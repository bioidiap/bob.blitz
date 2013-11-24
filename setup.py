#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['numpy', 'xbob.extension']))
import numpy
from xbob.extension import Extension

# Local include directory
import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'xbob', 'blitz', 'include')

# Add numpy includes
extra_compile_args = ['-isystem', numpy.get_include()]

# NumPy API macros necessary?
define_macros=[
    ("PY_ARRAY_UNIQUE_SYMBOL", "XBOB_BLITZ_NUMPY_C_API"),
    ("NO_IMPORT_ARRAY", "1"),
    ]
from distutils.version import StrictVersion
if StrictVersion(numpy.__version__) >= StrictVersion('1.7'):
  define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))

# Define package version
version = '0.0.1'

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='xbob.blitz',
    version=version,
    description='Bindings for Blitz++ (a C++ array template library)',
    url='http://github.com/anjos/blitz.array',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    namespace_packages = [
      'xbob',
    ],

    install_requires=[
      'setuptools',
      'numpy',
      'xbob.extension',
    ],

    ext_modules = [
      Extension("xbob.blitz._library",
        [
          "xbob/blitz/api.cpp",
          "xbob/blitz/array.cpp",
          "xbob/blitz/main.cpp",
          ],
        packages=[
          'blitz >= 0.10',
          ],
        version=version,
        define_macros=define_macros,
        include_dirs=[package_dir],
        extra_compile_args=extra_compile_args,
        )
      ],

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],
 
    )
