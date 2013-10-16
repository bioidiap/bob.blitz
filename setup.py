#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
from distutils.extension import Extension

dist.Distribution(dict(setup_requires=['pypkg', 'numpy']))
import pypkg
import numpy

# Pkg-config dependencies
blitz = pypkg.pkgconfig('blitz')

# Local include directory
import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'blitz', 'include')
include_dirs = [package_dir]

# Add system include directories
extra_compile_args = []
system_includes = blitz.include_directories() + [numpy.get_include()]
for k in system_includes: extra_compile_args += ['-isystem', k]

# NumPy API macros necessary?
define_macros=[
    ("PY_ARRAY_UNIQUE_SYMBOL", "BLITZ_NUMPY_ARRAY_API"),
    ("NO_IMPORT_ARRAY", "1"),
    ]
import numpy
from distutils.version import StrictVersion
if StrictVersion(numpy.__version__) >= StrictVersion('1.7'):
  define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))

# Compilation options
import platform
#extra_compile_args += ['-O0', '-g']
if platform.system() == 'Darwin':
  extra_compile_args += ['-std=c++11', '-stdlib=libc++', '-Wno-#warnings']
else:
  extra_compile_args += ['-std=c++11']

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='blitz.array',
    version='0.0.1',
    description='Bindings for Blitz++ (a C++ array template library)',
    url='http://github.com/anjos/blitz.array',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'numpy',
    ],

    ext_modules = [
      Extension("blitz._array",
        [
          "blitz/src/api.cpp",
          "blitz/src/array.cpp",
          ],
        define_macros=define_macros,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        library_dirs=blitz.library_directories(),
        libraries=blitz.libraries(),
        language="c++",
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
