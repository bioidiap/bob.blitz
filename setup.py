#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
from distutils.extension import Extension
import subprocess
import numpy

def pkgconfig(package):

  def uniq(seq, idfun=None):
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

  flag_map = {
      '-I': 'include_dirs',
      '-L': 'library_dirs',
      '-l': 'libraries',
      }

  cmd = [
      'pkg-config',
      '--libs',
      '--cflags',
      package,
      ]

  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT)

  output = proc.communicate()[0]
  if isinstance(output, bytes) and not isinstance(output, str):
    output = output.decode('utf8')

  if proc.returncode != 0: return {}

  kw = {}

  for token in output.split():
    if token[:2] in flag_map:
      kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])

    else: # throw others to extra_link_args
      kw.setdefault('extra_compile_args', []).append(token)

  for k, v in kw.items(): # remove duplicated
      kw[k] = uniq(v)

  return kw

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'xbob', 'blitz')
blitz_config = pkgconfig('blitz')
include_dirs = blitz_config.get('include_dirs', []) + \
    [numpy.get_include(), package_dir]

# NumPy API macros necessary?
define_macros=[
    ("PY_ARRAY_UNIQUE_SYMBOL", "BOB_NUMPY_ARRAY_API"),
    ("NO_IMPORT_ARRAY", "1"),
    ]
import numpy
from distutils.version import StrictVersion
if StrictVersion(numpy.__version__) >= StrictVersion('1.7'):
  define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))

# Compilation options
import platform
extra_compile_args=[
  '-O0', '-g',
  ]
if platform.system() == 'Darwin':
  extra_compile_args += ['-std=c++11', '-stdlib=libc++', '-Wno-#warnings']
else:
  extra_compile_args += ['-std=c++11']

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='xbob.blitz',
    version='0.0.1a0',
    description='Bindings for Blitz++ (a C++ array template library)',

    url='http://pypi.python.org/pypi/plytz',
    license='GPLv3',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'numpy',
    ],

    namespace_packages = [
      'xbob',
    ],

    #cmdclass = {'build_ext': build_ext},
    ext_modules = [
      Extension("xbob.blitz.array2",
        [
          "xbob/blitz/bob/py.cpp",
          "xbob/blitz/array2.cpp",
          ],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        define_macros=define_macros,
        )
      ],

    entry_points={
      },

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],)
