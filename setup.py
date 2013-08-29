#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    name='xbob.blitz',
    version='0.0.1a0',
    description='Cython bindings for Blitz++ (a C++ array template library)',

    url='http://pypi.python.org/pypi/plytz',
    license='GPLv3',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
    ],

    namespace_packages = [
      'xbob',
    ],

    cmdclass = {'build_ext': build_ext},
    ext_modules = [
      Extension("xbob.blitz.array", ["xbob/blitz/array.pyx"], language="c++")
      ],

    entry_points={
      },

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Cython',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],
)
