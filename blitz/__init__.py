#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 20 Sep 14:45:01 2013

"""Blitz++ Array bindings for Python"""

from ._library import array, as_blitz, __version__, __api_version__

def get_include():
  """Returns the directory containing the C/C++ API include directives"""

  return __import__('pkg_resources').resource_filename(__name__, 'include')

def get_numpy_api():
  """Returns the name of the numpy API used for compilation"""

  from ._library import __numpy_api_name__
  return __numpy_api_name__

__all__ = ['array', 'as_blitz']
