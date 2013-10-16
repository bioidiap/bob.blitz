#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 20 Sep 14:45:01 2013

"""Blitz++ Array bindings for Python"""

from ._array import array, as_blitz

def get_include():
  """Returns the directory containing the C/C++ API include directives"""

  return __import__('pkg_resources').resource_filename(__name__, 'include')

__version__ = __import__('pkg_resources').require('blitz.array')[0].version

__all__ = ['array', 'as_blitz']
