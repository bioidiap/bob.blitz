#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 20 Sep 14:45:01 2013

"""Blitz++ Array bindings for Python"""

from ._library import array, as_blitz
from . import version
from .version import module as __version__
from .version import api as __api_version__

def get_include():
  """Returns the directory containing the C/C++ API include directives"""

  return __import__('pkg_resources').resource_filename(__name__, 'include')

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
