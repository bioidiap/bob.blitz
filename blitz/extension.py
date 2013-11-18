#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 18 Nov 21:38:19 2013 

"""Extension building for using this package
"""

import numpy
from pkg_resources import resource_filename
from xbob.extension import Extension as XbobExtension
from distutils.version import StrictVersion

class Extension(XbobExtension):
  """Extension building with pkg-config packages and blitz.array.

  See the documentation for :py:class:`distutils.extension.Extension` for more
  details on input parameters.
  """

  def __init__(self, *args, **kwargs):
    """Initialize the extension with parameters.

    This extension adds ``blitz>=0.10`` as a requirement for extensions derived
    from this class.

    See the help for :py:class:`xbob.extension.Extension` for more details on
    options.
    """

    require = 'blitz>=0.10'

    kwargs.setdefault('packages', []).append(require)

    numpy_include = ['-isystem', numpy.get_include()]
    kwargs.setdefault('extra_compile_args', []).extend(numpy_include)

    blitz_array_include = resource_filename(__name__, 'include')
    kwargs.setdefault('include_dirs', []).append(blitz_array_include)

    macros = [
          ("PY_ARRAY_UNIQUE_SYMBOL", "BLITZ_ARRAY_NUMPY_C_API"),
          ("NO_IMPORT_ARRAY", "1"),
          ]

    if StrictVersion(numpy.__version__) >= StrictVersion('1.7'):
      macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))

    kwargs.setdefault('define_macros', []).extend(macros)

    # Run the constructor for the base class
    XbobExtension.__init__(self, *args, **kwargs)
