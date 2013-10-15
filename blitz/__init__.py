from ._array import array, as_blitz
del _array
__version__ = __import__('pkg_resources').require('blitz.array')[0].version
__all__ = ['array', 'as_blitz']
