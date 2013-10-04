#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Sun 15 Sep 20:29:36 2013 

"""Conversion utilities between array types and ranges
"""

cimport array
import numpy

cdef extern from "<convert.h>":

  cdef cppclass Converter[T,U,N]:

    Converter() nogil

    Array[T,N] convert_full(Array[U,N] arr,
      T dst_min, T dst_max, U src_min, U src_max) nogil except +
    Array[T,N] convert_from(Array[U,N] arr, U src_min, U src_max) nogil except +
    Array[T,N] convert_to(Array[U,N] arr, T dst_min, T dst_max) nogil except +
    Array[T,N] convert_simple(Array[U,N] arr) nogil except +

def __convert_to_uint8_from_bool__(arr, dest_range=None, source_range=None):
  """(Internal) Converts from uint8 to bool"""

  if arr.ndim == 1:
    cdef Converter[
    pass
  elif arr.ndim == 2:
    pass
  elif arr.ndim == 3:
    pass
  elif arr.ndim == 4:
    pass
  else:
    raise TypeError("conversion does not support %d dimensions" % arr.ndim)

def __convert_to_uint8__(arr, dest_range=None, source_range=None):
  """(Internal) Converts to uint8 arrays, specifically"""

  if arr.dtype == numpy.bool_:
    return __convert_to_uint8_from_bool__(arr, dest_range, source_range)
  elif arr.dtype == numpy.int8:
    return __convert_to_uint8_from_int8__(arr, dest_range, source_range)
  elif arr.dtype == numpy.int16:
    return __convert_to_uint8_from_int16__(arr, dest_range, source_range)
  elif arr.dtype == numpy.int32:
    return __convert_to_uint8_from_int32__(arr, dest_range, source_range)
  elif arr.dtype == numpy.int64:
    return __convert_to_uint8_from_int64__(arr, dest_range, source_range)
  elif arr.dtype == numpy.uint8:
    return __convert_to_uint8_from_uint8__(arr, dest_range, source_range)
  elif arr.dtype == numpy.uint16:
    return __convert_to_uint8_from_uint16__(arr, dest_range, source_range)
  elif arr.dtype == numpy.uint32:
    return __convert_to_uint8_from_uint32__(arr, dest_range, source_range)
  elif arr.dtype == numpy.uint64:
    return __convert_to_uint8_from_uint64__(arr, dest_range, source_range)
  elif arr.dtype == numpy.float32:
    return __convert_to_uint8_from_float32__(arr, dest_range, source_range)
  elif arr.dtype == numpy.float64:
    return __convert_to_uint8_from_float64__(arr, dest_range, source_range)
  else:
    raise TypeError("conversion from `%s' is not supported" % arr.dtype)

def convert(array, dtype, dest_range=None, source_range=None):
  """Converts and array to a different type, possibly with re-scaling.

  Function which allows to convert/rescale a array of a given type into another
  array of a possibly different type with re-scaling. Typically, this can be
  used to rescale a 16 bit precision grayscale image (2D array) into an 8 bit
  precision grayscale image.
  
  Parameters:
  
    array
      (array) Input array
      
    dtype
      (string) Controls the output element type for the returned array
      
    dest_range
      (tuple) Determines the range to be deployed at the returned array
      
    source_range
      (tuple) Determines the input range that will be used for the scaling
      
  Returns a new py:class:`numpy.ndarray` with the same shape as this one, but
  re-scaled and with its element type as indicated by the user.
  """

  arr = numpy.array(array)
  dt = numpy.dtype(dtype)
  
  if dt == numpy.uint8:
    return __convert_to_uint8__(arr, source_range, dest_range)
  elif dt == numpy.uint16:
    return __convert_to_uint16__(arr, source_range, dest_range)
  elif dt == numpy.float64:
    return __convert_to_float64__(arr, source_range, dest_range)
  else:
    raise TypeError("conversion to `%s' is not supported" % dt)
