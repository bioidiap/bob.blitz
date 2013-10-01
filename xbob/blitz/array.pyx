#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 29 Aug 2013 16:17:00 CEST

"""Cython bindings for blitz::Array<>
"""

cimport libcpp
cimport numpy
cimport array
from cython.operator cimport dereference as deref

import numpy

# imports the array interface once
array.bob_import_array()

cdef class vector:

  # holds a C++ instance of a blitz::Array<>
  cdef numpy.dtype dtype
  cdef array.Array[libcpp.bool,array._1]* boolptr
  cdef array.Array[numpy.uint8_t,array._1]* u8ptr
  cdef array.Array[numpy.uint16_t,array._1]* u16ptr
  cdef array.Array[numpy.uint32_t,array._1]* u32ptr
  cdef array.Array[numpy.uint64_t,array._1]* u64ptr
  cdef array.Array[numpy.int8_t,array._1]* s8ptr
  cdef array.Array[numpy.int16_t,array._1]* s16ptr
  cdef array.Array[numpy.int32_t,array._1]* s32ptr
  cdef array.Array[numpy.int64_t,array._1]* s64ptr
  cdef array.Array[numpy.float32_t,array._1]* f32ptr
  cdef array.Array[numpy.float64_t,array._1]* f64ptr
  #cdef array.Array[numpy.float128_t,array._1]* f128ptr
  cdef array.Array[numpy.complex64_t,array._1]* c64ptr
  cdef array.Array[numpy.complex128_t,array._1]* c128ptr
  #cdef array.Array[numpy.complex256_t,array._1]* c256ptr

  def __cinit__(self, object shape, object dtype):

    self.dtype = numpy.dtype(dtype)
    cdef numpy.uint64_t size = shape

    if self.dtype == numpy.bool_:
      self.boolptr = new array.Array[libcpp.bool,array._1](size)
    elif self.dtype == numpy.uint8:
      self.u8ptr = new array.Array[numpy.uint8_t,array._1](size)
    elif self.dtype == numpy.uint16:
      self.u16ptr = new array.Array[numpy.uint16_t,array._1](size)
    elif self.dtype == numpy.uint32:
      self.u32ptr = new array.Array[numpy.uint32_t,array._1](size)
    elif self.dtype == numpy.uint64:
      self.u64ptr = new array.Array[numpy.uint64_t,array._1](size)
    elif self.dtype == numpy.int8:
      self.s8ptr = new array.Array[numpy.int8_t,array._1](size)
    elif self.dtype == numpy.int16:
      self.s16ptr = new array.Array[numpy.int16_t,array._1](size)
    elif self.dtype == numpy.int32:
      self.s32ptr = new array.Array[numpy.int32_t,array._1](size)
    elif self.dtype == numpy.int64:
      self.s64ptr = new array.Array[numpy.int64_t,array._1](size)
    elif self.dtype == numpy.float32:
      self.f32ptr = new array.Array[numpy.float32_t,array._1](size)
    elif self.dtype == numpy.float64:
      self.f64ptr = new array.Array[numpy.float64_t,array._1](size)
    #elif self.dtype == numpy.float128:
    #  self.f128ptr = new array.Array[numpy.float128_t,array._1](size)
    elif self.dtype == numpy.complex64:
      self.c64ptr = new array.Array[numpy.complex64_t,array._1](size)
    elif self.dtype == numpy.complex128:
      self.c128ptr = new array.Array[numpy.complex128_t,array._1](size)
    #elif self.dtype == numpy.complex256:
    #  self.c256ptr = new array.Array[numpy.complex256_t,array._1](size)
    else:
      raise TypeError("No support for dtype `%s'" % self.dtype)

  def __dealloc__(self):
    if self.dtype == numpy.bool_:
      del self.boolptr
    elif self.dtype == numpy.uint8:
      del self.u8ptr
    elif self.dtype == numpy.uint16:
      del self.u16ptr
    elif self.dtype == numpy.uint32:
      del self.u32ptr
    elif self.dtype == numpy.uint64:
      del self.u64ptr
    elif self.dtype == numpy.int8:
      del self.s8ptr
    elif self.dtype == numpy.int16:
      del self.s16ptr
    elif self.dtype == numpy.int32:
      del self.s32ptr
    elif self.dtype == numpy.int64:
      del self.s64ptr
    elif self.dtype == numpy.float32:
      del self.f32ptr
    elif self.dtype == numpy.float64:
      del self.f64ptr
    #elif self.dtype == numpy.float128:
    #  del self.f128ptr
    elif self.dtype == numpy.complex64:
      del self.c64ptr
    elif self.dtype == numpy.complex128:
      del self.c128ptr
    #elif self.dtype == numpy.complex256:
    #  del self.c256ptr
    else:
      raise TypeError("No support for dtype `%s'" % self.dtype)

  def __len__(self):
    if self.dtype == numpy.bool_:
      return self.boolptr.size()
    elif self.dtype == numpy.uint8:
      return self.u8ptr.size()
    elif self.dtype == numpy.uint16:
      return self.u16ptr.size()
    elif self.dtype == numpy.uint32:
      return self.u32ptr.size()
    elif self.dtype == numpy.uint64:
      return self.u64ptr.size()
    elif self.dtype == numpy.int8:
      return self.s8ptr.size()
    elif self.dtype == numpy.int16:
      return self.s16ptr.size()
    elif self.dtype == numpy.int32:
      return self.s32ptr.size()
    elif self.dtype == numpy.int64:
      return self.s64ptr.size()
    elif self.dtype == numpy.float32:
      return self.f32ptr.size()
    elif self.dtype == numpy.float64:
      return self.f64ptr.size()
    #elif self.dtype == numpy.float128:
    #  return self.f128ptr.size()
    elif self.dtype == numpy.complex64:
      return self.c64ptr.size()
    elif self.dtype == numpy.complex128:
      return self.c128ptr.size()
    #elif self.dtype == numpy.complex256:
    #  return self.c256ptr.size()
    else:
      raise TypeError("No support for dtype `%s'" % self.dtype)

  def __getitem__(self, int key):
    if key < 0 or key >= len(self):
      raise IndexError("index %d is beyond scope (%d)" % (key, len(self)))
    if self.dtype == numpy.bool_:
      return deref(self.boolptr)(key)
    elif self.dtype == numpy.uint8:
      return deref(self.u8ptr)(key)
    elif self.dtype == numpy.uint16:
      return deref(self.u16ptr)(key)
    elif self.dtype == numpy.uint32:
      return deref(self.u32ptr)(key)
    elif self.dtype == numpy.uint64:
      return deref(self.u64ptr)(key)
    elif self.dtype == numpy.int8:
      return deref(self.s8ptr)(key)
    elif self.dtype == numpy.int16:
      return deref(self.s16ptr)(key)
    elif self.dtype == numpy.int32:
      return deref(self.s32ptr)(key)
    elif self.dtype == numpy.int64:
      return deref(self.s64ptr)(key)
    elif self.dtype == numpy.float32:
      return deref(self.f32ptr)(key)
    elif self.dtype == numpy.float64:
      return deref(self.f64ptr)(key)
    #elif self.dtype == numpy.float128:
    #  return deref(self.f128ptr)(key)
    elif self.dtype == numpy.complex64:
      return deref(self.c64ptr)(key)
    elif self.dtype == numpy.complex128:
      return deref(self.c128ptr)(key)
    #elif self.dtype == numpy.complex256:
    #  return deref(self.c256ptr)(key)
    else:
      raise TypeError("No support for dtype `%s'" % self.dtype)

  def __setitem__(self, int key, object value):
    if key < 0 or key >= len(self):
      raise IndexError("index %d is beyond scope (%d)" % (key, len(self)))
    if self.dtype == numpy.bool_:
      deref(self.boolptr)[key] = value
    elif self.dtype == numpy.uint8:
      deref(self.u8ptr)[key] = value
    elif self.dtype == numpy.uint16:
      deref(self.u16ptr)[key] = value
    elif self.dtype == numpy.uint32:
      deref(self.u32ptr)[key] = value
    elif self.dtype == numpy.uint64:
      deref(self.u64ptr)[key] = value
    elif self.dtype == numpy.int8:
      deref(self.s8ptr)[key] = value
    elif self.dtype == numpy.int16:
      deref(self.s16ptr)[key] = value
    elif self.dtype == numpy.int32:
      deref(self.s32ptr)[key] = value
    elif self.dtype == numpy.int64:
      deref(self.s64ptr)[key] = value
    elif self.dtype == numpy.float32:
      deref(self.f32ptr)[key] = value
    elif self.dtype == numpy.float64:
      deref(self.f64ptr)[key] = value
    #elif self.dtype == numpy.float128:
    #  deref(self.f128ptr)[key] = value
    #elif self.dtype == numpy.complex64:
    #  deref(self.c64ptr)[key] = value
    #elif self.dtype == numpy.complex128:
    #  deref(self.c128ptr)[key] = value
    #elif self.dtype == numpy.complex256:
    #  deref(self.c256ptr)[key] = value
    else:
      raise TypeError("No support for dtype `%s'" % self.dtype)

  def ndarray(self):
    cdef array.NumpyArrayCopy[libcpp.bool,array._1] boolcopier
    cdef array.NumpyArrayCopy[numpy.uint8_t,array._1] u8copier
    cdef array.NumpyArrayCopy[numpy.uint16_t,array._1] u16copier
    cdef array.NumpyArrayCopy[numpy.uint32_t,array._1] u32copier
    cdef array.NumpyArrayCopy[numpy.uint64_t,array._1] u64copier
    cdef array.NumpyArrayCopy[numpy.int8_t,array._1] s8copier
    cdef array.NumpyArrayCopy[numpy.int16_t,array._1] s16copier
    cdef array.NumpyArrayCopy[numpy.int32_t,array._1] s32copier
    cdef array.NumpyArrayCopy[numpy.int64_t,array._1] s64copier
    cdef array.NumpyArrayCopy[numpy.float32_t,array._1] f32copier
    cdef array.NumpyArrayCopy[numpy.float64_t,array._1] f64copier
    #cdef array.NumpyArrayCopy[numpy.float128_t,array._1] f128copier
    cdef array.NumpyArrayCopy[numpy.complex64_t,array._1] c64copier
    cdef array.NumpyArrayCopy[numpy.complex128_t,array._1] c128copier
    #cdef array.NumpyArrayCopy[numpy.complex256_t,array._1] c256copier

    if self.dtype == numpy.bool_:
      return boolcopier.call(deref(self.boolptr))
    elif self.dtype == numpy.uint8:
      return u8copier.call(deref(self.u8ptr))
    elif self.dtype == numpy.uint16:
      return u16copier.call(deref(self.u16ptr))
    elif self.dtype == numpy.uint32:
      return u32copier.call(deref(self.u32ptr))
    elif self.dtype == numpy.uint64:
      return u64copier.call(deref(self.u64ptr))
    elif self.dtype == numpy.int8:
      return s8copier.call(deref(self.s8ptr))
    elif self.dtype == numpy.int16:
      return s16copier.call(deref(self.s16ptr))
    elif self.dtype == numpy.int32:
      return s32copier.call(deref(self.s32ptr))
    elif self.dtype == numpy.int64:
      return s64copier.call(deref(self.s64ptr))
    elif self.dtype == numpy.float32:
      return f32copier.call(deref(self.f32ptr))
    elif self.dtype == numpy.float64:
      return f64copier.call(deref(self.f64ptr))
    #elif self.dtype == numpy.float128:
    #  return f128copier.call(deref(self.f128ptr))
    elif self.dtype == numpy.complex64:
      return c64copier.call(deref(self.c64ptr))
    elif self.dtype == numpy.complex128:
      return c128copier.call(deref(self.c128ptr))
    #elif self.dtype == numpy.complex256:
    #  return c256copier.call(deref(self.c256ptr))
    else:
      raise TypeError("No support for dtype `%s'" % self.dtype)

cpdef libcpp.bool extract_bool (object o) except? 0:
  """Extracts a boolean out of the given object"""

  cdef array.Extract[libcpp.bool] x
  retval = x.call(o)

cpdef numpy.uint8_t extract_uint8 (object o) except? 0:
  """Extracts an unsigned integer of 8-bits out of the given object"""

  cdef array.Extract[numpy.uint8_t] x
  return x.call(o)

cpdef numpy.uint16_t extract_uint16 (object o) except? 0:
  """Extracts an unsigned integer of 16-bits out of the given object"""

  cdef array.Extract[numpy.uint16_t] x
  return x.call(o)

cpdef numpy.uint32_t extract_uint32 (object o) except? 0:
  """Extracts an unsigned integer of 32-bits out of the given object"""

  cdef array.Extract[numpy.uint32_t] x
  return x.call(o)

cpdef numpy.uint64_t extract_uint64 (object o) except? 0:
  """Extracts an unsigned integer of 64-bits out of the given object"""

  cdef array.Extract[numpy.uint64_t] x
  return x.call(o)

cpdef numpy.int8_t extract_int8 (object o) except? 0:
  """Extracts an unsigned integer of 8-bits out of the given object"""

  cdef array.Extract[numpy.int8_t] x
  return x.call(o)

cpdef numpy.int16_t extract_int16 (object o) except? 0:
  """Extracts an unsigned integer of 16-bits out of the given object"""

  cdef array.Extract[numpy.int16_t] x
  return x.call(o)

cpdef numpy.int32_t extract_int32 (object o) except? 0:
  """Extracts an unsigned integer of 32-bits out of the given object"""

  cdef array.Extract[numpy.int32_t] x
  return x.call(o)

cpdef numpy.int64_t extract_int64 (object o) except? 0:
  """Extracts an unsigned integer of 64-bits out of the given object"""

  cdef array.Extract[numpy.int64_t] x
  return x.call(o)

cpdef numpy.float32_t extract_float32 (object o) except? 0.0:
  """Extracts a 32-bits float out of the given object"""

  cdef array.Extract[numpy.float32_t] x
  return x.call(o)

cpdef numpy.float64_t extract_float64 (object o) except? 0.0:
  """Extracts a 64-bits float out of the given object"""

  cdef array.Extract[numpy.float64_t] x
  return x.call(o)

'''
cpdef numpy.float128_t extract_float128 (object o) except? 0.0:
  """Extracts a 128-bits float out of the given object"""

  cdef array.Extract[numpy.float128_t] x
  return x.call(o)
'''

cpdef numpy.complex64_t extract_complex64 (object o) except *:
  """Extracts a 64-bits complex numbers out of the given object"""

  cdef array.Extract[numpy.complex64_t] x
  return x.call(o)

cpdef numpy.complex128_t extract_complex128 (object o) except *:
  """Extracts a 128-bit complex numbers out of the given object"""

  cdef array.Extract[numpy.complex128_t] x
  return x.call(o)

'''
cpdef numpy.complex256_t extract_complex256 (object o) except *:
  """Extracts a 256-bit complex numbers out of the given object"""

  cdef array.Extract[numpy.complex256_t] x
  return x.call(o)
'''
