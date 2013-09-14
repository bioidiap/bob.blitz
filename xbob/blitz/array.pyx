#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 29 Aug 2013 16:17:00 CEST

"""Cython bindings for blitz::Array<>
"""

cimport libc.stdint
cimport array
from cython.operator cimport dereference as deref

# imports the array interface once
array.import_ndarray()

cdef class u8d1:

  # holds a C++ instance of a blitz::Array<>
  cdef array.Array[libc.stdint.uint8_t,array._1]* thisptr

  def __cinit__(self, libc.stdint.uint64_t size):
    self.thisptr = new array.Array[libc.stdint.uint8_t,array._1](size)

  def __dealloc__(self):
    del self.thisptr

  def __len__(self):
    return self.thisptr.size()

  def __getitem__(self, int key):
    if key < 0 or key > len(self):
      raise IndexError("index %d is beyond scope (%d)" % (key, len(self)))
    return deref(self.thisptr)(key)

  def __setitem__(self, int key, libc.stdint.uint8_t value):
    if key < 0 or key > len(self):
      raise IndexError("index %d is beyond scope (%d)" % (key, len(self)))
    deref(self.thisptr)[key] = value

  def ndarray(self):
    return shallow_ndarray_u8d1(deref(self.thisptr), self)
