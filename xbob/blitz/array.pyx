#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 29 Aug 2013 16:17:00 CEST

"""Cython bindings for blitz::Array<>
"""

cimport array
cimport numpy
cimport libc.stdint
from cython.operator cimport dereference as deref

cdef class bz1u8:

  # holds a C++ instance of a blitz::Array<>
  cdef array.Array[libc.stdint.uint8_t,array._1]* thisptr

  def __cinit__(self, int size):
    self.thisptr = new array.Array[libc.stdint.uint8_t,array._1](size)

  def __dealloc__(self):
    del self.thisptr

  def __getitem__(self, int key):
    cdef libc.stdint.uint8_t retval = deref(self.thisptr)(key)
    return retval
