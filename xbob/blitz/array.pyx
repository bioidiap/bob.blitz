#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 29 Aug 2013 16:17:00 CEST

"""Cython bindings for blitz::Array<>
"""

cimport numpy
cimport array
from cython.operator cimport dereference as deref

import numpy

cdef class bz1u8:

  # holds a C++ instance of a blitz::Array<>
  cdef array.Array[numpy.uint8_t,array._1]* thisptr

  def __cinit__(self, int size):
    self.thisptr = new array.Array[numpy.uint8_t,array._1](size)

  def __dealloc__(self):
    del self.thisptr

  def shallow_ndarray(self):
    return shallow_ndarray_u8d1(deref(self.thisptr))
