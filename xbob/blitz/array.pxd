#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 29 Aug 2013 16:14:11 CEST

"""Cython definitions for blitz::Array<>
"""

from libc.stdint cimport uint8_t

cdef extern from *:

  ctypedef int _1 "1"
  ctypedef int _2 "2"

cdef extern from "<blitz/array.h>" namespace "blitz":

  cdef cppclass Array[T,I]:

    Array() nogil except +
    Array(Array&) nogil except +
    Array(int) nogil except +
    Array(int,int) nogil except +

    # Use these for item readout (__getitem__)
    T operator()(int) nogil except +
    T operator()(int,int) nogil except +

    # Use these for item assignment (__setitem__)
    T& operator[](int) nogil except +
    T& operator[](int,int) nogil except +

    T* data() nogil

    int size() nogil
    int extent(int) nogil except +
    int stride(int) nogil except +

cdef extern from "<convert.h>":

  void import_ndarray()
  object shallow_ndarray_u8d1 (Array[uint8_t,_1], object)
