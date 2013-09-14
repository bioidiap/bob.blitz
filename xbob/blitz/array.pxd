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
  ctypedef int _3 "3"
  ctypedef int _4 "4"
  ctypedef int _5 "5"

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
  object shallow_ndarray_u8d1 "shallow_ndarray<uint8_t,1>" (Array[uint8_t,_1], object)
  object shallow_ndarray_u8d2 "shallow_ndarray<uint8_t,2>" (Array[uint8_t,_2], object)
  object shallow_ndarray_u8d3 "shallow_ndarray<uint8_t,3>" (Array[uint8_t,_3], object)
  object shallow_ndarray_u8d4 "shallow_ndarray<uint8_t,4>" (Array[uint8_t,_4], object)
  object shallow_ndarray_u8d4 "shallow_ndarray<uint8_t,5>" (Array[uint8_t,_5], object)
