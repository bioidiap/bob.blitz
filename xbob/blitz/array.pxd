#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 29 Aug 2013 16:14:11 CEST

"""Cython definitions for blitz::Array<>
"""

cimport libcpp

cdef extern from *:

  ctypedef int _1 "1"
  ctypedef int _2 "2"
  ctypedef int _3 "3"
  ctypedef int _4 "4"

cdef extern from "<blitz/array.h>" namespace "blitz":

  cdef cppclass Array[T,I]:

    Array() nogil except +
    Array(Array&) nogil except +
    Array(int) nogil except +
    Array(int,int) nogil except +
    Array(int,int,int) nogil except +
    Array(int,int,int,int) nogil except +

    # Use these for item readout (__getitem__)
    T operator()(int) nogil except +
    T operator()(int,int) nogil except +
    T operator()(int,int,int) nogil except +
    T operator()(int,int,int,int) nogil except +

    # Use these for item assignment (__setitem__)
    T& operator[](int) nogil except +
    T& operator[](int,int) nogil except +
    T& operator[](int,int,int) nogil except +
    T& operator[](int,int,int,int) nogil except +

    T* data() nogil

    int size() nogil
    int extent(int) nogil except +
    int stride(int) nogil except +

cdef extern from "<bob/py.h>" namespace "bob::python":

  void bob_import_array()

  cdef cppclass ShallowBlitzArray[T,I]:

    ShallowBlitzArray() nogil

    Array[T,I] call(object, libcpp.bool) except +

  cdef cppclass ReadonlyBlitzArray[T,I]:

    ReadonlyBlitzArray() nogil

    Array[T,I] call(object, libcpp.bool) except +

  cdef cppclass NumpyArrayCopy[T,I]:
    
    NumpyArrayCopy() nogil

    object call(Array[T,I]&) except +

  cdef cppclass CtypeToNum[T]:

    CtypeToNum() nogil

    int call()
  
  cdef cppclass Extract[T]:

    Extract() nogil

    T call(object)
