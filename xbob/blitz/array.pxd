#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 29 Aug 2013 16:14:11 CEST

"""Cython definitions for blitz::Array<>
"""

cimport libc.stdint

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
  object shallow_ndarray_u8d1 "shallow_ndarray<uint8_t,1>" (Array[libc.stdint.uint8_t,_1], object) except +
  object shallow_ndarray_u8d2 "shallow_ndarray<uint8_t,2>" (Array[libc.stdint.uint8_t,_2], object) except +
  object shallow_ndarray_u8d3 "shallow_ndarray<uint8_t,3>" (Array[libc.stdint.uint8_t,_3], object) except +
  object shallow_ndarray_u8d4 "shallow_ndarray<uint8_t,4>" (Array[libc.stdint.uint8_t,_4], object) except +
  object shallow_ndarray_u8d5 "shallow_ndarray<uint8_t,5>" (Array[libc.stdint.uint8_t,_5], object) except +
  object shallow_ndarray_u16d1 "shallow_ndarray<uint16_t,1>" (Array[libc.stdint.uint16_t,_1], object) except +
  object shallow_ndarray_u16d2 "shallow_ndarray<uint16_t,2>" (Array[libc.stdint.uint16_t,_2], object) except +
  object shallow_ndarray_u16d3 "shallow_ndarray<uint16_t,3>" (Array[libc.stdint.uint16_t,_3], object) except +
  object shallow_ndarray_u16d5 "shallow_ndarray<uint16_t,4>" (Array[libc.stdint.uint16_t,_4], object) except +
  object shallow_ndarray_u16d4 "shallow_ndarray<uint16_t,5>" (Array[libc.stdint.uint16_t,_5], object) except +
  object shallow_ndarray_u32d1 "shallow_ndarray<uint32_t,1>" (Array[libc.stdint.uint32_t,_1], object) except +
  object shallow_ndarray_u32d2 "shallow_ndarray<uint32_t,2>" (Array[libc.stdint.uint32_t,_2], object) except +
  object shallow_ndarray_u32d3 "shallow_ndarray<uint32_t,3>" (Array[libc.stdint.uint32_t,_3], object) except +
  object shallow_ndarray_u32d4 "shallow_ndarray<uint32_t,4>" (Array[libc.stdint.uint32_t,_4], object) except +
  object shallow_ndarray_u64d5 "shallow_ndarray<uint32_t,5>" (Array[libc.stdint.uint32_t,_5], object) except +
  object shallow_ndarray_u64d1 "shallow_ndarray<uint64_t,1>" (Array[libc.stdint.uint64_t,_1], object) except +
  object shallow_ndarray_u64d2 "shallow_ndarray<uint64_t,2>" (Array[libc.stdint.uint64_t,_2], object) except +
  object shallow_ndarray_u64d3 "shallow_ndarray<uint64_t,3>" (Array[libc.stdint.uint64_t,_3], object) except +
  object shallow_ndarray_u64d4 "shallow_ndarray<uint64_t,4>" (Array[libc.stdint.uint64_t,_4], object) except +
  object shallow_ndarray_u64d5 "shallow_ndarray<uint64_t,5>" (Array[libc.stdint.uint64_t,_5], object) except +
  object shallow_ndarray_s8d1 "shallow_ndarray<int8_t,1>" (Array[libc.stdint.int8_t,_1], object) except +
  object shallow_ndarray_s8d2 "shallow_ndarray<int8_t,2>" (Array[libc.stdint.int8_t,_2], object) except +
  object shallow_ndarray_s8d3 "shallow_ndarray<int8_t,3>" (Array[libc.stdint.int8_t,_3], object) except +
  object shallow_ndarray_s8d4 "shallow_ndarray<int8_t,4>" (Array[libc.stdint.int8_t,_4], object) except +
  object shallow_ndarray_s8d5 "shallow_ndarray<int8_t,5>" (Array[libc.stdint.int8_t,_5], object) except +
  object shallow_ndarray_s16d1 "shallow_ndarray<int16_t,1>" (Array[libc.stdint.int16_t,_1], object) except +
  object shallow_ndarray_s16d2 "shallow_ndarray<int16_t,2>" (Array[libc.stdint.int16_t,_2], object) except +
  object shallow_ndarray_s16d3 "shallow_ndarray<int16_t,3>" (Array[libc.stdint.int16_t,_3], object) except +
  object shallow_ndarray_s16d5 "shallow_ndarray<int16_t,4>" (Array[libc.stdint.int16_t,_4], object) except +
  object shallow_ndarray_s16d4 "shallow_ndarray<int16_t,5>" (Array[libc.stdint.int16_t,_5], object) except +
  object shallow_ndarray_s32d1 "shallow_ndarray<int32_t,1>" (Array[libc.stdint.int32_t,_1], object) except +
  object shallow_ndarray_s32d2 "shallow_ndarray<int32_t,2>" (Array[libc.stdint.int32_t,_2], object) except +
  object shallow_ndarray_s32d3 "shallow_ndarray<int32_t,3>" (Array[libc.stdint.int32_t,_3], object) except +
  object shallow_ndarray_s32d4 "shallow_ndarray<int32_t,4>" (Array[libc.stdint.int32_t,_4], object) except +
  object shallow_ndarray_s64d5 "shallow_ndarray<int32_t,5>" (Array[libc.stdint.int32_t,_5], object) except +
  object shallow_ndarray_s64d1 "shallow_ndarray<int64_t,1>" (Array[libc.stdint.int64_t,_1], object) except +
  object shallow_ndarray_s64d2 "shallow_ndarray<int64_t,2>" (Array[libc.stdint.int64_t,_2], object) except +
  object shallow_ndarray_s64d3 "shallow_ndarray<int64_t,3>" (Array[libc.stdint.int64_t,_3], object) except +
  object shallow_ndarray_s64d4 "shallow_ndarray<int64_t,4>" (Array[libc.stdint.int64_t,_4], object) except +
  object shallow_ndarray_s64d5 "shallow_ndarray<int64_t,5>" (Array[libc.stdint.int64_t,_5], object) except +
  object shallow_ndarray_f32d1 "shallow_ndarray<float,1>" (Array[float,_1], object) except +
  object shallow_ndarray_f32d2 "shallow_ndarray<float,2>" (Array[float,_2], object) except +
  object shallow_ndarray_f32d3 "shallow_ndarray<float,3>" (Array[float,_3], object) except +
  object shallow_ndarray_f32d4 "shallow_ndarray<float,4>" (Array[float,_4], object) except +
  object shallow_ndarray_f32d5 "shallow_ndarray<float,5>" (Array[float,_5], object) except +
  object shallow_ndarray_f64d1 "shallow_ndarray<double,1>" (Array[double,_1], object) except +
  object shallow_ndarray_f64d2 "shallow_ndarray<double,2>" (Array[double,_2], object) except +
  object shallow_ndarray_f64d3 "shallow_ndarray<double,3>" (Array[double,_3], object) except +
  object shallow_ndarray_f64d4 "shallow_ndarray<double,4>" (Array[double,_4], object) except +
  object shallow_ndarray_f64d5 "shallow_ndarray<double,5>" (Array[double,_5], object) except +
