#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 20 Sep 14:45:01 2013 

"""Tests for bob::python glue methods
"""

import numpy
import nose
from . import __test_array__ as array

def test_u8d1_from_scratch():

  bz = array.vector(10, dtype='uint8')
  nose.tools.eq_(len(bz), 10)

  bz = array.vector(20000, dtype='uint8')
  nose.tools.eq_(len(bz), 20000)

@nose.tools.raises(OverflowError)
def test_u8d1_negative_size():

  bz = array.vector(-1, dtype='uint8')

def test_u8d1_zero_size():

  bz = array.vector(0, dtype='uint8')
  nose.tools.eq_(len(bz), 0)

def test_u8d1_assign_and_read():

  bz = array.vector(2, dtype='uint8')

  # assign a value to each position, check back
  bz[0] = 3
  nose.tools.eq_(bz[0], 3)
  bz[1] = 12
  nose.tools.eq_(bz[1], 12)

def test_u32d1_assign_and_read():

  bz = array.vector(2, dtype='uint32')

  # assign a value to each position, check back
  bz[0] = 3
  nose.tools.eq_(bz[0], 3)
  bz[1] = 0
  nose.tools.eq_(bz[1], 0)

@nose.tools.raises(IndexError)
def test_u8d1_protect_segfault_high():

  bz = array.vector(2, dtype='uint8')
  bz[3] = 4

@nose.tools.raises(IndexError)
def test_u8d1_protect_segfault_low():

  bz = array.vector(2, dtype='uint8')
  bz[-1] = 4

@nose.tools.raises(OverflowError)
def test_u8d1_overflow_detection():

  bz = array.vector(1, dtype='uint8')
  bz[0] = 256

@nose.tools.raises(OverflowError)
def test_u16d1_overflow_detection():

  bz = array.vector(1, dtype='uint16')
  bz[0] = 2**16 + 1

@nose.tools.raises(OverflowError)
def test_s16d1_overflow_detection():

  bz = array.vector(1, dtype='int16')
  bz[0] = 2**15 + 1

@nose.tools.raises(OverflowError)
def test_u32d1_overflow_detection():

  bz = array.vector(1, dtype='uint32')
  bz[0] = 2**32 + 1

@nose.tools.raises(OverflowError)
def test_u8d1_underflow_detection():

  bz = array.vector(1, dtype='uint8')
  bz[0] = -1

@nose.tools.raises(OverflowError)
def test_u64d1_underflow_detection():

  bz = array.vector(1, dtype='uint64')
  bz[0] = -1

def test_can_extract_uint8():

  assert array.extract_uint8(22) == 22
  assert array.extract_uint8(numpy.float64(3.14)) == 3
  assert array.extract_uint8(numpy.uint16(255)) == 255

def test_can_extract_uint16():

  assert array.extract_uint16(22) == 22
  assert array.extract_uint16(numpy.float64(3.14)) == 3
  assert array.extract_uint16(numpy.uint32(255)) == 255

def test_can_extract_uint32():

  assert array.extract_uint32(22) == 22
  assert array.extract_uint32(numpy.float64(3.14)) == 3
  assert array.extract_uint32(numpy.uint64(255)) == 255

def test_can_extract_float64():

  assert array.extract_float64(22) == 22.0
  assert array.extract_float64(numpy.uint64(255))

def test_can_extract_complex128():

  assert array.extract_complex128(22) == complex(22, 0)
  assert array.extract_complex128(numpy.complex128(complex(1, 2))) == complex(1, 2)

def test_u8d1_ndarray_from_blitz():

  bz = array.vector(2, dtype='uint8')
  bz[0] = 32
  bz[1] = 10
  nd = bz.ndarray()
  nose.tools.eq_(nd.shape, (len(bz),))
  nose.tools.eq_(bz[0], nd[0])
  nose.tools.eq_(bz[1], nd[1])
  assert nd.flags.owndata
  assert nd.flags.behaved
  assert nd.flags.c_contiguous
  assert nd.flags.writeable
  nose.tools.eq_(nd.dtype, numpy.dtype('uint8'))
  del bz
  nose.tools.eq_(nd[0], 32)
  nose.tools.eq_(nd[1], 10)

def test_u64d1_ndarray_from_blitz():

  bz = array.vector(2, dtype='uint64')
  bz[0] = 2**33
  bz[1] = 2**64 - 1
  nd = bz.ndarray()
  nose.tools.eq_(nd.shape, (len(bz),))
  nose.tools.eq_(bz[0], nd[0])
  nose.tools.eq_(bz[1], nd[1])
  assert nd.flags.owndata
  assert nd.flags.behaved
  assert nd.flags.c_contiguous
  assert nd.flags.writeable
  nose.tools.eq_(nd.dtype, numpy.dtype('uint64'))
  del bz
  nose.tools.eq_(nd[0], 2**33)
  nose.tools.eq_(nd[1], 2**64-1)

def test_u32d1_ndarray_from_blitz():

  bz = array.vector(2, dtype='uint32')
  bz[0] = 2**32-1
  bz[1] = 0
  nd = bz.ndarray()
  nose.tools.eq_(nd.shape, (len(bz),))
  nose.tools.eq_(bz[0], nd[0])
  nose.tools.eq_(bz[1], nd[1])
  del bz
  assert nd.flags.owndata
  assert nd.flags.behaved
  assert nd.flags.c_contiguous
  assert nd.flags.writeable
  nose.tools.eq_(nd.dtype, numpy.dtype('uint32'))
  nose.tools.eq_(nd[0], 2**32-1)
  nose.tools.eq_(nd[1], 0)

def test_u32d1_ndarray_from_blitz_2():

  bz = array.vector(2, dtype='uint32')
  bz[0] = 2**32-1
  bz[1] = 0
