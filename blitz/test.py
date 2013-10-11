#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 20 Sep 14:45:01 2013

"""Tests for blitz.array glue methods
"""

import numpy
import nose
from . import array as bzarray

def test_array_from_scratch():

  bz = bzarray(10, dtype='uint8')
  nose.tools.eq_(bz.shape, (10,))
  nose.tools.eq_(len(bz), 10)
  nose.tools.eq_(bz.dtype, numpy.uint8)

  bz = bzarray((20000,), dtype='bool')
  nose.tools.eq_(bz.shape, (20000,))
  nose.tools.eq_(len(bz), 20000)
  nose.tools.eq_(bz.dtype, numpy.bool_)

  bz = bzarray((3,3), dtype='uint32')
  nose.tools.eq_(bz.shape, (3,3))
  nose.tools.eq_(len(bz), 9)
  nose.tools.eq_(bz.dtype, numpy.uint32)

@nose.tools.raises(ValueError)
def test_array_negative_size():

  bz = bzarray(-2, dtype='uint8')

@nose.tools.raises(ValueError)
def test_array_zero_size():

  bz = bzarray(0, dtype='uint8')
  nose.tools.eq_(len(bz), 0)

def test_array_assign_and_read_u8():

  bz = bzarray(2, dtype='uint8')

  # assign a value to each position, check back
  bz[0] = 3
  nose.tools.eq_(bz[0].dtype, numpy.uint8)
  nose.tools.eq_(bz[0], 3)
  bz[1] = 12
  nose.tools.eq_(bz[0].dtype, numpy.uint8)
  nose.tools.eq_(bz[1], 12)

def test_array_assign_and_read_u32():

  bz = bzarray(2, dtype='uint32')

  # assign a value to each position, check back
  bz[0] = 3
  nose.tools.eq_(bz[0].dtype, numpy.uint32)
  nose.tools.eq_(bz[0], 3)
  bz[1] = 0
  nose.tools.eq_(bz[0].dtype, numpy.uint32)
  nose.tools.eq_(bz[1], 0)

def test_array_assign_and_read_u32d2():

  bz = bzarray((2,2), dtype='uint32')

  # assign a value to each position, check back
  bz[0,0] = 3
  bz[0,1] = 12
  bz[1,0] = 25
  bz[1,1] = 255
  nose.tools.eq_(bz[0,0].dtype, numpy.uint32)
  nose.tools.eq_(bz[0,0], 3)
  nose.tools.eq_(bz[0,1].dtype, numpy.uint32)
  nose.tools.eq_(bz[0,1], 12)
  nose.tools.eq_(bz[1,0].dtype, numpy.uint32)
  nose.tools.eq_(bz[1,0], 25)
  nose.tools.eq_(bz[1,1].dtype, numpy.uint32)
  nose.tools.eq_(bz[1,1], 255)

def test_array_assign_and_read_c128d2():

  bz = bzarray((2,2), dtype='complex128')
  bz[0,0] = complex(3, 4.2)
  bz[0,1] = complex(1.5, 2)
  bz[1,0] = complex(33, 4)
  bz[1,1] = complex(2, 2)
  nose.tools.eq_(bz[0,0].dtype, numpy.complex128)
  nose.tools.eq_(bz[0,0], complex(3,4.2))
  nose.tools.eq_(bz[0,1].dtype, numpy.complex128)
  nose.tools.eq_(bz[0,1], complex(1.5,2))
  nose.tools.eq_(bz[1,0].dtype, numpy.complex128)
  nose.tools.eq_(bz[1,0], complex(33,4))
  nose.tools.eq_(bz[1,1].dtype, numpy.complex128)
  nose.tools.eq_(bz[1,1], complex(2,2))

@nose.tools.raises(IndexError)
def test_array_protect_segfault_high_get():

  bz = bzarray(2, dtype='complex64')
  k = bz[3]

@nose.tools.raises(IndexError)
def test_array_protect_segfault_high_set():

  bz = bzarray(2, dtype='complex64')
  bz[3] = complex(2,3)

@nose.tools.raises(IndexError)
def test_array_protect_segfault_low_get():

  bz = bzarray(2, dtype='complex128')
  k = bz[-3]

@nose.tools.raises(IndexError)
def test_array_protect_segfault_low_set():

  bz = bzarray(2, dtype='complex128')
  bz[-3] = complex(2,3)

def test_u8d1_as_ndarray():

  bz = bzarray(2, dtype='uint8')
  bz[0] = 32
  bz[1] = 10
  nd = bz.as_ndarray()
  nose.tools.eq_(nd.shape, bz.shape)
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

def test_u64d1_as_ndarray():

  bz = bzarray(2, dtype='uint64')
  bz[0] = 2**33
  bz[1] = 2**64 - 1
  nd = bz.as_ndarray()
  nose.tools.eq_(nd.shape, bz.shape)
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

def test_u32d1_as_ndarray():

  bz = bzarray(2, dtype='uint32')
  bz[0] = 2**32-1
  bz[1] = 0
  nd = bz.as_ndarray()
  nose.tools.eq_(nd.shape, bz.shape)
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

def test_s64d2_shallow_array():

  bz = bzarray((2,2), dtype='int64')
  bz[0,0] = 1
  bz[0,1] = 2
  bz[1,0] = 3
  bz[1,1] = -1
  nd = bz.as_shallow_ndarray()
  nose.tools.eq_(nd.shape, bz.shape)
  nose.tools.eq_(bz[0,0], nd[0,0])
  nose.tools.eq_(bz[0,1], nd[0,1])
  nose.tools.eq_(bz[1,0], nd[1,0])
  nose.tools.eq_(bz[1,1], nd[1,1])
  nose.tools.eq_(nd.base, bz)
  del bz
  assert nd.flags.behaved
  assert nd.flags.c_contiguous
  assert nd.flags.writeable
  nose.tools.eq_(nd.flags.owndata, False)

  nose.tools.eq_(nd.dtype, numpy.dtype('int64'))
  nose.tools.eq_(nd[0,0], 1)
  nose.tools.eq_(nd[0,1], 2)
  nose.tools.eq_(nd[1,0], 3)
  nose.tools.eq_(nd[1,1], -1)

  nd[1,0] = 32
  nose.tools.eq_(nd.base[1,0], nd[1,0])

  # tests blitz::Array<> out lives attached ndarray
  bz = nd.base
  del nd
  nose.tools.eq_(bz.shape, (2,2))
  nose.tools.eq_(bz[0,0], 1)
  nose.tools.eq_(bz[0,1], 2)
  nose.tools.eq_(bz[1,0], 32)
  nose.tools.eq_(bz[1,1], -1)

@nose.tools.raises(ValueError)
def test_s64d2_cannot_resize_shallow():
  
  bz = bzarray((2,2), dtype='int64')
  bz[0,0] = 1
  bz[0,1] = 2
  bz[1,0] = 3
  bz[1,1] = -1
  nd = bz.as_shallow_ndarray()
  nd.resize(3,3)
