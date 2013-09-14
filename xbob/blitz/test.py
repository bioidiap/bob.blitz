#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Fri 13 Sep 2013 11:55:09 CEST

"""Test cases for the Blitz::Array <=> numpy.ndarray converters
"""

from xbob.blitz import array

import numpy
import nose.tools

def test_from_scratch():

  bz = array.u8d1(10)
  nose.tools.eq_(len(bz), 10)

  bz = array.u8d1(20000)
  nose.tools.eq_(len(bz), 20000)

@nose.tools.raises(OverflowError)
def test_negative_size():

  bz = array.u8d1(-1)

def test_zero_size():

  bz = array.u8d1(0)
  nose.tools.eq_(len(bz), 0)

def test_assign_and_read():

  bz = array.u8d1(2)

  # assign a value to each position, check back
  bz[0] = 3
  nose.tools.eq_(bz[0], 3)
  bz[1] = 12
  nose.tools.eq_(bz[1], 12)

@nose.tools.raises(IndexError)
def test_protect_segfault_high():

  bz = array.u8d1(2)
  bz[3] = 4

@nose.tools.raises(IndexError)
def test_protect_segfault_low():

  bz = array.u8d1(2)
  bz[-1] = 4

@nose.tools.raises(OverflowError)
def test_overflow_detection():

  bz = array.u8d1(1)
  bz[0] = 256

@nose.tools.raises(OverflowError)
def test_underflow_detection():

  bz = array.u8d1(1)
  bz[0] = -1

def test_shallow_ndarray():

  bz = array.u8d1(2)
  bz[0] = 22
  bz[1] = 4
  nd = bz.ndarray()
  assert nd.dtype == numpy.uint8
  assert nd.shape == (2,)
  assert nd[0] == 22
  assert nd[1] == 4
  assert nd.flags.owndata == False
  assert nd.flags.c_contiguous == True
  assert nd.flags.aligned == True
  assert nd.flags.writeable == False
  assert nd.base == bz
