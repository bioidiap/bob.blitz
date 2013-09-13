/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 12 Sep 08:16:31 2013 
 *
 * @brief Detects and converts blitz::Array<T,N> to and from numpy.ndarray
 */

#ifndef XBOB_BLITZ_CONVERT_H
#define XBOB_BLITZ_CONVERT_H

#include <blitz/array.h>
#include <stdint.h>
#include <Python.h>

/**
 * Shallow conversion from blitz::Array<T,N> to numpy.ndarray
 */
PyObject* shallow_ndarray_u8d1(blitz::Array<uint8_t,1>& a);

#endif /* XBOB_BLITZ_CONVERT_H */
