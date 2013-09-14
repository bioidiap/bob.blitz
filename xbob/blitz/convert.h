/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 12 Sep 08:16:31 2013
 *
 * @brief Detects and converts blitz::Array<T,N> to and from numpy.ndarray
 */

#ifndef XBOB_BLITZ_CONVERT_H
#define XBOB_BLITZ_CONVERT_H

#include <Python.h>
#include <blitz/array.h>
#include <stdint.h>

/**
 * Imports the numpy.ndarray infrastructure once
 */
void import_ndarray();

/**
 * Shallow conversion from blitz::Array<T,N> to numpy.ndarray
 */
PyObject* shallow_ndarray_u8d1(blitz::Array<uint8_t,1>& a, PyObject* owner);

#endif /* XBOB_BLITZ_CONVERT_H */
