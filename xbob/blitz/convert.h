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
#include <stdexcept>

// ============================================================================
// Note: Header files that are distributed and include numpy/arrayobject.h need
//       to have these protections. Be warned.

// Defines a unique symbol for the API
#if !defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PY_ARRAY_UNIQUE_SYMBOL xbob_NUMPY_ARRAY_API
#endif

// Normally, don't import_array(), except if xbob_IMPORT_ARRAY is defined.
#if !defined(xbob_IMPORT_ARRAY) and !defined(NO_IMPORT_ARRAY)
#define NO_IMPORT_ARRAY
#endif

// Define the numpy C-API we are compatible with
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// Finally, we include numpy's arrayobject header. Not before!
#include <numpy/arrayobject.h>
// ============================================================================

#define NUMPY17_API 0x00000007
#define NUMPY16_API 0x00000006
#define NUMPY14_API 0x00000004

/**
 * Imports the numpy.ndarray infrastructure once
 */
void import_ndarray();

/**
 * @brief Converts from C/C++ type to ndarray type_num.
 */
template <typename T> int ctype_to_num() {
  throw std::runtime_error("unsupported C/C++ type");
}

template <> int ctype_to_num<bool>(void);
template <> int ctype_to_num<int8_t>(void);
template <> int ctype_to_num<uint8_t>(void);
template <> int ctype_to_num<int16_t>(void);
template <> int ctype_to_num<uint16_t>(void);
template <> int ctype_to_num<int32_t>(void);
template <> int ctype_to_num<uint32_t>(void);
template <> int ctype_to_num<int64_t>(void);
template <> int ctype_to_num<uint64_t>(void);
template <> int ctype_to_num<float>(void);
template <> int ctype_to_num<double>(void);
#ifdef NPY_FLOAT128
template <> int ctype_to_num<long double>(void);
#endif
template <> int ctype_to_num<std::complex<float> >(void);
template <> int ctype_to_num<std::complex<double> >(void);
#ifdef NPY_COMPLEX256
template <> int ctype_to_num<std::complex<long double> >(void);
#endif

// maximum number of dimensions supported by this converter
#define XBOB_BLITZ_MAXDIM 11

/**
 * @brief Encapsulation of special type information of interfaces.
 */
struct typeinfo {

  int type_num; ///< data type
  Py_ssize_t nd; ///< number of dimensions
  Py_ssize_t shape[XBOB_BLITZ_MAXDIM]; ///< length along each dimension
  Py_ssize_t stride[XBOB_BLITZ_MAXDIM]; ///< strides in each dimension

  /**
   * @brief Constructs from a blitz::Array<T,N>
   */
  template <typename T, int N> typeinfo(const blitz::Array<T,N>& a) {
    // maximum supported number of dimensions
    if (N > XBOB_BLITZ_MAXDIM) {
      throw std::runtime_error("can only work with blitz::Array<>'s with up to 11 dimensions");
    }

    nd = N;
    type_num = ctype_to_num<T>();
    for (int i=0; i<N; ++i) {
      shape[i] = a.extent(i);
      stride[i] = a.stride(i);
    }
  }
};

/**
 * Returns a shallow ndarray from a blitz::Array<T,N>
 */
template<typename T, int N>
PyObject* shallow_ndarray(blitz::Array<T,N>& a, PyObject* owner) {

  // maximum supported number of dimensions
  if (N > XBOB_BLITZ_MAXDIM) {
    throw std::runtime_error("can only work with blitz::Array<>'s with up to 11 dimensions");
  }

  // array has to be contiguous
  if (!a.isStorageContiguous()) {
    throw std::runtime_error("input blitz::Array<T,N> is not contiguous");
  }

  // array has to be in C-order
  for (int i=0; i<N; ++i) {
    if (!(a.isRankStoredAscending(i) && a.ordering(i)==a.rank()-1-i)) {
      throw std::runtime_error("input blitz::Array<T,N> is not stored in C-order");
    }
  }

  // array base has to be zero
  for (int i=0; i<N; ++i) {
    if (a.base(i) != 0) {
      throw std::runtime_error("input blitz::Array<T,N> is not zero-based in all dimensions");
    }
  }

  //if you survived to this point, converts into shallow numpy.ndarray
  typeinfo info(a);
#if NPY_FEATURE_VERSION > NUMPY16_API /* NumPy C-API version > 1.6 */
  int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
#else
  int flags = NPY_C_CONTIGUOUS | NPY_ALIGNED;
#endif
  PyObject* retval = PyArray_New(&PyArray_Type, N, info.shape, info.type_num,
      info.stride, static_cast<void*>(a.data()), sizeof(T), flags, owner);

  //set base object so the array can go independently
#if NPY_FEATURE_VERSION > NUMPY16_API /* NumPy C-API version > 1.6 */
  int status = PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(retval), owner);
  if (status != 0) {
    Py_XDECREF(retval);
    throw std::runtime_error("cannot set base object of numpy.ndarray");
  }
#else
  PyArray_BASE(retval) = owner;
#endif
  Py_INCREF(owner);

  return retval;
}

#endif /* XBOB_BLITZ_CONVERT_H */
