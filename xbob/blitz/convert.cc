/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 12 Sep 18:40:06 2013
 *
 * @brief Implementation of conversion routines
 */

#include <convert.h>
#include <stdexcept>

// Define the numpy C-API we are compatible with
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// Finally, we include numpy's arrayobject header. Not before!
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

// maximum number of dimensions supported by this converter
#define XBOB_BLITZ_MAXDIM 11

#define NUMPY17_API 0x00000007
#define NUMPY16_API 0x00000006
#define NUMPY14_API 0x00000004

/**
 * @brief Handles conversion checking possibilities
 */
typedef enum {
  IMPOSSIBLE = 0,    ///< not possible to get array from object
  BYREFERENCE = 1,   ///< possible, by only referencing the array
  WITHARRAYCOPY = 2, ///< possible, object is an array, but has to copy
  WITHCOPY = 3       ///< possible, object is not an array, has to copy
} convert_t;

/**
 * @brief Converts from C/C++ type to ndarray type_num.
 */
template <typename T> int ctype_to_num() {
  throw std::runtime_error("unsupported C/C++ type");
}

template <> int ctype_to_num<bool>(void)     { return NPY_BOOL; }
template <> int ctype_to_num<int8_t>(void)   { return NPY_INT8; }
template <> int ctype_to_num<uint8_t>(void)  { return NPY_UINT8; }
template <> int ctype_to_num<int16_t>(void)  { return NPY_INT16; }
template <> int ctype_to_num<uint16_t>(void) { return NPY_UINT16; }
template <> int ctype_to_num<int32_t>(void)  { return NPY_INT32; }
template <> int ctype_to_num<uint32_t>(void) { return NPY_UINT32; }
template <> int ctype_to_num<int64_t>(void)  { return NPY_INT64; }
template <> int ctype_to_num<uint64_t>(void) { return NPY_UINT64; }
template <> int ctype_to_num<float>(void)    { return NPY_FLOAT32; }
template <> int ctype_to_num<double>(void)   { return NPY_FLOAT64; }
#ifdef NPY_FLOAT128
template <> int ctype_to_num<long double>(void) { return NPY_FLOAT128; }
#endif
template <> int ctype_to_num<std::complex<float> >(void)
{ return NPY_COMPLEX64; }
template <> int ctype_to_num<std::complex<double> >(void)
{ return NPY_COMPLEX128; }
#ifdef NPY_COMPLEX256
template <> int ctype_to_num<std::complex<long double> >(void)
{ return NPY_COMPLEX256; }
#endif

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
  int status = PyArray_SetBaseObject(retval, owner);
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

PyObject* shallow_ndarray_u8d1(blitz::Array<uint8_t,1>& a, PyObject* owner) {
  return shallow_ndarray<uint8_t,1>(a, owner);
}
