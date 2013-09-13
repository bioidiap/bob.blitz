/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 12 Sep 18:40:06 2013
 *
 * @brief Implementation of conversion routines
 */

#include <convert.h>
#include <stdexcept>

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

// maximum number of dimensions supported by this converter
#define XBOB_BLITZ_MAXDIM 11

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

// The C/C++ types we support should be declared here.
template <> int ctype_to_num<bool>();
template <> int ctype_to_num<signed char>();
template <> int ctype_to_num<unsigned char>();
template <> int ctype_to_num<short>();
template <> int ctype_to_num<unsigned short>();
template <> int ctype_to_num<int>();
template <> int ctype_to_num<unsigned int>();
template <> int ctype_to_num<long>();
template <> int ctype_to_num<unsigned long>();
template <> int ctype_to_num<long long>();
template <> int ctype_to_num<unsigned long long>();
template <> int ctype_to_num<float>();
template <> int ctype_to_num<double>();
#ifdef NPY_FLOAT128
template <> int ctype_to_num<long double>();
#endif
template <> int ctype_to_num<std::complex<float> >();
template <> int ctype_to_num<std::complex<double> >();
#ifdef NPY_COMPLEX256
template <> int ctype_to_num<std::complex<long double> >();
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
      shape = a.extent(i);
      stride = a.stride(i);
    }
  }
};

/**
 * Returns a shallow ndarray from a blitz::Array<T,N>
 */
template<typename T, int N> PyObject* shallow_ndarray(blitz::Array<T,N>& a) {

  //check
  if (!a.isStorageContiguous()) {
    throw std::runtime_error("input blitz::Array<T,N> is not contiguous");
  }

  for (int i=0; i<a.rank(); ++i) {
    if (!(a.isRankStoredAscending(i) && a.ordering(i)==a.rank()-1-i)) {
      throw std::runtime_error("input blitz::Array<T,N> is not stored in C-order");
    }
  }

  for (int i=0; i<a.rank(); ++i) {
    if (a.base(i)!=0 ) {
      throw std::runtime_error("input blitz::Array<T,N> is not zero-based in all dimensions");
    }
  }

  //if you survived to this point, converts into shallow numpy.ndarray
  typeinfo_t info(a);
  return PyArray_New(&PyArray_Type, N, info.shape, info.type_num, info.strides,
      static_cast<void*>(a.data()), 0, 0);

}

PyObject* shallow_ndarray_u8d1(blitz::Array<uint8_t,1>& a); {
  return shallow_ndarray<uint8_t,1>(a);
}
