/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 12 Sep 18:40:06 2013
 *
 * @brief Implementation of conversion routines
 */

#define xbob_IMPORT_ARRAY
#include <blitz_numpy.h>
#undef xbob_IMPORT_ARRAY

#if PY_VERSION_HEX >= 0x03000000
static void* wrap_import_array() {
  import_array();
  return 0;
}
#else
static void wrap_import_array() {
  import_array();
  return;
}
#endif

void import_ndarray() {
  static bool array_imported = false;
  if (array_imported) return;
  wrap_import_array();
  array_imported = true;
}

/**
 * @brief Handles conversion checking possibilities
 */
typedef enum {
  IMPOSSIBLE = 0,    ///< not possible to get array from object
  BYREFERENCE = 1,   ///< possible, by only referencing the array
  WITHARRAYCOPY = 2, ///< possible, object is an array, but has to copy
  WITHCOPY = 3       ///< possible, object is not an array, has to copy
} convert_t;

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
