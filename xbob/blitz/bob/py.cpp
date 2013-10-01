/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  1 Oct 13:52:57 2013
 *
 * @brief Implements some constructions exported to all modules
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#include <bob/py.h>
#endif

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

static void pyobject_delete(PyObject* o) {
  Py_XDECREF(o);
}

static void pyobject_keep(PyObject* o) {
}

namespace bob { namespace python {

  void bob_import_array() {
    wrap_import_array();
  }

  std::shared_ptr<PyObject> new_reference(PyObject* o) {
    return std::shared_ptr<PyObject>(o, &pyobject_deleter);
  }

  std::shared_ptr<PyObject> borrowed(PyObject* o) {
    return std::shared_ptr<PyObject>(o, &pyobject_keep);
  }

  const char* num_to_str(int typenum) {
    switch (typenum) {
      case NPY_BOOL:
        {
          static char s[] = "bool";
          return s;
        }
      case NPY_UINT8:
        {
          static char s[] = "uint8";
          return s;
        }
      case NPY_UINT16:
        {
          static char s[] = "uint16";
          return s;
        }
      case NPY_UINT32:
        {
          static char s[] = "uint32";
          return s;
        }
      case NPY_UINT64:
        {
          static char s[] = "uint64";
          return s;
        }
      case NPY_INT8:
        {
          static char s[] = "int8";
          return s;
        }
      case NPY_INT16:
        {
          static char s[] = "int16";
          return s;
        }
      case NPY_INT32:
        {
          static char s[] = "int32";
          return s;
        }
      case NPY_INT64:
        {
          static char s[] = "int64";
          return s;
        }
      case NPY_FLOAT32:
        {
          static char s[] = "float32";
          return s;
        }
      case NPY_FLOAT64:
        {
          static char s[] = "float64";
          return s;
        }
#ifdef NPY_FLOAT128
      case NPY_FLOAT128:
        {
          static char s[] = "float128";
          return s;
        }
#endif
      case NPY_COMPLEX64:
        {
          static char s[] = "complex64";
          return s;
        }
      case NPY_COMPLEX128:
        {
          static char s[] = "complex128";
          return s;
        }
#ifdef NPY_COMPLEX256
      case NPY_COMPLEX256:
        {
          static char s[] = "complex256";
          return s;
        }
#endif
      default:
        PyErr_Format(PyExc_NotImplementedError, "no support for converting type number %d to string", typenum);
        return 0;
    }
  }

  template <> int ctype_to_num<bool>() { return NPY_BOOL; }
  template <> int ctype_to_num<int8_t>() { return NPY_INT8; }
  template <> int ctype_to_num<uint8_t>() { return NPY_UINT8; }
  template <> int ctype_to_num<int16_t>() { return NPY_INT16; }
  template <> int ctype_to_num<uint16_t>() { return NPY_UINT16; }
  template <> int ctype_to_num<int32_t>() { return NPY_INT32; }
  template <> int ctype_to_num<uint32_t>() { return NPY_UINT32; }
  template <> int ctype_to_num<int64_t>() { return NPY_INT64; }
  template <> int ctype_to_num<uint64_t>() { return NPY_UINT64; }
  template <> int ctype_to_num<float>() { return NPY_FLOAT32; }
  template <> int ctype_to_num<double>() { return NPY_FLOAT64; }
#ifdef NPY_FLOAT128
  template <> int ctype_to_num<long double>() { return NPY_FLOAT128; }
#endif
  template <> int ctype_to_num<std::complex<float>>() { return NPY_COMPLEX64; }
  template <> int ctype_to_num<std::complex<double>>() { return NPY_COMPLEX128; }
#ifdef NPY_COMPLEX256
  template <> int ctype_to_num<std::complex<long double>>() { return NPY_COMPLEX256; }
#endif

#ifdef __APPLE__
  template <> int ctype_to_num<long>() {
    if (sizeof(long) == 4) return NPY_INT32;
    return NPY_INT64;
  }

  template <> int ctype_to_num<unsigned long>() {
    if (sizeof(unsigned long) == 4) return NPY_UINT32;
    return NPY_UINT64;
  }
#endif

}}
