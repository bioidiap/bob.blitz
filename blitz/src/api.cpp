/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  1 Oct 13:52:57 2013
 *
 * @brief Implements some constructions exported to all modules
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#include <blitz.array/helper.h>
#endif

static void pyobject_delete(PyObject* o) {
  Py_XDECREF(o);
}

static void pyobject_keep(PyObject* o) {
}

template<typename T, int N>
std::shared_ptr<void> allocate_innest(Py_ssize_t* shape) {

  blitz::TinyVector<int,N> tv_shape;
  for (int i=0; i<N; ++i) tv_shape(i) = shape[i];
  auto retval = std::make_shared<blitz::Array<T,N>>(tv_shape);

  /** some test code
  std::cout << "allocating array" << std::endl;
  std::shared_ptr<blitz::Array<T,N>> retval(new blitz::Array<T,N>(tv_shape),
      &delete_array<T,N>);
  **/

  return retval;

}

template<typename T>
std::shared_ptr<void> allocate_inner(Py_ssize_t ndim, Py_ssize_t* shape) {
  switch (ndim) {

    case 1: 
      return allocate_innest<T,1>(shape);

    case 2: 
      return allocate_innest<T,2>(shape);

    case 3: 
      return allocate_innest<T,3>(shape);
     
    case 4: 
      return allocate_innest<T,4>(shape);

    default:
      PyErr_Format(PyExc_TypeError, "cannot allocate blitz::Array<> array with %" PY_FORMAT_SIZE_T "d dimensions", ndim);
      return std::shared_ptr<void>();
  }

}

template <typename T>
PyObject* getitem_inner(PyArray_Descr* dtype, Py_ssize_t ndim,
    std::shared_ptr<void> bz, Py_ssize_t* pos) {

  switch (ndim) {

    case 1:
      {
        T& val = (*reinterpret_cast<blitz::Array<T,1>*>(bz.get()))((int)pos[0]);
        return PyArray_Scalar(&val, dtype, 0);
      }

    case 2:
      {
        T& val = (*reinterpret_cast<blitz::Array<T,2>*>(bz.get()))((int)pos[0], (int)pos[1]);
        return PyArray_Scalar(&val, dtype, 0);
      }

    case 3:
      {
        T& val = (*reinterpret_cast<blitz::Array<T,2>*>(bz.get()))((int)pos[0], (int)pos[1], (int)pos[2]);
        return PyArray_Scalar(&val, dtype, 0);
      }

    case 4:
      {
        T& val = (*reinterpret_cast<blitz::Array<T,2>*>(bz.get()))((int)pos[0], (int)pos[1], (int)pos[2], (int)pos[3]);
        return PyArray_Scalar(&val, dtype, 0);
      }

    default:
      PyErr_Format(PyExc_TypeError, "cannot index blitz::Array<> array with %" PY_FORMAT_SIZE_T "d dimensions", ndim);
      return 0;
  }
}

/**
 * Sets a given item from the blitz::Array<>
 */
template <typename T> int setitem_inner(Py_ssize_t ndim, std::shared_ptr<void> bz, Py_ssize_t* pos, PyObject* value) {

  switch (ndim) {

    case 1:
      {
        T tmp = pybz::detail::extract<T>(value);
        if (PyErr_Occurred()) return -1;
        (*reinterpret_cast<blitz::Array<T,1>*>(bz.get()))((int)pos[0]) = tmp;
        return 0;
      }

    case 2:
      {
        T tmp = pybz::detail::extract<T>(value);
        if (PyErr_Occurred()) return -1;
        (*reinterpret_cast<blitz::Array<T,2>*>(bz.get()))((int)pos[0], (int)pos[1]) = tmp;
        return 0;
      }

    case 3:
      {
        T tmp = pybz::detail::extract<T>(value);
        if (PyErr_Occurred()) return -1;
        (*reinterpret_cast<blitz::Array<T,3>*>(bz.get()))((int)pos[0], (int)pos[1], (int)pos[2]) = tmp;
        return 0;
      }

    case 4:
      {
        T tmp = pybz::detail::extract<T>(value);
        if (PyErr_Occurred()) return -1;
        (*reinterpret_cast<blitz::Array<T,4>*>(bz.get()))((int)pos[0], (int)pos[1], (int)pos[2], (int)pos[3]) = tmp;
        return 0;
      }

    default:
      PyErr_Format(PyExc_TypeError, "cannot index blitz::Array<> array with %" PY_FORMAT_SIZE_T "d dimensions", ndim);
      return -1;
  }

}

template <typename T>
PyObject* ndarray_copy_inner(Py_ssize_t ndim, std::shared_ptr<void> bz) {

  switch (ndim) {

    case 1:
      {
        return pybz::detail::ndarray_copy(*reinterpret_cast<blitz::Array<T,1>*>(bz.get()));
      }

    case 2:
      {
        return pybz::detail::ndarray_copy(*reinterpret_cast<blitz::Array<T,2>*>(bz.get()));
      }

    case 3:
      {
        return pybz::detail::ndarray_copy(*reinterpret_cast<blitz::Array<T,3>*>(bz.get()));
      }

    case 4:
      {
        return pybz::detail::ndarray_copy(*reinterpret_cast<blitz::Array<T,4>*>(bz.get()));
      }

    default:

      PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<> to numpy.ndarray with number of dimensions = %" PY_FORMAT_SIZE_T "d", ndim);
      return 0;
  }

}

template <typename T>
PyObject* ndarray_shallow_inner(Py_ssize_t ndim, std::shared_ptr<void> bz) {

  switch (ndim) {

    case 1:
      {
        return pybz::detail::ndarray_shallow(*reinterpret_cast<blitz::Array<T,1>*>(bz.get()));
      }

    case 2:
      {
        return pybz::detail::ndarray_shallow(*reinterpret_cast<blitz::Array<T,2>*>(bz.get()));
      }

    case 3:
      {
        return pybz::detail::ndarray_shallow(*reinterpret_cast<blitz::Array<T,3>*>(bz.get()));
      }

    case 4:
      {
        return pybz::detail::ndarray_shallow(*reinterpret_cast<blitz::Array<T,4>*>(bz.get()));
      }

    default:
      PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<> to numpy.ndarray with number of dimensions = %" PY_FORMAT_SIZE_T "d", ndim);
      return 0;

  }

}

namespace pybz { namespace detail {

  void numpy_import_array() {
    wrap_import_array();
  }

  std::shared_ptr<PyObject> handle(PyObject* o) {
    return std::shared_ptr<PyObject>(o, &pyobject_delete);
  }

  std::shared_ptr<PyObject> borrowed(PyObject* o) {
    return std::shared_ptr<PyObject>(o, &pyobject_keep);
  }

  PyObject* new_reference(std::shared_ptr<PyObject> o) {
    PyObject* retval = o.get();
    if (retval) {
      Py_INCREF(retval);
      return retval;
    }
    return retval;
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

  std::shared_ptr<void> allocate(int typenum, Py_ssize_t ndim,
      Py_ssize_t* shape) {

    switch (typenum) {

      case NPY_BOOL: 
        return allocate_inner<bool>(ndim, shape);

      case NPY_INT8: 
        return allocate_inner<int8_t>(ndim, shape);

      case NPY_INT16: 
        return allocate_inner<int16_t>(ndim, shape);

      case NPY_INT32: 
        return allocate_inner<int32_t>(ndim, shape);

      case NPY_INT64: 
        return allocate_inner<int64_t>(ndim, shape);

      case NPY_UINT8: 
        return allocate_inner<uint8_t>(ndim, shape);

      case NPY_UINT16: 
        return allocate_inner<uint16_t>(ndim, shape);

      case NPY_UINT32: 
        return allocate_inner<uint32_t>(ndim, shape);

      case NPY_UINT64: 
        return allocate_inner<uint64_t>(ndim, shape);

      case NPY_FLOAT32: 
        return allocate_inner<float>(ndim, shape);

      case NPY_FLOAT64: 
        return allocate_inner<double>(ndim, shape);

#ifdef NPY_FLOAT128
      case NPY_FLOAT128: 
        return allocate_inner<long double>(ndim, shape);

#endif

      case NPY_COMPLEX64: 
        return allocate_inner<std::complex<float>>(ndim, shape);

      case NPY_COMPLEX128: 
        return allocate_inner<std::complex<double>>(ndim, shape);

#ifdef NPY_COMPLEX256
      case NPY_COMPLEX256: 
        return allocate_inner<std::complex<long double>>(ndim, shape);

#endif

      default:
        PyErr_Format(PyExc_TypeError, 
            "cannot create array with data type number = %d", typenum);
        return std::shared_ptr<void>();

    }

  }

  PyObject* getitem(PyArray_Descr* dtype, Py_ssize_t ndim,
      std::shared_ptr<void> bz, Py_ssize_t* pos) {

    switch (dtype->type_num) {
      case NPY_BOOL: 
        return getitem_inner<bool>(dtype, ndim, bz, pos);

      case NPY_INT8: 
        return getitem_inner<int8_t>(dtype, ndim, bz, pos);

      case NPY_INT16: 
        return getitem_inner<int16_t>(dtype, ndim, bz, pos);

      case NPY_INT32: 
        return getitem_inner<int32_t>(dtype, ndim, bz, pos);

      case NPY_INT64: 
        return getitem_inner<int64_t>(dtype, ndim, bz, pos);

      case NPY_UINT8: 
        return getitem_inner<uint8_t>(dtype, ndim, bz, pos);

      case NPY_UINT16: 
        return getitem_inner<uint16_t>(dtype, ndim, bz, pos);

      case NPY_UINT32: 
        return getitem_inner<uint32_t>(dtype, ndim, bz, pos);

      case NPY_UINT64: 
        return getitem_inner<uint64_t>(dtype, ndim, bz, pos);

      case NPY_FLOAT32: 
        return getitem_inner<float>(dtype, ndim, bz, pos);

      case NPY_FLOAT64: 
        return getitem_inner<double>(dtype, ndim, bz, pos);

#ifdef NPY_FLOAT128
      case NPY_FLOAT128: 
        return getitem_inner<long double>(dtype, ndim, bz, pos);

#endif

      case NPY_COMPLEX64: 
        return getitem_inner<std::complex<float>>(dtype, ndim, bz, pos);

      case NPY_COMPLEX128: 
        return getitem_inner<std::complex<double>>(dtype, ndim, bz, pos);

#ifdef NPY_COMPLEX256
      case NPY_COMPLEX256: 
        return getitem_inner<std::complex<long double>>(dtype, ndim, bz, pos);

#endif

      default:
        PyErr_Format(PyExc_TypeError, "cannot index array with data type number = %d", dtype->type_num);
        return 0;

    }

  }

  int setitem(PyArray_Descr* dtype, Py_ssize_t ndim, std::shared_ptr<void> bz, Py_ssize_t* pos, PyObject* value) {
    switch (dtype->type_num) {

      case NPY_BOOL: 
        return setitem_inner<bool>(ndim, bz, pos, value);

      case NPY_INT8: 
        return setitem_inner<int8_t>(ndim, bz, pos, value);

      case NPY_INT16: 
        return setitem_inner<int16_t>(ndim, bz, pos, value);

      case NPY_INT32: 
        return setitem_inner<int32_t>(ndim, bz, pos, value);

      case NPY_INT64: 
        return setitem_inner<int64_t>(ndim, bz, pos, value);

      case NPY_UINT8: 
        return setitem_inner<uint8_t>(ndim, bz, pos, value);

      case NPY_UINT16: 
        return setitem_inner<uint16_t>(ndim, bz, pos, value);

      case NPY_UINT32: 
        return setitem_inner<uint32_t>(ndim, bz, pos, value);

      case NPY_UINT64: 
        return setitem_inner<uint64_t>(ndim, bz, pos, value);

      case NPY_FLOAT32: 
        return setitem_inner<float>(ndim, bz, pos, value);

      case NPY_FLOAT64: 
        return setitem_inner<double>(ndim, bz, pos, value);

#ifdef NPY_FLOAT128
      case NPY_FLOAT128: 
        return setitem_inner<long double>(ndim, bz, pos, value);

#endif

      case NPY_COMPLEX64: 
        return setitem_inner<std::complex<float>>(ndim, bz, pos, value);

      case NPY_COMPLEX128: 
        return setitem_inner<std::complex<double>>(ndim, bz, pos, value);

#ifdef NPY_COMPLEX256
      case NPY_COMPLEX256: 
        return setitem_inner<std::complex<long double>>(ndim, bz, pos, value);

#endif

      default:
        PyErr_Format(PyExc_TypeError, "cannot index array with data type number = %d", dtype->type_num);
        return -1;

    }

  }

  PyObject* ndarray_copy(int typenum, Py_ssize_t ndim,
      std::shared_ptr<void> bz) {

    switch (typenum) {

      case NPY_BOOL: 
        return ndarray_copy_inner<bool>(ndim, bz);

      case NPY_INT8: 
        return ndarray_copy_inner<int8_t>(ndim, bz);

      case NPY_INT16: 
        return ndarray_copy_inner<int16_t>(ndim, bz);

      case NPY_INT32: 
        return ndarray_copy_inner<int32_t>(ndim, bz);
        
      case NPY_INT64: 
        return ndarray_copy_inner<int64_t>(ndim, bz);

      case NPY_UINT8: 
        return ndarray_copy_inner<uint8_t>(ndim, bz);

      case NPY_UINT16: 
        return ndarray_copy_inner<uint16_t>(ndim, bz);

      case NPY_UINT32: 
        return ndarray_copy_inner<uint32_t>(ndim, bz);

      case NPY_UINT64: 
        return ndarray_copy_inner<uint64_t>(ndim, bz);

      case NPY_FLOAT32: 
        return ndarray_copy_inner<float>(ndim, bz);

      case NPY_FLOAT64: 
        return ndarray_copy_inner<double>(ndim, bz);

#ifdef NPY_FLOAT128
      case NPY_FLOAT128: 
        return ndarray_copy_inner<long double>(ndim, bz);

#endif
 
      case NPY_COMPLEX64: 
        return ndarray_copy_inner<std::complex<float>>(ndim, bz);

      case NPY_COMPLEX128: 
        return ndarray_copy_inner<std::complex<double>>(ndim, bz);

#ifdef NPY_COMPLEX256
      case NPY_COMPLEX256: 
        return ndarray_copy_inner<std::complex<long double>>(ndim, bz);

#endif

      default:
        PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<> to numpy.ndarray with data type number = %d", typenum);
        return 0;

    }
  }

  PyObject* ndarray_shallow(int typenum, Py_ssize_t ndim,
      std::shared_ptr<void> bz) {

    switch (typenum) {

      case NPY_BOOL: 
        return ndarray_shallow_inner<bool>(ndim, bz);

      case NPY_INT8: 
        return ndarray_shallow_inner<int8_t>(ndim, bz);

      case NPY_INT16: 

        return ndarray_shallow_inner<int16_t>(ndim, bz);
      case NPY_INT32: 

        return ndarray_shallow_inner<int32_t>(ndim, bz);
      case NPY_INT64:

        return ndarray_shallow_inner<int64_t>(ndim, bz);
      case NPY_UINT8: 

        return ndarray_shallow_inner<uint8_t>(ndim, bz);

      case NPY_UINT16: 
        return ndarray_shallow_inner<uint16_t>(ndim, bz);
        
      case NPY_UINT32: 
        return ndarray_shallow_inner<uint32_t>(ndim, bz);
        
      case NPY_UINT64: 
        return ndarray_shallow_inner<uint64_t>(ndim, bz);

      case NPY_FLOAT32: 
        return ndarray_shallow_inner<float>(ndim, bz);
        
      case NPY_FLOAT64: 
        return ndarray_shallow_inner<double>(ndim, bz);

#ifdef NPY_FLOAT128
      case NPY_FLOAT128: 
        return ndarray_shallow_inner<long double>(ndim, bz);
#endif

      case NPY_COMPLEX64: 
        return ndarray_shallow_inner<std::complex<float>>(ndim, bz);

      case NPY_COMPLEX128: 
        return ndarray_shallow_inner<std::complex<double>>(ndim, bz);

#ifdef NPY_COMPLEX256
      case NPY_COMPLEX256: 
        return ndarray_shallow_inner<std::complex<long double>>(ndim, bz);
#endif

      default:
        PyErr_Format(PyExc_TypeError, "cannot convert blitz::Array<> to numpy.ndarray with data type number = %d", typenum);
        return 0;

    }

  }

}}
