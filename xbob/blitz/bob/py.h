/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 12 Sep 08:16:31 2013
 *
 * @brief Utility classes and functions for conversion between Bob C++
 * interface and Python's C-API.
 */

#ifndef BOB_PYTHON_H
#define BOB_PYTHON_H

#include <Python.h>
#include <complex>
#include <blitz/array.h>
#include <stdint.h>
#include <stdexcept>
#include <numpy/arrayobject.h>
#include <typeinfo>

#define NUMPY17_API 0x00000007
#define NUMPY16_API 0x00000006
#define NUMPY14_API 0x00000004

namespace bob { namespace python {

  /**
   * @brief Imports the numpy.ndarray infrastructure once
   */
  void bob_import_array();

  /**
   * Converts from numpy type_num to a string representation
   */
  const char* num_to_str(int typenum);

  /**
   * @brief Converts from C/C++ type to ndarray type_num.
   *
   * We cover only simple conversions (i.e., standard integers, floats and
   * complex numbers only). If the input type is not convertible, an exception
   * is set on the Python error stack. You must check `PyErr_Occurred()` after
   * a call to this function to make sure things are OK and act accordingly.
   * For example:
   *
   * int typenum = ctype_to_num<my_weird_type>(obj);
   * if (PyErr_Occurred()) return 0; ///< propagate exception
   *
   * Note: This generic implementation only raises an "NotImplementedError"
   * exception in Python.
   */
  template <typename T> int ctype_to_num() {
    PyErr_Format(PyExc_NotImplementedError, "c++ type to numpy type_num conversion unsupported for typeid.name() `%s'", typeid(T).name());
    return -1;
  }

  template <> int ctype_to_num<bool>();
  template <> int ctype_to_num<int8_t>();
  template <> int ctype_to_num<uint8_t>();
  template <> int ctype_to_num<int16_t>();
  template <> int ctype_to_num<uint16_t>();
  template <> int ctype_to_num<int32_t>();
  template <> int ctype_to_num<uint32_t>();
  template <> int ctype_to_num<int64_t>();
  template <> int ctype_to_num<uint64_t>();
  template <> int ctype_to_num<float>();
  template <> int ctype_to_num<double>();
#ifdef NPY_FLOAT128
  template <> int ctype_to_num<long double>();
#endif
  template <> int ctype_to_num<std::complex<float>>();
  template <> int ctype_to_num<std::complex<double>>();
#ifdef NPY_COMPLEX256
  template <> int ctype_to_num<std::complex<long double>>();
#endif

  // support for common C types which can be declared differently
  // depending on the hosting platform
  template <> int ctype_to_num<long>();
  template <> int ctype_to_num<unsigned long>();

  /**
   * @brief Extraction API for **simple** types.
   *
   * We cover only simple conversions (i.e., standard integers, floats and
   * complex numbers only). If the input object is not convertible to the given
   * type, an exception is set on the Python error stack. You must check
   * `PyErr_Occurred()` after a call to this function to make sure things are
   * OK and act accordingly. For example:
   *
   * auto z = extract<uint8_t>(obj);
   * if (PyErr_Occurred()) return 0; ///< propagate exception
   */
  template <typename T> T extract(PyObject* o) {

    int type_num = ctype_to_num<T>();
    if (PyErr_Occurred()) {
      T retval = 0;
      return retval;
    }

    PyArray_Descr* dtype = PyArray_DescrFromType(type_num);

    // create a zero-dimensional array on the expected type
    PyArrayObject* zerodim =
      reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(0, 0, type_num));

    if (!zerodim) {
      T retval = 0;
      return retval;
    }

    int status = PyArray_SETITEM(zerodim, 
        reinterpret_cast<char*>(PyArray_DATA(zerodim)), o);

    if (status != 0) {
      T retval = 0;
      return retval;
    }

    // note: this will decref `zerodim'
    PyObject* scalar=PyArray_Return(zerodim);

    if (!scalar) {
      T retval = 0;
      return retval;
    }
    
    T retval = 0;
    PyArray_ScalarAsCtype(scalar, &retval);
    Py_DECREF(scalar);
    return retval;
  }

  /**
   * @brief Wraps the input numpy ndarray with a blitz::Array<> skin.
   *
   * You should use this kind of conversion when you either want the ultimate
   * speed (as no data copy is involved on this procedure) or when you'd like
   * the resulting blitz::Array<> to be writeable, so that you can pass this as
   * an input argument to a function and get the results written to the
   * original numpy ndarray memory.
   *
   * Blitz++ is a little more inflexible than numpy ndarrays are, so there are
   * limitations in this conversion. For example, normally we can't wrap
   * non-contiguous memory areas. In such cases, an exception is set on the
   * Python error stack. You must check `PyErr_Occurred()` after a call to this
   * function to make sure things are OK and act accordingly. For example:
   *
   * auto z = shallow_blitz_array<uint8_t,4>(obj);
   * if (PyErr_Occurred()) return 0; ///< propagate exception
   *
   * Notice that the lifetime of the blitz::Array<> extracted with this
   * procedure is bound to the lifetime of the source numpy ndarray. You'd have
   * to copy it to create an independent object.
   */
  template <typename T, int N> blitz::Array<T,N> shallow_blitz_array
    (PyObject* o, bool readwrite) {

      int type_num = ctype_to_num<T>();
      if (PyErr_Occurred()) {
        // propagates exception
        blitz::Array<T,N> retval;
        return retval;
      }

      PyArray_Descr* req_dtype = PyArray_DescrFromType(type_num); ///< borrowed

      // has to be an ndarray
      if (!PyArray_Check(o)) {
        PyErr_Format(PyExc_TypeError, "object is not of type numpy.ndarray and cannot be wrapped as shallow blitz::Array<%s,%d>", type_num, N);
        blitz::Array<T,N> retval;
        return retval;
      }

      PyArrayObject* ao = reinterpret_cast<PyArrayObject*>(o);

      // checks that the data type matches
      if (!PyArray_EquivTypes(PyArray_DESCR(ao), req_dtype)) {
        PyErr_Format(PyExc_TypeError, "numpy.ndarray has data type `%c%d' and cannot be wrapped as shallow blitz::Array<%s,%d>", PyArray_DESCR(ao)->kind, PyArray_DESCR(ao)->elsize, num_to_str(type_num), N);
        blitz::Array<T,N> retval;
        return retval;
      }

      // checks that the shape matches
      if (PyArray_NDIM(ao) != N) {
        PyErr_Format(PyExc_TypeError, "numpy.ndarray has %d dimensions and cannot be wrapped as shallow blitz::Array<%s,%d>", PyArray_NDIM(ao), num_to_str(type_num), N);
        blitz::Array<T,N> retval;
        return retval;
      }

      // checks for writeability
      if (readwrite && !PyArray_ISWRITEABLE(ao)) {
        PyErr_Format(PyExc_TypeError, "numpy.ndarray is not writeable and cannot be wrapped as a read-writeable blitz::Array<%s,%d>", num_to_str(type_num), N);
        blitz::Array<T,N> retval;
        return retval;
      }

      // checks if the array is C-style, aligned and in contiguous memory
      if (!PyArray_ISCARRAY_RO(ao)) {
        PyErr_Format(PyExc_TypeError, "numpy.ndarray does not behave (C-style, memory aligned and contiguous) and cannot be wrapped as shallow blitz::Array<%s,%d>", num_to_str(type_num), N);
        blitz::Array<T,N> retval;
        return retval;
      }

      // if you survived to this point, do the real work
      blitz::TinyVector<int,N> shape;
      blitz::TinyVector<int,N> stride;
      for (int i=0; i<N; ++i) {
        shape[i] = PyArray_DIMS(ao)[i];
        stride[i] = PyArray_STRIDES(ao)[i];
      }
      return blitz::Array<T,N>(static_cast<T*>(PyArray_DATA(ao)),
          shape, stride, blitz::neverDeleteData);
  }

  /**
   * @brief Wraps the input numpy ndarray with a blitz::Array<> skin, even if
   * it has to copy the input data.
   *
   * You should use this kind of conversion when you only care about finally
   * retrieving a blitz::Array<> of the desired type and shape so as to pass it
   * as a const (read-only) input parameter to your C++ method.
   *
   * At first, we will try a shallow conversion using `shallow_blitz_array()`
   * declared above. If that does not work, then we will try a brute force
   * conversion using `PyArray_FromAny()`. This opens the possibility of, for
   * example, converting from objects that support the iteration, buffer, array
   * or memory view protocols in python.
   *
   * Notice that, in this case, the output blitz::Array may or may not be bound
   * to the input object. Because you don't know what the outcome is, it is
   * recommend you copy the output if you want to preserve it beyond the scope
   * of the conversion.
   *
   * In case of errors, a Python exception will be set. You must check it
   * properly:
   *
   * auto z = readonly_blitz_array<uint8_t,4>(obj);
   * if (PyErr_Occurred()) return 0; ///< propagate exception
   *
   * Notice that the lifetime of the blitz::Array<> extracted with this
   * procedure is bound to the lifetime of the source numpy ndarray. You'd have
   * to copy it to create an independent object.
   *
   * Als notice this procedure will copy the data twice, if the input data is
   * not already on the right format for a blitz::Array<> shallow wrap to take
   * place. This is not optimal in all conditions, namely with very large
   * read-only arrays. We hope this is not a common condition when users want
   * to convert read-only arrays.
   */
  template <typename T, int N> blitz::Array<T,N> readonly_blitz_array
    (PyObject* o) {

      blitz::Array<T,N> shallow = shallow_blitz_array<T,N>(o, false);
      if (!PyErr_Occurred()) return shallow;

      // if you get to this point, than shallow conversion did not work

      int type_num = ctype_to_num<T>();
      if (PyErr_Occurred()) {
        // propagates exception
        blitz::Array<T,N> retval;
        return retval;
      }

      PyArray_Descr* req_dtype = PyArray_DescrFromType(type_num); ///< borrowed

      // transforms the data and wraps with an auto-deletable object
      PyObject* newref = PyArray_FromAny(o, req_dtype, N, N,
#     if NPY_FEATURE_VERSION >= NUMPY16_API /* NumPy C-API version >= 1.6 */
          NPY_ARRAY_CARRAY_RO,
#     else
          NPY_CARRAY_RO,
#     endif
          0);

      if (!newref) {
        // propagates exception
        blitz::Array<T,N> retval;
        return retval;
      }

      // wrap the new array with a shallow skin
      blitz::Array<T,N> bz_shallow = shallow_blitz_array<T,N>(newref, false);
      if (!PyErr_Occurred()) {
        Py_DECREF(newref);
        blitz::Array<T,N> retval;
        return retval;
      }

      // copies and returns - do not use "copy" - this is the SECOND copy
      blitz::Array<T,N> copy(bz_shallow.shape());
      copy = bz_shallow;
      Py_DECREF(newref);
      return copy;
    }

  /**
   * @brief Copies the contents of the input blitz::Array<> into a newly
   * allocated numpy ndarray.
   *
   * The newly allocated array is a classical Pythonic **new** reference. The
   * client taking the object must call Py_XDECREF when done.
   *
   * This function returns NULL if an error has occurred, following the
   * standard python protocol.
   */
  template <typename T, int N>
    PyObject* ndarray_copy(const blitz::Array<T,N>& a) {

      int type_num = ctype_to_num<T>();
      if (PyErr_Occurred()) return 0;

      // maximum supported number of dimensions
      if (N > 11) {
        PyErr_Format(PyExc_TypeError, "input blitz::Array<%s,%d> has more dimensions than we can support (max. = 11)", num_to_str(type_num), N);
        return 0;
      }

      // copies the shape
      npy_intp shape[NPY_MAXDIMS];
      for (int i=0; i<N; ++i) shape[i] = a.extent(i);

      // creates a new array that will be returned
      PyObject* retval = PyArray_SimpleNew(N, shape, type_num);
      if (!retval) return 0; ///< propagate exception

      // creates a simple blitz::Array wrapper around the ndarray
      blitz::TinyVector<int,N> bz_shape;
      for (int i=0; i<N; ++i) bz_shape[i] = shape[i];
      PyArrayObject* aret = reinterpret_cast<PyArrayObject*>(retval);
      blitz::Array<T,N> shallow(static_cast<T*>(PyArray_DATA(aret)),
          bz_shape, blitz::neverDeleteData);

      // copies the data from the source array to the shallow blitz array
      shallow = a;

      return retval;

    }

  /**
   * Classes to help Cython integration (only supports templated classes)
   */
  template <typename T> struct CtypeToNum {

    int call() const { return ctype_to_num<T>(); }

  };

  template <typename T> struct Extract {

    T call(PyObject* o) const { return extract<T>(o); }

  };

  template <typename T, int N> struct ShallowBlitzArray {

    blitz::Array<T,N> call(PyObject* o, bool readwrite) const { 
      return shallow_blitz_array<T,N>(o);
    }

  };

  template <typename T, int N> struct ReadonlyBlitzArray {

    blitz::Array<T,N> call(PyObject* o) const { 
      return readonly_blitz_array<T,N>(o);
    }

  };

  template <typename T, int N> struct NumpyArrayCopy {

    PyObject* call(const blitz::Array<T,N>& a) const { 
      return ndarray_copy(a);
    }

  };

}}

#endif /* BOB_PYTHON_H */
