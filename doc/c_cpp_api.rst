.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 14:59:05 2013

===============
 C and C++ API
===============

This section includes information for using the pure C or C++ API for
manipulating :py:class:`blitz.array` objects in compiled code.

C API
-----

The C API of ``blitz.array`` allows users to leverage from automatic converters
between :py:class:`numpy.ndarray` and :py:class:`blitz.array` within their own
python extensions. To use the C API, clients should first, include the header
file ``<blitz.array/capi.h>`` on their compilation units and then, make sure to
call once ``import_blitz_array()`` at their module instantiation, as explained
at the `Python manual
<http://docs.python.org/2/extending/extending.html#using-capsules>`_.

Here is a dummy C example showing how to include the header and where to call
the import function:

.. code-block:: c

   #include <blitz.array/capi.h>

   #ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
   #define PyMODINIT_FUNC void
   #endif
   PyMODINIT_FUNC initclient(void) {

     PyObject *m;
 
     m = Py_InitModule("client", ClientMethods);
     if (m == NULL) return;

     /* imports the NumPy C-API */
     import_array();

     /* imports blitz.array C-API */
     import_blitz_array();

   }

.. note::

  The include directory can be discovered using :py:func:`blitz.get_include`.

Array Structure
===============

.. c:type:: PyBlitzArrayObject

   The basic array structure represents a ``blitz.array`` instance from the
   C-side of the interpreter. You should **avoid direct access to
   the structure components** (it is presented just as an overview on the
   functionality). Instead, use the accessor methods described below.

   .. code-block:: c

      typedef struct {
        PyObject_HEAD
        void* bzarr;
        int type_num;
        Py_ssize_t ndim;
        Py_ssize_t shape[BLITZ_ARRAY_MAXDIMS];
        Py_ssize_t stride[BLITZ_ARRAY_MAXDIMS];
        int writeable;
        PyObject* base;

      } PyBlitzArrayObject;

   .. c:macro:: BLITZ_ARRAY_MAXDIMS
      
      The maximum number of dimensions supported by the current ``blitz.array``
      implementation.

   .. c:member:: void* bzarr

      This is a pointer that points to the allocated ``blitz::Array``
      structure. This pointer is cast to the proper type and number of
      dimensions when operations on the data are requested.

   .. c:member:: void* data

      A pointer to the data entry in the ``blitz::Array<>``. This is equivalent
      to the operation ``blitz::Array<>::data()``.

   .. c:member:: int type_num

      The numpy type number that is compatible with the elements of this
      array. It is a C representation of the C++ template parameter ``T``. Only
      some types are current supported, namely:

      =============================== ==================== ==================
         C/C++ type                      Numpy Enum            Notes
      =============================== ==================== ==================
       ``bool``                        ``NPY_BOOL``
       ``uint8_t``                     ``NPY_UINT8``
       ``uint16_t``                    ``NPY_UINT16``
       ``uint32_t``                    ``NPY_UINT32``
       ``uint64_t``                    ``NPY_UINT64``
       ``int8_t``                      ``NPY_INT8``
       ``int16_t``                     ``NPY_INT16``
       ``int32_t``                     ``NPY_INT32``
       ``int64_t``                     ``NPY_INT64``
       ``float``                       ``NPY_FLOAT32``
       ``double``                      ``NPY_FLOAT64``
       ``long double``                 ``NPY_FLOAT128``     Plat. Dependent
       ``std::complex<float>``         ``NPY_COMPLEX64``
       ``std::complex<double>``        ``NPY_COMPLEX128``
       ``std::complex<long double>``   ``NPY_COMPLEX256``   Plat. Dependent
      =============================== ==================== ==================

   .. c:member:: Py_ssize_t ndim

      The rank of the ``blitz::Array<>`` allocated on ``bzarr``.

   .. c:member:: Py_ssize_t shape[BLITZ_ARRAY_MAXDIMS]

      The shape of the ``blitz::Array<>`` allocated on ``bzarr``, in number of
      **elements** in each dimension.

   .. c:member:: Py_ssize_t stride[BLITZ_ARRAY_MAXDIMS]

      The strides of the ``blitz::Array<>`` allocated on ``bzarr``, in number
      of **bytes** to jump to read the next element in each dimensions.

   .. c:member:: int writeable

      Assumes the value of ``1`` (true), if the data is read-write. ``0`` is
      set otherwise.

   .. c:member:: PyObject* base

      If the memory pointed by the currently allocated ``blitz::Array<>``
      belongs to another Python object, the object is ``Py_INCREF()``'ed and a
      pointer is kept on this structure member.
   

Basic Properties and Checking
=============================

.. c:function:: int PyBlitzArray_Check(PyObject* o)

   Checks if the input object ``o`` is a ``PyBlitzArrayObject``. Returns ``1``
   if it is, and ``0`` otherwise.


.. c:function:: int PyBlitzArray_CheckNumpyBase(PyArrayObject* o)

   Checks if the input object ``o`` is a ``PyArrayObject`` (i.e. a
   :py:class:`numpy.ndarray`), if so, checks if the base of the object is set
   and that it corresponds to the current ``PyArrayObject`` shape and stride
   settings. If so, returns ``1``. It returns ``0`` otherwise.


.. c:function:: int PyBlitzArray_TYPE (PyBlitzArrayObject* o)

   Returns integral type number (as defined by the Numpy C-API) of elements
   in this blitz::Array<>. This is the formal method to query for
   ``o->type_num``.
   

.. c:function:: PyArray_Descr* PyBlitzArray_PyDTYPE (PyBlitzArrayObject* o)

   Returns a **new reference** to a numpy C-API ``PyArray_Descr*`` equivalent
   to the internal type element T.


.. c:function:: Py_ssize_t PyBlitzArray_NDIM (PyBlitzArrayObject* o)

   Returns the number of dimensions in a given ``blitz.array``. This is the
   formal way to check for ``o->ndim``.


.. c:function:: Py_ssize_t* PyBlitzArray_SHAPE (PyBlitzArrayObject* o)

   Returns the C-stype shape for this blitz::Array<>. This is the formal method
   to query for ``o->shape``. The shape represents the number of elements in
   each dimension of the array.
   

.. c:function:: PyObject* PyBlitzArray_PySHAPE (PyBlitzArrayObject* o)

   Returns a **new reference** to a Python tuple holding a copy of the shape
   for the given array. The shape represents the number of elements in each
   dimension of the array.
   

.. c:function:: Py_ssize_t* PyBlitzArray_STRIDE (PyBlitzArrayObject* o)

   Returns the C-stype stride for this blitz::Array<>. This is the formal
   method to query for ``o->stride``. The strides in this object are
   represented in number of bytes and **not** in number of elements considering
   its ``type_num``. This is compatible with the :py:class:`numpy.ndarray`
   strategy.
   

.. c:function:: PyObject* PyBlitzArray_PySTRIDE (PyBlitzArrayObject* o)

   Returns a **new reference** to a Python tuple holding a copy of the strides
   for the given array. The strides in this object are represented in number of
   bytes and **not** in number of elements considering its ``type_num``. This
   is compatible with the :py:class:`numpy.ndarray` strategy.
   

.. c:function:: int PyBlitzArray_WRITEABLE (PyBlitzArrayObject* o)

   Returns ``1`` if the object is writeable, ``0`` otherwise. This is the
   formal way to check for ``o->writeable``.


.. c:function:: PyObject* PyBlitzArray_PyWRITEABLE (PyBlitzArrayObject* o)

   Returns ``True`` if the object is writeable, ``False`` otherwise.


.. c:function:: PyObject* PyBlitzArray_BASE (PyBlitzArrayObject* o)

   Returns a **borrowed reference** to the base of this object. The return
   value of this function may be ``NULL``.


.. c:function:: PyObject* PyBlitzArray_PyBASE (PyBlitzArrayObject* o)

   Returns a **new reference** to the base of this object. If the internal
   ``o->base`` is ``NULL``, then returns ``Py_None``. Use this when interfacing
   with the Python interpreter.
  

Indexing
========

.. c:function:: PyObject* PyBlitzArray_GetItem (PyBlitzArrayObject* o, Py_ssize_t* pos)

   Returns, as a PyObject, an item from the array. This will be a copy of the
   internal item. If you set it, it won't set the original array.  ``o`` should
   be the PyBlitzArrayObject to be queried. ``pos`` should be a C-style array
   indicating the precise position to fetch. It is considered to have the same
   number of entries as the current array shape.
   

.. c:function:: int PyBlitzArray_SetItem (PyBlitzArrayObject* o, Py_ssize_t* pos, PyObject* value)

   Sets an given position on the array using any Python or numpy scalar. ``o``
   should be the PyBlitzArrayObject to be set. ``pos`` should be a C-style
   array indicating the precise position to set and ``value``, the Python
   or numpy scalar to set the value to.


Construction and Destruction
============================

.. c:function:: PyObject* PyBlitzArray_New (PyTypeObject* type, PyObject *args, PyObject* kwds)

   Allocates memory and pre-initializes a ``PyBlitzArrayObject*`` object. This
   is the base allocator - seldomly used in user code.
   

.. c:function:: void PyBlitzArray_Delete (PyBlitzArrayObject* o)

   Completely deletes a ``PyBlitzArrayObject*`` and associated memory areas.
   This is the base deallocator - seldomly used in user code.
   

.. c:function:: PyObject* PyBlitzArray_SimpleNew (int typenum, Py_ssize_t ndim, Py_ssize_t* shape)

   Allocates a new ``blitz.array`` with a given (supported) type and return it
   as a python object. ``typenum`` should be set to the numpy type number of
   the array type (e.g. ``NPY_FLOAT64``). ``ndim`` should be set to the total
   number of dimensions the array should have. ``shape`` should be set to the
   array shape.

   
.. c:function:: PyObject* PyBlitzArray_SimpleNewFromData (int type_num, Py_ssize_t ndim, Py_ssize_t* shape, Py_ssize_t* stride, void* data, int writeable)

   Allocates a new ``blitz.array`` with a given (supported) type and return it
   as a python object. ``typenum`` should be set to the numpy type number of
   the array type (e.g. ``NPY_FLOAT64``). ``ndim`` should be set to the total
   number of dimensions the array should have. ``shape`` should be set to the
   array shape. ``stride`` should be set to the array stride in the numpy style
   (in number of bits). ``data`` should be a pointer to the begin of the data
   area. ``writeable`` indicates if the resulting array should be writeble (set
   it to ``1``), or read-only (set it to ``0``).

   
To/From Numpy Converters
========================

.. c:function:: PyObject* PyBlitzArray_AsNumpyArray (PyBlitzArrayObject* o)

   Creates a **shallow** copy of the given ``blitz.array`` as a
   ``numpy.ndarray``.
    

.. c:function:: PyObject* PyBlitzArray_FromNumpyArray (PyObject* o)

   Creates a new ``blitz.array`` from a ``numpy.ndarray`` object in a shallow
   manner.


Converter Functions for PyArg_Parse* family
===========================================

.. c:function:: int PyBlitzArray_Converter(PyObject* o, PyBlitzArrayObject** a) 

   This function is meant to be used with :c:func:`PyArg_ParseTupleAndKeywords`
   family of functions in the Python C-API. It converts an arbitrary input
   object into a ``PyBlitzArrayObject`` that can be used as input into another
   function.

   You should use this converter when you don't need to write-back into the
   input array. As any other standard Python converter, it returns a **new**
   reference to a ``PyBlitzArrayObject``.

   It works efficiently if the input array is already a ``PyBlitzArrayObject``
   or if it is a ``PyArrayObject`` (i.e., a :py:class:``numpy.ndarray``), with
   a matching base which is a ``PyBlitzArrayObject``. Otherwise, it creates a
   new ``PyBlitzArrayObject`` by first creating a ``PyArrayObject`` and then
   shallow wrapping it with a ``PyBlitzArrayObject``.

.. c:function:: int PyBlitzArray_OutputConverter(PyObject* o, PyBlitzArrayObject** a)

   This function is meant to be used with :c:func:`PyArg_ParseTupleAndKeywords`
   family of functions in the Python C-API. It converts an arbitrary input
   object into a ``PyBlitzArrayObject`` that can be used as input/output or
   output into another function.

   You should use this converter when you need to write-back into the input
   array. The input type should be promptly convertible to a
   :py:class:`numpy.ndarray` as with :c:func:`PyArray_OutputConverter`. As any
   other standard Python converter, it returns a **new** reference to a
   ``PyBlitzArrayObject*``.

.. c:function:: int PyBlitzArray_IndexConverter (PyObject* o, PyBlitzArrayObject** shape)

   Converts any compatible sequence into a C-array containing the shape
   information. The shape information and number of dimensions is stored on
   the previously allocated ``PyBlitzArrayObject*`` you should provide. This
   method is supposed to be used with ``PyArg_ParseTupleAndKeywords`` and
   derivatives.

   Parameters are:
   
   ``o``
     The input object to be converted into a C-shape

   ``shape``
     A preallocated (double) address for storing the shape value, on successful
     conversion
   
   Returns 0 if an error is detected, 1 on success.


.. c:function:: int PyBlitzArray_TypenumConverter (PyObject* o, int** type_num)

   Converts any compatible sequence into a Numpy integer type number. This
   method is supposed to be used with ``PyArg_ParseTupleAndKeywords`` and
   derivatives.

   Parameters are:
   
   ``o``
     The input object to be converted into a C-shape

   ``type_num``
      A preallocated (double) address for storing the type on successful
      conversion.
   
   Returns 0 if an error is detected, 1 on success.

  
Other Utilities
===============

.. c:function:: const char* PyBlitzArray_TypenumAsString (int typenum)

   Converts from numpy type_num to a string representation


C++ API
-------

The C++ API consists mostly of templated methods for manipulating the C++ type
``blitz::Array<>`` so as to convert ``PyObject*``'s from and to objects of that
type. To use the C++ API you must include the header file
``<blitz.array/cppapi.h>`` and ``import_blitz_array()`` on your module, as
explained on the C-API section of this document.

Basic Properties and Checking
=============================

.. cpp:function:: int PyBlitzArrayCxx_IsBehaved<T,N>(blitz::Array<T,N>& a)

   Tells if a ``blitz::Array<>`` is memory contiguous and C-style.


Construction and Destruction
============================

.. cpp:function:: PyObject* PyBlitzArrayCxx_NewFromConstArray<T,N>(const blitz::Array<T,N>& a)

   Builds a new read-only ``PyBlitzArrayObject`` from an existing Blitz++
   array, without copying the data. Returns a new reference.


.. cpp:function:: PyObject* PyBlitzArrayCxx_NewFromArray<T,N>(blitz::Array<T,N>& a)

   Builds a new writeable ``PyBlitzArrayObject`` from an existing Blitz++
   array, without copying the data. Returns a new reference.


Other Utilities
===============

.. cpp:function:: blitz::Array<T,N>* PyBlitzArrayCxx_AsBlitz(PyBlitzArrayObject* o)

   Casts a ``PyBlitzArrayObject`` to a specific ``blitz::Array<>`` type. Notice
   this is a brute-force cast. You are responsible for checking if that it is
   correct.

.. cpp:function:: int PyBlitzArrayCxx_CToTypenum<T>()

   Converts from C/C++ type to ndarray type_num.
   
   We cover only simple conversions (i.e., standard integers, floats and
   complex numbers only). If the input type is not convertible, an exception
   is set on the Python error stack. You must check ``PyErr_Occurred()`` after
   a call to this function to make sure things are OK and act accordingly.  For
   example:

   .. code-block:: c++
   
      int typenum = PyBlitzArrayCxx_CToTypenum<my_weird_type>(obj);
      if (PyErr_Occurred()) return 0; ///< propagate exception


.. cpp:function:: T PyBlitzArrayCxx_AsCScalar<T>(PyObject* o)

   Extraction API for **simple** types.
   
   We cover only simple conversions (i.e., standard integers, floats and
   complex numbers only). If the input object is not convertible to the given
   type, an exception is set on the Python error stack. You must check
   ``PyErr_Occurred()`` after a call to this function to make sure things are OK
   and act accordingly. For example:

   .. code-block:: c++
   
      auto z = extract<uint8_t>(obj);
      if (PyErr_Occurred()) return 0; ///< propagate exception
 
.. cpp:function:: PyBlitzArrayCxx_FromCScalar<T>(T v)

   Converts **simple** C types into numpy scalars

   We cover only simple conversions (i.e., standard integers, floats and
   complex numbers only). If the input object is not convertible to the given
   type, an exception is set on the Python error stack and ``0`` (``NULL``) is
   returned. 
