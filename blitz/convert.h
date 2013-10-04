/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 16 Sep 16:21:47 2013 
 *
 * @brief Wraps bob::core::array convert functionality into structures to ease
 * the migration to Cython.
 */

#include <Python.h>

/**
 * @brief Wraps our conversion function so that it accepts simple PyObject* as
 * inputs and produces a PyObject* as output.
 *
 * We assume the input is checked and properly converted at the Cythonic
 * binding, so that:
 *
 * @param array Is a PyArrayObject*, well-behaved (contiguous, C-order,
 *              aligned) and readable at least.
 *
 * @param dtype Is PyArrayDescr*, pointing to one of the well supported
 *              destination data types.
 *
 * @param dst_range Is a 2-entry iterable that contains the minimum and the
 *                  maximum of the destination array. The values therein should
 *                  be castable to the destination array element type. This
 *                  value can be Py_NONE, in which case the destination range
 *                  is calculated using the maximum and the minimum of the
 *                  destination element type.
 *
 * @param src_range Is a 2-entry iterable that contains the minimum and the
 *                  maximum of the source array. The values therein should
 *                  be castable to the source array element type. This
 *                  value can be Py_NONE, in which case the source range
 *                  is calculated using the maximum and the minimum of the
 *                  source element type.
 *
 * @return a new reference to a PyArrayObject* in the required dtype and
 * re-scaled so that it respects the destination range, taking as start the
 * input array and the source range.
 */
PyObject* convert (PyObject* array, PyObject* dtype, 
    PyObject* dst_range, PyObject* src_range);
