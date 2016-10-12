.. vim: set fileencoding=utf-8 :
.. Mon 08 Aug 2016 10:52:47 CEST

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.blitz/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bob/bob.blitz/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.blitz/badges/v2.0.11/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.blitz/commits/v2.0.11
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.blitz
.. image:: http://img.shields.io/pypi/v/bob.blitz.png
   :target: https://pypi.python.org/pypi/bob.blitz
.. image:: http://img.shields.io/pypi/dm/bob.blitz.png
   :target: https://pypi.python.org/pypi/bob.blitz


====================================
 Python bindings for Blitz++ Arrays
====================================

This package is part of the signal-processing and machine learning toolbox
Bob_. It provides a bridge between our C++ array infrastructure (based on
Blitz++) and NumPy arrays. Almost all of our Python C/C++ extensions use this
package to transparently and efficiently convert NumPy arrays to Blitz++ arrays
and vice-versa.


Installation
------------

Follow our `installation`_ instructions. Then, using the Python interpreter
inside that distribution, bootstrap and buildout this package::

  $ python bootstrap-buildout.py
  $ ./bin/buildout


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://gitlab.idiap.ch/bob/bob/wikis/Installation
.. _mailing list: https://groups.google.com/forum/?fromgroups#!forum/bob-devel
