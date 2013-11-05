.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Thu 29 Aug 2013 16:07:57 CEST

====================================
 Python bindings for Blitz++ Arrays
====================================

This package contains a set of Pythonic bindings to the popular Blitz/C++
library. It also provides a C/C++ API that allows your Python extensions to
leverage from the interfaces provided by this package. For more information,
consult the package documentation.

Installation
------------

Install it through normal means, via PyPI or use ``zc.buildout`` to bootstrap
the package and run test units.

Documentation
-------------

You can generate the documentation for this package, after installation, using
Sphinx::

  $ ./bin/sphinx-build -b html doc sphinx

This shall place in the directory ``sphinx``, the current version for the
documentation of the package.

Testing
-------

You can run a set of tests using the nose test runner::

  $ ./bin/nosetests -sv

You can run our documentation tests using sphinx itself::

  $ ./bin/sphinx-build -b doctest doc sphinx

