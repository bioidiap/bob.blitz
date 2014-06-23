.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Thu 29 Aug 2013 16:07:57 CEST

.. image:: https://travis-ci.org/bioidiap/bob.blitz.svg?branch=master
   :target: https://travis-ci.org/bioidiap/bob.blitz
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.blitz/master/index.html
.. image:: https://coveralls.io/repos/bioidiap/bob.blitz/badge.png
   :target: https://coveralls.io/r/bioidiap/bob.blitz
.. image:: http://img.shields.io/github/tag/bioidiap/bob.blitz.png
   :target: https://github.com/bioidiap/bob.blitz
.. image:: http://img.shields.io/pypi/v/bob.blitz.png
   :target: https://pypi.python.org/pypi/bob.blitz
.. image:: http://img.shields.io/pypi/dm/bob.blitz.png
   :target: https://pypi.python.org/pypi/bob.blitz

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

The latest version of the documentation can be found `here <https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.blitz/master/index.html>`_.

Otherwise, you can generate the documentation for this package yourself, after installation, using Sphinx::

  $ sphinx-build -b html doc sphinx

This shall place in the directory ``sphinx``, the current version for the
documentation of the package.

Testing
-------

You can run a set of tests using the nose test runner::

  $ nosetests -sv

You can run our documentation tests using sphinx itself::

  $ sphinx-build -b doctest doc sphinx

You can test overall test coverage with::

  $ nosetests --with-coverage --cover-package=bob.blitz

The ``coverage`` egg must be installed for this to work properly.

Development
-----------

To develop this package, install using ``zc.buildout``, using the buildout
configuration found on the root of the package::

  $ python bootstrap.py
  ...
  $ ./bin/buildout

Tweak the options in ``buildout.cfg`` to disable/enable verbosity and debug
builds.
