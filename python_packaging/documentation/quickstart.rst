===========
Quick start
===========

From ``python_packaging/README.md``:

The ``src`` directory provides the files that will be copied to the GROMACS installation location from which users may
install Python packages.
This allows C++ extension modules to be built against a user-chosen GROMACS installation,
but for a Python interpreter that is very likely different from that used
by the system administrator who installed GROMACS.

To build and install the Python package,
first install GROMACS to ``/path/to/gromacs``.
Then, install the package in a Python virtualenv.
::

    source /path/to/gromacs/bin/GMXRC
    python3 -m venv $HOME/somevirtualenv
    source $HOME/somevirtualenv/bin/activate
    (cd src && pip install -r requirements.txt && pip install .)
    python -c 'import gmxapi as gmx'

Use ``pytest`` to run unit tests and integration tests.
::

    pip install -r requirements-test.txt
    pytest src/test
    pytest test

Build these docs::

    pip install -r requirements-docs.txt
    sphinx-build -b html documentation html
    open html/index.html

For additional discussion on packaging and distribution, see
https://redmine.gromacs.org/issues/2896
