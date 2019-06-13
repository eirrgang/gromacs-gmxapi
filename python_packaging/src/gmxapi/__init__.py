#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2019, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.

"""gmxapi Python package for GROMACS.

This package provides Python access to GROMACS molecular simulation tools.
Operations can be connected flexibly to allow high performance simulation and
analysis with complex control and data flows. Users can define new operations
in C++ or Python with the same tool kit used to implement this package.

Simulation Operations
---------------------

* mdrun()
* modify_input()
* read_tpr()

Data flow operations
--------------------

* logical_and()
* logical_or()
* logical_not()
* reduce()
* scatter()
* gather()
* subgraph()
* while_loop()

Extension
---------

* commandline_wrapper()
* make_operation()
* function_wrapper()

Data
----

Basic Data Types
~~~~~~~~~~~~~~~~

* Integer
* Float
* Boolean

Containers
~~~~~~~~~~

* NDArray
* String
* AssociativeArray

Proxies
-------

* File()
* Future()
* Handle()

"""

__all__ = ['commandline_operation',
           'concatenate_lists',
           'function_wrapper',
           'join_arrays',
           'logger',
           'make_constant',
           'make_operation',
           'mdrun',
           'ndarray',
           '__version__']

import collections
import typing

from ._logging import logger

from .datamodel import ndarray, NDArray
from .fileio import read_tpr, write_tpr_file
from .operation import InputCollectionDescription
from .operation import concatenate_lists, computed_result, function_wrapper, join_arrays, make_constant, make_operation, mdrun
from .commandline import commandline_operation
# TODO: decide where this lives
from .operation import subgraph
# TODO: decide where this lives
from .operation import while_loop
from .version import __version__
