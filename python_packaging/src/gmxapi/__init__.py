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
from gmxapi import _logging

__all__ = ['commandline_operation',
           'concatenate_lists',
           'context',
           'exceptions',
           'function_wrapper',
           'logger',
           'make_operation',
           'mdrun',
           'ndarray',
           'operation',
           'read_tpr',
           'version',
           'workflow',
           '__version__']

import collections
import os
from typing import TypeVar

import gmxapi._logging
from . import version
from ._logging import logger

from . import datamodel
from . import context
from . import datamodel
from . import exceptions
from . import workflow
from . import fileio
from . fileio import *


from .operation import computed_result, function_wrapper, make_operation
from .commandline import commandline_operation
from .datamodel import ndarray, NDArray
from .version import __version__

#
# class SimulationOperation(object):
#     pass


def mdrun(input=None):
    """MD simulation operation.

    Arguments:
        input : valid simulation input

    Returns:
        runnable operation to perform the specified simulation

    The returned object has a `run()` method to launch the simulation.
    Otherwise, this operation does not yet support the gmxapi data flow model.

    `input` may be a TPR file name.

    Note:
        New function names will be appearing to handle tasks that are separate

        "simulate" is plausibly a dispatcher or base class for various tasks
        dispatched by mdrun. Specific work factories are likely "minimize,"
        "test_particle_insertion," "legacy_simulation" (do_md), or "simulation"
        composition (which may be leap-frog, vv, and other algorithms)
    """
    return workflow.from_tpr(input)


# TODO: fix make_operation
# read_tpr = make_operation(fileio._SimulationInput, output={'parameters': dict})
@function_wrapper(output={'parameters': dict, 'topology': str, 'coordinates': str, 'simulation_state': str})
def read_tpr(filename: str = '', output=None):
    """Get simulation input sources from a TPR file.

    Outputs:
        parameters : MDP simulation parameters
        coordinates : atom (or CG particle) coordinates (not yet implemented)
        simulation_state : simulation internal state (checkpoint data) (not yet implemented)
        topology : molecular force field data (not yet implemented)
    """
    sim_input = fileio.read_tpr(filename)
    output._internal = sim_input
    output.parameters = sim_input.parameters.extract()
    output.topology = filename
    output.coordinates = filename
    output.simulation_state = filename


# @function_wrapper(output={'parameters': dict})
# def modify_input(parameters: dict = None, output=None):
#     output.parameters = parameters


@computed_result
def join_arrays(*, front: NDArray = (), back: NDArray = ()) -> NDArray:
    """Operation that consumes two sequences and produces a concatenated single sequence.

    Note that the exact signature of the operation is not determined until this
    helper is called. Helper functions may dispatch to factories for different
    operations based on the inputs. In this case, the dtype and shape of the
    inputs determines dtype and shape of the output. An operation instance must
    have strongly typed output, but the input must be strongly typed on an
    object definition so that a Context can make runtime decisions about
    dispatching work and data before instantiating.
    # TODO: elaborate and clarify.
    # TODO: check type and shape.
    # TODO: figure out a better annotation.
    """
    # TODO: (FR4) Returned list should be an NDArray.
    if isinstance(front, (str, bytes)) or isinstance(back, (str, bytes)):
        raise exceptions.ValueError('Input must be a pair of lists.')
    assert isinstance(front, NDArray)
    assert isinstance(back, NDArray)
    new_list = list(front._values)
    new_list.extend(back._values)
    return new_list


Scalar = TypeVar('Scalar')


def concatenate_lists(sublists: list = ()):
    """Combine data sources into a single list.

    A trivial data flow restructuring operation.
    """
    if isinstance(sublists, (str, bytes)):
        raise exceptions.ValueError('Input must be a list of lists.')
    if len(sublists) == 0:
        return ndarray([])
    else:
        return join_arrays(front=sublists[0], back=concatenate_lists(sublists[1:]))


def make_constant(value: Scalar):
    """Provide a predetermined value at run time.

    This is a trivial operation that provides a (typed) value, primarily for
    internally use to manage gmxapi data flow.

    Accepts a value of any type. The object returned has a definite type and
    provides same interface as other gmxapi outputs. Additional constraints or
    guarantees on data type may appear in future versions.
    """
    dtype = type(value)
    source = operation.StaticSourceManager(name='data', proxied_data=value, width=1, function=lambda x: x)
    description = datamodel.ResultDescription(dtype=dtype, width=1)
    future = operation.Future(source, 'data', description=description)
    return future


def scatter(array: NDArray) -> datamodel.EnsembleDataSource:
    """Convert array data to parallel data.

    Given data with shape (M,N), produce M parallel data sources of shape (N,).

    The intention is to produce ensemble data flows from NDArray sources.
    Currently, we only support zero and one dimensional data edge cross-sections.
    In the future, it may be clearer if `scatter()` always converts a non-ensemble
    dimension to an ensemble dimension or creates an error, but right now there
    are cases where it is best just to raise a warning.

    If provided data is a string, mapping, or scalar, there is no dimension to
    scatter from, and DataShapeError is raised.
    """
    if isinstance(array, operation.Future):
        # scatter if possible
        width = array.description.width
        if width > 1:
            return datamodel.EnsembleDataSource(source=array, width=width)
            # Recipient will need to call `result()`.
        else:
            raise exceptions.ValueError('No dimension to scatter from.')
    if isinstance(array, datamodel.EnsembleDataSource):
        # scatter if possible
        if array.width > 1:
            raise exceptions.DataShapeError('Cannot scatter. Only 1-D ensemble data is supported.')
        array = array.source
        if isinstance(array, operation.Future):
            return scatter(array)
        elif isinstance(array, NDArray):
            # Get the first meaningful scattering dimension
            width = 0
            source = array[:]
            for scatter_dimension, width in enumerate(array.shape):
                if width > 1:
                    break
                else:
                    # Strip unused outer dimensions.
                    source = source[0]
            if width > 1:
                return datamodel.EnsembleDataSource(source=source, width=width)
            else:
                raise exceptions.ValueError('No dimension to scatter from.')
    if isinstance(array, (str, bytes)):
        raise exceptions.DataShapeError(
            'Strings are not treated as sequences of characters to automatically scatter from.')
    if isinstance(array, collections.abc.Iterable):
        # scatter
        array = ndarray(array)
        return scatter(array)


def gather(data: datamodel.EnsembleDataSource):
    """Combines parallel data to an NDArray source.

    If the data source has an ensemble shape of (1,), result is an NDArray of
    length 1 if for a scalar data source. For a NDArray data source, the
    dimensionality of the NDArray is not increased, the original NDArray is
    produced, and gather() is a no-op.

    This may change in future versions so that gather always converts an
    ensemble dimension to an array dimension.
    """
    # TODO: Could be used as part of the clean up for join_arrays to convert a scalar Future to a 1-D list.
    # Note: Clearly, the implementation of gather() is an implementation detail of the execution Context.
    if hasattr(data, 'width'):
        if data.width == 1:
            # TODO: Do we want to allow this silent no-op?
            if isinstance(data.source, NDArray):
                return data.source
            elif isinstance(data.source, operation.Future):
                raise exceptions.ApiError('gather() not implemented for Future')
            else:
                raise exceptions.UsageError('Nothing to gather.')

        assert data.width > 1
        if isinstance(data.source, operation.Future):
            manager = operation.ProxyResourceManager(proxied_future=data.source, width=1, function=ndarray)
        else:
            if isinstance(data.source, NDArray):
                raise exceptions.ValueError('higher-dimensional NDArrays not yet implemented.')
            manager = operation.StaticSourceManager(proxied_data=data.source, width=1, function=ndarray)
        description = datamodel.ResultDescription(dtype=NDArray, width=1)
        future = operation.Future(resource_manager=manager, name=data.source.name, description=description)
        return future
    else:
        raise exceptions.TypeError('Expected data with "width".')


def logical_not(value: bool):
    """Boolean negation.

    If the argument is a gmxapi compatible Data or Future object, a new View or
    Future is created that proxies the boolean opposite of the input.

    If the argument is a callable, logical_not returns a wrapper function that
    returns a Future for the logical opposite of the callable's result.
    """
    # TODO: Small data transformations like this don't need to be formal Operations.
    # This could be essentially a data annotation that affects the resolver in a
    # DataEdge. As an API detail, coding for different Contexts and optimizations
    # within those Context implementations could be simplified.
    operation = function_wrapper(output={'data': bool})(lambda data=bool(): not bool(data))
    return operation(data=value).output.data


# TODO: decide where this lives
from .operation import subgraph

# TODO: decide where this lives
from .operation import while_loop


def File(suffix=''):
    """Placeholder for input/output files.

    Arguments:
        suffix: string to be appended to actual file name.

    Note:
        Some programs have logic influenced by aspects of the text in a file
        argument. The ``suffix`` key word parameter allows the proxied file's
        actual name to be constrained when passed as an argument to a program
        expecting a particular suffix.
    """
    assert not version.has_feature('fr21')
