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

"""gmxapi Python package for GROMACS."""

__all__ = ['commandline_operation',
           'concatenate_lists',
           'exceptions',
           'function_wrapper',
           'logger',
           'mdrun',
           'ndarray',
           'operation',
           'read_tpr']

import collections
import os
from typing import TypeVar

from . import _gmxapi
from . import datamodel
# from .context import ContextCharacteristics as ContextCharacteristics
# from .context import get_context
from . import datamodel
from . import exceptions
from ._logging import logger
from .commandline import commandline_operation
from .datamodel import ndarray, NDArray
from .operation import computed_result, function_wrapper


def mdrun(input=None):
    """MD simulation operation.

    Arguments:
        input : valid simulation input

    Returns:
        runnable operation to perform the specified simulation

    The returned object has a `run()` method to launch the simulation.
    Otherwise, this operation does not yet support the gmxapi data flow model.

    `input` may be a TPR file name.
    """
    try:
        filename = os.path.abspath(input)
    except Exception as E:
        raise exceptions.ValueError('input must be a valid file name.') from E
    try:
        system = _gmxapi.from_tpr(filename)
        context = _gmxapi.Context()
        md = system.launch(context)
    except Exception as e:
        raise exceptions.ApiError('Unhandled error from library: {}'.format(e)) from e
    return md


@computed_result
def join_arrays(a: NDArray = (), b: NDArray = ()) -> NDArray:
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
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        raise exceptions.ValueError('Input must be a pair of lists.')
    assert isinstance(a, NDArray)
    assert isinstance(b, NDArray)
    new_list = list(a._values)
    new_list.extend(b._values)
    return new_list


Scalar = TypeVar('Scalar')


def concatenate_lists(sublists: list = ()):
    """Combine data sources into a single list.

    A trivial data flow restructuring operation
    """
    if isinstance(sublists, (str, bytes)):
        raise exceptions.ValueError('Input must be a list of lists.')
    if len(sublists) == 0:
        return ndarray([])
    else:
        return join_arrays(a=sublists[0], b=concatenate_lists(sublists[1:]))


def make_constant(value: Scalar):
    """Provide a predetermined value at run time.

    This is a trivial operation that provides a (typed) value, primarily for
    internally use to manage gmxapi data flow.

    Accepts a value of any type. The object returned has a definite type and
    provides same interface as other gmxapi outputs. Additional constraints or
    guarantees on data type may appear in future versions.
    """
    # TODO: (FR4+) Manage type compatibility with gmxapi data interfaces.
    scalar_type = type(value)
    assert not isinstance(scalar_type, type(None))
    operation = function_wrapper(output={'data': scalar_type})(lambda data=scalar_type(): data)
    return operation(data=value).output.data


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
        pass
    if isinstance(array, datamodel.EnsembleDataSource):
        # scatter if possible
        pass
    if isinstance(array, (str, bytes)):
        raise exceptions.DataShapeError(
            'Strings are not treated as sequences of characters to automatically scatter from.')
    if isinstance(array, collections.abc.Iterable):
        # scatter
        pass
    return


def gather(data: datamodel.EnsembleDataSource) -> NDArray:
    """Combines parallel data to an NDArray source.

    If the data source has an ensemble shape of (1,), result is an NDArray of
    length 1 if for a scalar data source. For a NDArray data source, the
    dimensionality of the NDArray is not increased, the original NDArray is
    produced, and gather() is a no-op.

    This may change in future versions so that gather always converts an
    ensemble dimension to an array dimension.
    """
    # TODO: Could be used as part of the clean up for join_arrays to convert a scalar Future to a 1-D list.
    # Note: gather() is implemented in terms of details of the execution Context.


def logical_and():
    pass


@function_wrapper()
def read_tpr(tprfile: str = '') -> str:
    """Prepare simulation input pack from a TPR file."""
    return tprfile
