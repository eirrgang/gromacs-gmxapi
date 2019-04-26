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
            'datamodel',
           'exceptions',
           'logger',
           'operation',
           'read_tpr']

import os

from ._logging import logger
from . import datamodel
from . import exceptions
from . import operation
from .commandline import commandline_operation
from .context import ContextCharacteristics as ContextCharacteristics
from .context import get_context
from .datamodel import ndarray
from .operation import read_tpr
from . import _gmxapi


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
