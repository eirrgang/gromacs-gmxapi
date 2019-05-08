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

"""Support execution context abstraction for gmxapi.

This module performs initialization for some global resources the first time it
is imported, such as MPI environment initialization. The base module of the
gmxapi package does not immediately import gmxapi.context, allowing the user to
control initialization of the gmxapi.context module before the first time it is
used by other gmxapi submodules.
"""

__all__ = ['get_context']

# Note: New in version 3.6: Flag, IntFlag, auto
import enum
import weakref

from gmxapi import exceptions


# TODO: this enum is an implementation detail and will be reconsidered as requirements become clearer.
# class ContextCharacteristics(enum.Enum):
#     MODULE_DEFAULT = auto()
#     IMMEDIATE_EXECUTION = auto()
#     SERIAL_ENSEMBLE = auto()
#     MPI_ENSEMBLE = auto()
#     ENSEMBLE = SERIAL_ENSEMBLE | MPI_ENSEMBLE


class __Context(object):
    """
    When a Context instance receives a request for a new handle, the instance
    can return a handle to itself or to a new instance (if the current instance
    cannot meet the requirements expressed in the request for a handle). The
    current instance may retain a reference or proxy to the child instance to
    facilitate resource reuse. A child Context may retain a weak reference to
    the parent Context to surrender or request resources or data.
    """

    def __init__(self):
        self.characteristics = ContextCharacteristics.MODULE_DEFAULT
        self.width = 1
        self.parent = None

    def clear(self):
        """Reinitialize current context."""
        self.__init__()

    def clone(self):
        new_context = __class__
        new_context.parent = weakref.proxy(self)
        return new_context

    def handle(self, flags: ContextCharacteristics = None):
        if flags is None:
            return self
        if flags & self.characteristics:
            return self
        if flags & ContextCharacteristics.IMMEDIATE_EXECUTION:
            self.characteristics |= ContextCharacteristics.IMMEDIATE_EXECUTION
            return self.handle(flags)
        if flags & ContextCharacteristics.ENSEMBLE:
            if not self.characteristics & ContextCharacteristics.ENSEMBLE:
                raise exceptions.UsageError('should have used ensemble_handle()')
            else:
                return self

    def ensemble_handle(self, options=None):
        handle = self
        if options is None:
            if not bool(self.characteristics & ContextCharacteristics.ENSEMBLE):
                raise exceptions.UsageError(
                    'Must provide ensemble options when requesting handle from non-ensemble context.')
        else:
            try:
                width = options['width']
            except:
                raise exceptions.UsageError('Must specify ensemble width when requesting ensemble handle.')
            if width != self.width:
                child = self.clone()
                child.width = width
                child.characteristics |= ContextCharacteristics.ENSEMBLE
                handle = child
        assert bool(handle.characteristics & ContextCharacteristics.ENSEMBLE)
        return handle


# Context stack.
__current_context = [__Context()]


def get_context(requirements=None):
    """Get a reference to the currently active gmxapi execution Context."""
    if len(__current_context) < 1:
        __current_context.append(__Context())
    handle = __current_context[-1]
    if requirements is not None:
        if 'width' in requirements:
            width = requirements['width']
            if width != __current_context[-1].width:
                handle = handle.ensemble_handle(options={'width': width})
    return handle
