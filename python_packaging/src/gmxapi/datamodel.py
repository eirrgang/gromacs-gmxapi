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

__all__ = ['ndarray']

import abc
import typing

from gmxapi import exceptions

# class Type(abc.ABC):
#     """gmxapi data type base class"""
#
#     @property
#     @abc.abstractmethod
#     def value(self):
#         return self._value
#
#     def __init__(self, value):
#         self._value = value
Type = object


class Boolean(Type):
    """gmxapi logical value type."""


class String(Type):
    """gmxapi string data type."""


class Integer64(Type):
    """gmxapi 64-bit integer data type."""


class Float64(Type):
    """gmxapi 64-bit floating point data type."""


class NDArray(Type):
    """N-Dimensional array type.
    """
    def __init__(self):
        self.shape = ()
        self.dtype = type(None)
        self.values = []

    def __len__(self):
        return len(self.values)


from typing import NamedTuple
class Map(NamedTuple):
    """Associative data structure providing a key-value map.

    Instances more closely resemble Python `collections.namedtuple` objects than
    native Python `dict` objects. However, fields are typed, and objects must
    be initialized with appropriately typed values or Futures for each field.
    """


class BoundInput(object):
    pass


class OutputCollection(object):
    pass


class InputCollection(object):
    """A collection of named data inputs for an operation.

    An instance of InputCollection allows type-checking when binding data
    sources to an operation as operation is initialized. An operation type has
    a well-defined InputCollection. An operation instance must have bound data
    sources of valid type.

    TODO: We can use a parameterized factory (decorator factory function) to
     dynamically define an InputCollection class for a signature.
    """
    def __init__(self, signature):
        parameters = []
        for name, parameter in signature.parameters.items():
            if parameter.annotation is parameter.empty:
                if parameter.default is parameter.empty:
                    raise exceptions.ProtocolError('Cannot determine type for input {}.'.format(name))
                else:
                    parameters.append(parameter.replace(annotation=type(parameter.default)))
            else:
                #This can't work for things like "make_constant" that don't know the type until they are instantiated...
                # if not isinstance(parameter.annotation, type):
                #     raise exceptions.ProtocolError('gmxapi parameters must have type annotations.')
                parameters.append(parameter)
        for i, parameter in enumerate(parameters):
            if not isinstance(parameter.annotation, (str, bytes)):
                if isinstance(parameter.annotation, type) and issubclass(parameter.annotation, typing.Iterable):
                    parameters[i] = parameter.replace(annotation=NDArray)
        self.signature = signature
        self.parameters = parameters
        self._input_pack = None

    def bind(self, *args, **kwargs):
        """Bind the provided positional and keyword arguments.
        """
        # Check the provided value
        input_pack = {}

        def arg_iterator():
            for i, arg in enumerate(args):
                yield (self.parameters[i].name, arg)
            for key, value in kwargs.items():
                yield (key, value)
        i = 0
        for name, arg in arg_iterator():
            param = self.parameters[i]
            assert name == param.name
            i += 1

            ptype = param.annotation
            if ptype == NDArray:
                if isinstance(arg, NDArray):
                    input_pack[name] = arg
                elif isinstance(arg, typing.Iterable) and not isinstance(arg, (str, bytes)):
                    input_pack[name] = arg
                elif hasattr(arg, 'result'):
                    assert hasattr(arg, 'dtype')
                    if arg.dtype != NDArray:
                        if not issubclass(arg.dtype, typing.Iterable):
                            raise exceptions.TypeError('Expected array. Got {} of type {}'.format(arg, arg.dtype))
                    input_pack[name] = arg
                else:
                    raise exceptions.TypeError('Expected NDArray, but got {} of type {}'.format(arg, type(arg)))
            else:
                if isinstance(arg, typing.Iterable) and not isinstance(arg, (str, bytes)):
                    raise exceptions.ApiError('Ensemble inputs not yet supported.')
                elif hasattr(arg, 'result'):
                    assert hasattr(arg, 'dtype')
                    assert arg.dtype == ptype
                    input_pack[name] = arg
                else:
                    if callable(ptype):
                        input_pack[name] = ptype(arg)
                    else:
                        input_pack[name] = arg
        self._input_pack = input_pack

    def input_pack(self, member=0):
        """Get the bound input pack for the operation.

        If the bound operation is an ensemble operation, get the input pack for
        the `member`th ensemble member (default 0).
        """
        for name, arg in self._input_pack.items():
            if hasattr(arg, 'result'):
                self._input_pack[name] = arg.result()
            if isinstance(arg, NDArray):
                self._input_pack[name] = arg.values
        pack = self.signature.bind(**self._input_pack)
        return pack


def ndarray(data=None, shape: tuple = (), dtype=None):
    """Create an NDArray object from the provided iterable.

    Arguments:
        data: object supporting sequence, buffer, or Array Interface protocol
        shape: integer tuple specifying the size of each of one or more dimensions
        dtype: data type understood by gmxapi

    If `data` is provided, `shape` and `dtype` are optional. If `data` is not
    provided, both `shape` and `dtype` are required.

    If `data` is provided and shape is provided, `data` must be compatible with
    or convertible to `shape`. See Broadcast Rules in :doc:`datamodel` documentation.

    If `data` is provided and `dtype` is not provided, data type is inferred
    as the narrowest scalar type necessary to hold any element in `data`.
    `dtype`, whether inferred or explicit, must be compatible with all elements
    of `data`.

    The returned object implements the gmxapi N-dimensional Array Interface.

    ToDo: Does ndarray accept Futures in data and produce a Future? Or does it
     reject Futures in input? In other words, is ndarray guaranteed to produce a
     locally available object, and, if so, does it do so by rejecting non-local
     data, by implicitly resolving data locally, or with behavior controlled by
     additional keywords? Suggestion (MEI): uniform local-only behavior; separate
     function(s) for other use cases, a la numpy.asarray()
    """
    assert isinstance(shape, tuple)
    if data is None:
        if shape is None or dtype is None:
            raise exceptions.ValueError('If data is not provided, both shape and dtype are required.')
        array = NDArray()
        array.dtype = dtype
        array.shape = shape
        array.values = [dtype()] * shape[0]
    else:
        length = 0
        try:
            length = len(data)
        except:
            length = 1
            data = [data]
        if len(shape) > 0:
            if length > 1:
                assert length == shape[0]
        else:
            shape = (length,)
        if dtype is not None:
            assert isinstance(dtype, type)
            assert isinstance(data[0], dtype)
        else:
            dtype = type(data[0])
        for value in data:
            assert dtype(value) == value
        array = NDArray()
        array.dtype = dtype
        array.shape = shape
        array.values = list([dtype(value) for value in data])
    return array
