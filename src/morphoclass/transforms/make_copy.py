# Copyright © 2022-2022 Blue Brain Project/EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the `MakeCopy` transform."""

from __future__ import annotations

from copy import copy
from copy import deepcopy

import morphio
import torch
from morphio.mut import Morphology

from neurom.core import Morphology as Neuron
from tmd.Tree.Tree import Tree

from morphoclass.data.morphology_data import MorphologyData
from morphoclass.utils import print_warning


class MakeCopy:
    """Transform that makes a copies of morphology data samples.

    Parameters
    ----------
    keep_fields : container of str
        The fields of `Data` objects to keep. If equal to `None`, then all
        fields are copied.
    """

    def __init__(self, keep_fields=None):
        # Normalize keep_fields to a set for consistent membership testing
        if keep_fields is None:
            self.keep_fields = None
        elif isinstance(keep_fields, str):
            self.keep_fields = {keep_fields}
        else:
            self.keep_fields = set(keep_fields)

    @classmethod
    def clone_obj(cls, key, obj):
        """Try to clone an object.

        If the cloning fails then the original object is returned.

        Parameters
        ----------
        key : str
            The object name. Only used for printing information about the object.
        obj
            The object to clone.

        Returns
        -------
        new_obj
            A clone of `obj`.
        """
        if torch.is_tensor(obj):
            return obj.clone()
        elif type(obj) is Morphology:
            # morphio.mut.Morphology (mutable)
            return Morphology(obj)
        elif isinstance(obj, morphio.Morphology):
            # morphio._morphio.Morphology (immutable) - convert to neurom for compatibility
            return Neuron(obj)
        elif type(obj) is Neuron:
            # neurom.core.Morphology
            return Neuron(obj)
        elif type(obj) in [list, tuple, set, frozenset]:
            return type(obj)(cls.clone_obj(None, x) for x in obj)
        elif type(obj) is Tree:
            new_obj = copy(obj)
            new_obj.x = copy(obj.x)
            new_obj.y = copy(obj.y)
            new_obj.z = copy(obj.z)
            new_obj.d = copy(obj.d)
            new_obj.t = copy(obj.t)
            new_obj.p = copy(obj.p)
            new_obj.dA = copy(obj.dA)

            return new_obj
        else:
            try:
                new_obj = deepcopy(obj)
            except TypeError:
                new_obj = obj
                print_warning(f"Couldn't copy contents of key {key}, keeping the original.")
            return new_obj

    def __call__(self, data):
        """Make a copy of the data sample.

        Parameters
        ----------
        data
            The input morphology data sample.

        Returns
        -------
        data
            A copy of the input morphology data sample.
        """
        # In newer PyTorch Geometric, data attributes are stored in data._store
        # Use the internal _store to access all data attributes
        if hasattr(data, '_store'):
            # Modern PyTorch Geometric with _store
            items = data._store.items()
        else:
            # Fallback for older versions
            items = data.__dict__.items()

        new_data = MorphologyData.from_dict({
            k: self.clone_obj(k, v)
            for k, v in items
            if self.keep_fields is None or k in self.keep_fields
        })
        return new_data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}(keep_fields={self.keep_fields})"
