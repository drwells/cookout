#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class encapsulating a 1D finite element mesh.
"""
import numpy as np


class Mesh1D(object):
    """Finite element mesh in 1D.
    """

    def __init__(self, num_elements, element_order):
        """Encapsulation of a 1D finite element mesh. Assumes that all
        elements have the same width.

        Arguments:
        - `num_elements`: Number of elements in the mesh.
        - `element_order`: Polynomial order of the elements.
        """
        self.element_order = element_order

        num_nodes = (num_elements)*element_order + 1
        self.nodes = np.linspace(0, 1, num_nodes)
        self.elements = np.zeros((num_elements)*(element_order + 1),
                                 dtype=np.int)
        self.elements.shape = (num_elements, element_order + 1)

        node_index = 0
        for element in self.elements:
            element[:] = range(node_index, node_index + element_order + 1)
            node_index += element_order
