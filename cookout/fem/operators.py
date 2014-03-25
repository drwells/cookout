#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class encapsulating the linear operators in 1D.
"""
import numpy as np
import cookout.fem.local_matrices as lm
import scipy.sparse as sp


class Operators1D(object):
    """Discretizations of linear operators in 1D.
    """
    def __init__(self, num_elements, element_order, left_dirichlet_value=0.0,
                 right_dirichlet_value=0.0):
        """Encapsulation of the common finite element matricies (mass,
        stiffness, convection) in a single dimension. The mesh is
        assumed to discretize the interval [0, 1]. The boundary
        conditions are Dirichlet (not necessarily homogeneous).

        Arguments:
        - `num_elements`: Number of elements in the mesh.
        - `element_order`: Polynomial order of the elements.
        - `left_dirchlet_value`: Value on the left. Defaults to 0.0.
        - `right_dirchlet_value`: Value on the right. Defaults to 0.0.
        """
        self._num_elements = num_elements
        self._element_order = element_order
        self.left_dirichlet_value = left_dirichlet_value
        self.right_dirichlet_value = right_dirichlet_value
        self._num_nodes = (num_elements)*element_order + 1
        self.nodes = np.linspace(0, 1, self._num_nodes)
        self.elements = np.zeros((num_elements)*(element_order + 1),
                                 dtype=np.int)
        self.elements.shape = (num_elements, element_order + 1)

        node_index = 0
        for element in self.elements:
            element[:] = range(node_index, node_index + element_order + 1)
            node_index += element_order

        self.mass = self._get_matrix(lm.mass)
        self.convection = self._get_matrix(lm.convection)
        self.stiffness = self._get_matrix(lm.stiffness)

    def get_load_vector(self, time, forcing):
        """Calculate the load vector, excluding the boundary nodes.

        Arguments:
        - `time`: time index.
        - `forcing`: forcing function, called as f(t, x), capable of
        taking vectorized (x) input.
        """
        load_vector_1 = np.zeros(self.nodes.shape)
        load_vector_2 = np.zeros(self.nodes.shape)
        element_lengths = (self.nodes[self.elements[:, -1]]
                           - self.nodes[self.elements[:, 0]])
        # use Simpson's rule to integrate on each element.
        if self._element_order == 1:
            halfway_nodes = self.nodes[:-1] + element_lengths/2.0
            # left halves of basis functions.
            load_vector_1[1:-1] += 2*forcing(time, halfway_nodes[:-1])
            load_vector_1[1:-1] += forcing(time, self.nodes[1:-1])
            load_vector_1[1:-1] *= element_lengths[:-1]
            # right halves of basis functions.
            load_vector_2[1:-1] += forcing(time, self.nodes[1:-1])
            load_vector_2[1:-1] += 2*forcing(time, halfway_nodes[1:])
            load_vector_2[1:-1] *= element_lengths[1:]
            load_vector_1 += load_vector_2
            load_vector_1 /= 6.0
        elif self._element_order == 2:
            # bubble functions.
            load_vector_1[1:-1:2] += 4.0*element_lengths/6.0*forcing(time,
                self.nodes[1:-1:2])
            # left halves of basis functions with multi-element support.
            load_vector_1[2:-1:2] += element_lengths[1:]/6.0*forcing(time,
                self.nodes[2:-1:2])
            # right halves of basis functions with multi-element support.
            load_vector_1[2:-1:2] += element_lengths[:-1]/6.0*forcing(time,
                self.nodes[2:-1:2])
        elif self._element_order == 3:
            # use Simpson's 3/8 rule.
            raise NotImplementedError
        else:
            raise NotImplementedError
        return load_vector_1

    def _get_matrix(self, local_matrix_function):
        """Assemble a global matrix from a local matrix function.

        Arguments:
        - `local_matrix_function`: function returning a local matrix
          given arguments of polynomial order and element length (see
          local_matrices.py)
        """
        rows = list()
        columns = list()
        values = list()
        node_numbers = np.zeros((self._element_order + 1,
                                 self._element_order + 1), dtype=np.int)
        for element in self.elements:
            stepsize = self.nodes[element[-1]] - self.nodes[element[0]]
            local_matrix = local_matrix_function(self._element_order, stepsize)
            for row in node_numbers:
                row[:] = element

            local_matrix.shape = (local_matrix.size,)
            values.append(local_matrix)
            rows.append(node_numbers.flatten(order='C'))
            columns.append(node_numbers.flatten(order='F'))

        return sp.coo_matrix(
            (np.hstack(values), (np.hstack(rows), np.hstack(columns)))).tocsr()


def set_boundary_condition_rows(matrix, value=0.0):
    """Set the first and last rows of a matrix to a scalar along the
    main diagonal and zero otherwise.

    Arguments:
    - `matrix`: sparse CSR matrix which will be modified *in place*.
    - `value`: new value of the first and last entries of the main
      diagonal. Defaults to 0.0.
    """
    if not sp.csr.isspmatrix_csr(matrix):
        raise NotImplementedError("Internal formats besides CSR are "
                                  "not supported.")

    for row_index, name in [[0, "first"], [-1, "last"]]:
        if len(matrix.getrow(row_index).data) == 0:
            raise ValueError("The matrix must have entries in its {} row."
                             .format(name))

    matrix.sort_indices()
    matrix.data[matrix.indptr[0]:matrix.indptr[1]] = 0.0
    matrix.data[matrix.indptr[-2]:matrix.indptr[-1]] = 0.0
    matrix.data[matrix.indptr[0]] = value
    matrix.data[matrix.indptr[-1] - 1] = value
