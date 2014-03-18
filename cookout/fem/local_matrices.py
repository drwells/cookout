#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basis functions, quadrature points, and local matrices in 1D.
"""
import numpy as np
import sympy

x, a, b = sympy.symbols('x a b')

basis_function_lookup = {
    1: (1 - (x - a)/(b - a), (x - a)/(b - a)),
    2: ((x - (a + b)/2)*(x - b)/((a - (a + b)/2)*(a - b)),
        (x - a)*(x - b)/(((a + b)/2 - a)*((a + b)/2 - b)),
        (x - a)*(x - (a + b)/2)/((b - a)*(b - (a + b)/2))),
    3: ((3*a + 2*(b - a) - 3*x)*(3*a + (b - a) - 3*x)*(a + (b - a) - x)
        /(2*(b - a)**3),
        -9*(3*a + 2*(b - a) - 3*x)*(a + (b - a) - x)*(a - x)/(2*(b - a)**3),
         9*(3*a + (b - a) - 3*x)*(a + (b - a) - x)*(a - x)/(2*(b - a)**3),
        -1*(3*a + 2*(b - a) - 3*x)*(3*a + b - a - 3*x)*(a - x)/(2*(b - a)**3)),
}

# gauss points from
# http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules/quadrature_rules.html
# mapped to [0, 1]
gauss_points = 0.5 + 0.5*np.array([
    -0.906179845938663992797626878299,
    -0.538469310105683091036314420700,
    0.000000000000000000000000000000,
    0.538469310105683091036314420700,
    0.906179845938663992797626878299])

gauss_weights = 0.5*np.array([
    0.236926885056189087514264040720,
    0.478628670499366468041291514836,
    0.568888888888888888888888888889,
    0.478628670499366468041291514836,
    0.236926885056189087514264040720])


def mass(order, h):
    """Calculate the local mass matrix of a 1D finite element.

    Arguments:
    - `order`: Polynomial order of the local basis functions.
    - `h`: Length of the current element.
    """
    if order == 1:
        return np.array([
            [1.0/3*h, 1.0/6*h],
            [1.0/6*h, 1.0/3*h]])
    elif order == 2:
        return h/15.0*np.array([
            [2.0,  1.0, -0.5],
            [1.0,  8.0,  1.0],
            [-0.5, 1.0,  2.0]])
    elif order == 3:
        return h*np.array([[8.0/105, 33.0/560, -3.0/140, 19.0/1680],
                           [33.0/560, 27.0/70, -27.0/560, -3.0/140],
                           [-3.0/140, -27.0/560, 27.0/70, 33.0/560],
                           [19.0/1680, -3.0/140, 33.0/560, 8.0/105]])
    else:
        raise NotImplementedError


def convection(order, h):
    """Calculate the local convection matrix of a 1D finite element.

    Arguments:
    - `order`: Polynomial order of the local basis functions.
    - `h`: Length of the current element.
    """
    if order == 1:
        return np.array([
            [-1.0/2.0, 1.0/2.0],
            [-1.0/2.0, 1.0/2.0]])
    elif order == 2:
        return np.array([
            [-1.0/2.0, 2.0/3.0, -1.0/6.0],
            [-2.0/3.0, 0.0,      2.0/3.0],
            [1.0/6.0, -2.0/3.0,  1.0/2.0]])
    elif order == 3:
        return np.array([[-1.0/2.0,    57.0/80.0, -3.0/10.0,   7.0/80.0],
                         [-57.0/80.0,  0.0,        81.0/80.0, -3.0/10.0],
                         [3.0/10.0,   -81.0/80.0,  0.0,        57.0/80.0],
                         [-7.0/80.0,   3.0/10.0,  -57.0/80.0,  1.0/2.0]])
    else:
        raise NotImplementedError


def stiffness(order, h):
    """Calculate the local stiffness matrix of a 1D finite element.

    Arguments:
    - `order`: Polynomial order of the local basis functions.
    - `h`: Length of the current element.
    """
    if order == 1:
        return np.array([
            [1.0/h, -1.0/h],
            [-1.0/h, 1.0/h]])
    elif order == 2:
        return 1/(h*3.0)*np.array([
            [7.0, -8.0,   1.0],
            [-8.0, 16.0, -8.0],
            [1.0, -8.0,   7.0]])
    elif order == 3:
        return 1.0/h*np.array([[37.0/10, -189.0/40, 27.0/20, -13.0/40],
                               [-189.0/40, 54.0/5, -297.0/40, 27.0/20],
                               [27.0/20, -297.0/40, 54.0/5, -189.0/40],
                               [-13.0/40, 27.0/20, -189.0/40, 37.0/10]])
    else:
        raise NotImplementedError


def values(order):
    """Evaluate the basis functions at the quadrature points.

    Rows of the returned array correspond to basis functions and columns
    correspond to quadrature points.

    Arguments:
    - `order`: Polynomial order of the local basis functions.
    """
    values = list()
    basis = basis_function_lookup[order]

    for func in basis:
        values.append([func.subs({a: 0, b: 1, x: xg})
                            for xg in gauss_points])
    return np.array(values, dtype=np.float64)


def derivatives(order):
    """Calculate the derivatives of the reference basis functions at the
    quadrature points.

    Rows of the returned array correspond to basis functions and columns
    correspond to quadrature points.

    Arguments:
    - `order`: Polynomial order of the local basis functions.
    """
    values = list()
    basis = basis_function_lookup[order]

    for func in basis:
        values.append([func.diff(x).subs({a: 0, b: 1, x: xg})
                       for xg in gauss_points])
    return np.array(values, dtype=np.float64)

quad_values = {key: values(key) for key in basis_function_lookup.keys()}
quad_dx = {key: derivatives(key) for key in basis_function_lookup.keys()}
