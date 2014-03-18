#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time integrators in 1D.
"""
import numpy as np
import scipy.sparse.linalg as splg
from ..core import ArchiveDictionary as ad
import operators as op


def trapezoid_step(operators, forcing, coeffs, operator_lhs, previous_solution,
                   time, timestep, constant_load=False, load=None):
    """Perform a single step of the Trapezoid rule.

    Arguments:
    - `operators`: an instance of Operators1D.
    - `forcing`: forcing function, capable of taking numpy arrays as
      input: see the `get_load_vector` of Operators1D.
    - `coeffs`: coefficients for the linear problem. Should be a
      dictionary with keys 'diffusion' and 'convection'.
    - `operator_lhs`: a factorized linear system (see
      scipy.sparse.linalg.factorized)
    - `previous_solution`: solution at `time - stepstep`.
    - `time`: current time index.
    - `timestep`: current time step.

    Returns:
    Trapezoid-rule approximation of the solution at time `time`.
    """
    if constant_load:
        rhs = 2*load
    else:
        rhs = operators.get_load_vector(time, forcing)
        rhs += operators.get_load_vector(time - timestep, forcing)
    rhs -= coeffs['diffusion']*operators.stiffness.dot(previous_solution)
    rhs -= coeffs['convection']*operators.convection.dot(previous_solution)
    rhs *= timestep/2.0
    rhs += operators.mass.dot(previous_solution)
    rhs[0] = operators.left_dirichlet_value
    rhs[-1] = operators.right_dirichlet_value

    return operator_lhs(rhs)


def time_integrate(operators, forcing, coeffs, times, initial,
                   constant_load=False, solver='trapezoid'):
    """Integrate a linear convection-diffusion problem in time.

    Arguments:
    - `operators`: an instance of Operators1D.
    - `forcing`: forcing function, capable of taking numpy arrays as
      input: see the `get_load_vector` of Operators1D.
    - `coeffs`: coefficients for the linear problem. Should be a
      dictionary with keys 'diffusion' and 'convection'.
    - `times`: time indices.
    - `initial`: Initial value of solution.
    - `constant_load`: truth-value of whether or not the load vector
      varies in time. Defaults to False.
    - `solver`: time integrator to use. Defaults to 'trapezoid'.

    Returns:
    An instance of ArchiveDictionary.
    """
    time_steps = times[1:] - times[:-1]
    if np.linalg.norm(time_steps - time_steps.mean()) > 10e-12:
        raise NotImplementedError("Unequal time stepping not available")
    timestep = times[1] - times[0]
    archive = ad.ArchiveDictionary()
    archive[times[0]] = initial
    if solver == 'trapezoid':
        operator_lhs = operators.mass + timestep/2.0*(
            coeffs['diffusion']*operators.stiffness
            + coeffs['convection']*operators.convection)
        if constant_load:
            load = operators.get_load_vector(0.0, forcing)
        else:
            load = None
        op.set_boundary_condition_rows(operator_lhs, value=1.0)
        factorized = splg.factorized(operator_lhs.tocsc())
        for index, time in enumerate(times[1:], start=1):
            archive[time] = trapezoid_step(operators, forcing, coeffs,
                factorized, archive[times[index - 1]], time, timestep,
                constant_load=constant_load, load=load)
    else:
        raise NotImplementedError

    return archive
