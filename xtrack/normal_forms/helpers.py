# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from __future__ import annotations

import numpy as np
from math import factorial
from typing import TYPE_CHECKING

from madng_tpsa import symplectic_unit_matrix, Tpsa, ComplexTpsa, TpsaMap, TpsaTensor

from ..line import Line

if TYPE_CHECKING:
    from madng_tpsa import Descriptor

    from .longitudinal import MapLongitudinalIndices
    from ..environment import Environment


def get_descriptor_from_env_or_line(env: Environment | Line) -> Descriptor | None:
    if isinstance(env, Line):
        env = env.env

    descriptor = None
    for name in env.vars.keys():
        value = env.vars.val[name]
        if not isinstance(value, (Tpsa, ComplexTpsa)):
            continue
        if descriptor is None:
            descriptor = value.descriptor
        elif value.descriptor is not descriptor:
            raise ValueError(f"Inconsistent TPSA descriptor for variable '{name}'.")

    manager = env.ref_manager
    refs = manager.rdeps.keys() | manager.tartasks.keys()
    for ref in refs:
        value = ref._get_value()
        if not isinstance(value, (Tpsa, ComplexTpsa)):
            continue
        if descriptor is None:
            descriptor = value.descriptor
        elif value.descriptor is not descriptor:
            raise ValueError(f"Inconsistent TPSA descriptor for reference '{ref}'.")

    return descriptor


def as_tpsa_tensor(values, descriptor):
    values = np.asarray(values)
    return TpsaTensor([
        [descriptor.constant(float(value)) for value in row]
        for row in values
    ])


def symplectic_inverse(A: TpsaTensor):
    d = A.descriptor
    n = A.shape[0]
    S = symplectic_unit_matrix(n)
    return -(S @ A.T @ S)


def exp_nilpotent_matrix(K: TpsaTensor, max_order: int, min_degree: int):
    d = K.descriptor
    n = K.shape[0]
    result = as_tpsa_tensor(np.eye(n), d)
    power = result

    for j in range(1, max_order // min_degree + 1):
        power = power @ K
        result = result + power / factorial(j)
    return result


def matrix_to_map(matrix: TpsaTensor, lidx: MapLongitudinalIndices) -> TpsaMap:
    """Represent a matrix as a canonical map.

    For a regular 6D system, we have the following matrix normalisation:
    z_phys = A(k) z_norm

    In a 5D system, A is a 4x4 matrix and pzeta is untouched:
    z^4D_phys = A(k) z^4D_norm
    (pzeta)_phys = (pzeta)_norm

    However, in a 56D system, zeta needs correction:
    z^4D_phys = A(k) z^4D_norm
    zeta_phys = zeta_norm - 1/2 (z^4D_norm)^T A^T S (dA / dpzeta) z^4D_norm
    pzeta_phys = pzeta_norm

    Note that, because $A$ is symplectic, we have
    A^-1 = -S A^T S
    and hence
    zeta_norm = zeta_phys - 1/2 (z^4D_phys)^T A^-1^T S (dA^-1 / dpzeta) z^4D_phys

    Hence, the argument `matrix` can represent both A and A^-1
    """
    d = matrix.descriptor
    z = TpsaMap(d)
    rk = lidx.rank
    ti = lidx.time_index
    pi = lidx.momentum_index

    result = TpsaMap(d)

    A = matrix[:rk, :rk]
    w = z[:rk]
    result[:rk] = A @ w

    if ti is not None and pi is not None:
        dA = A.partial_derivative(pi + 1)
        S = symplectic_unit_matrix(rk)
        G = A.T @ S @ dA
        result[ti] = z[ti] - (w @ G @ w) / 2


    return TpsaMap(result.components)
