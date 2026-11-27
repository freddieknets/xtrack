# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
from warnings import warn
from typing import Literal
from collections import defaultdict
from collections.abc import Mapping

from madng_tpsa import Tpsa, TpsaMap, symplectic_unit_matrix

from .longitudinal import Method, MapLongitudinalIndices, _parse_lidx


def nominal_symplecticity_error(m: TpsaMap, lidx: MapLongitudinalIndices) -> float:
    n = lidx.map_size if lidx.method == '56d' else lidx.rank
    J = m.evaluate_jacobian()[:n, :n]
    S = symplectic_unit_matrix(n)
    return float(np.linalg.norm(J.T @ S @ J - S, ord='fro'))


def domain_symplecticity_error(
    m: TpsaMap,
    lidx: MapLongitudinalIndices,
    radii: Mapping[str, float],
) -> float:
    """Upper bound on the truncated symplectic defect
    over a box in phase space and parameter space.
    """
    d = m.descriptor
    names = d.var_names + d.param_names

    if set(radii) != set(names):
        raise ValueError("Provide one radius for each variable and parameter.")

    scales = np.array([radii[name] for name in names])
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Radii must be finite and positive.")

    n = lidx.map_size if lidx.method == '56d' else lidx.rank
    defect = lidx.symplectic_defect(m)

    # Reconstruct one coefficient matrix per monomial.
    coefficients = {}

    for index, t in enumerate(defect.flat):
        i, j = divmod(index, n)
        for mono, value in t.monomial_coeffs(tol=0.).items():
            if mono not in coefficients:
                coefficients[mono] = np.zeros((n, n))
            coefficients[mono][i, j] = value

    error = 0.
    for mono, matrix in coefficients.items():
        weight = np.prod(scales ** np.asarray(mono))
        norm = np.linalg.norm(matrix, ord='fro')
        error += weight * norm

    return float(error)


def symplectify(
    centred_map: TpsaMap[Tpsa],
    *,
    method: Method | None = None,
    centre_tol: float = 1e-11,
    rcond: float = 1e-12,
    symp_tol: float = 1e-12,
    damp_tol: float = 1e-8,
    radii: Mapping[str, float] | None = None,
    on_large_defect: Literal['raise', 'warn', 'ignore'] = 'raise',
    force: bool = False,
    _lidx: MapLongitudinalIndices | None = None,
) -> TpsaMap[Tpsa]:
    """Symplectify a centred parametric TPSA map.

    Nonlinear extension of Healy's symplectification algorithm
    using a midpoint generating function.

    External parameters are treated as spectators. In 5D and
    56D, the conserved momentum is retained, and in 56D the
    longitudinal map is canonically completed.

    The input map is not modified.
    """
    lidx = _parse_lidx(method, _lidx)

    if not 0 <= symp_tol < damp_tol:
        raise ValueError("Expected 0 <= `symp_tol` < `damp_tol`.")

    if on_large_defect not in ('raise', 'warn', 'ignore'):
        raise ValueError("Invalid `on_large_defect` policy.")

    if centred_map.is_complex:
        raise ValueError("Expected a real Cartesian map.")

    # Require the coasting structure to hold exactly.
    lidx.check_map(centred_map, tol=0.)

    d = centred_map.descriptor
    rank = lidx.rank
    order = centred_map.truncation_order

    if d.var_names != lidx.var_names:
        raise ValueError("Unexpected map coordinate ordering.")

    if order < 1:
        raise ValueError("Map order must be positive.")

    canonical_dim = 6 if lidx.method == '56d' else rank

    if any(v < order for v in d.max_orders[:canonical_dim]):
        raise ValueError(
            "Symplectification requires unrestricted canonical "
            "coordinate orders up to the map truncation order."
        )

    identity = TpsaMap(d.vars())
    m = centred_map.copy()

    # Ensure exact centring
    # ---------------------

    # Check the entire parameter-dependent fixed-point residual,
    # not merely the nominal constant part.
    for t in m.components[:rank]:
        for mono, coeff in t.monomial_coeffs(tol=0.).items():
            if any(mono[:rank]):
                continue
            if abs(coeff) > centre_tol:
                raise ValueError(f"Map is not centred: monomial {mono}, coefficient {coeff:.6e}")
            t.set(mono, 0.)

    # A 56D map may have a nominal longitudinal drift.
    # Remove it temporarily because inverse() requires a
    # vanishing constant part, and restore it afterwards.
    longitudinal_drift = 0.

    if lidx.method == '56d':
        longitudinal_drift = m.const_part[lidx.time_index]

    m.const_part = 0.

    # Verify magnitude of symplectic defect
    # -------------------------------------

    if not force:
        nominal_error = nominal_symplecticity_error(centred_map, lidx)
        error = nominal_error
        error_type = 'nominal'

        # Only compute the expensive full TPSA defect if
        # the nominal map has passed the first guard.
        if np.isfinite(error) and error < damp_tol and radii is not None:
            error = domain_symplecticity_error(centred_map, lidx, radii)
            error_type = 'domain-weighted'

        if not np.isfinite(error) or error >= damp_tol:
            message = (
                "Map is too far from symplectic to apply "
                "Healy symplectification: "
                f"{error_type} defect={error:.6e}, "
                f"threshold={damp_tol:.6e}."
            )

            if on_large_defect == 'raise':
                raise ValueError(message)

            if on_large_defect == 'warn':
                warn(message, RuntimeWarning, stacklevel=2)

            return centred_map.copy()

        # Only skip symplectification if the FULL defect
        # has been checked.
        if radii is not None and error <= symp_tol:
            # No need to symplectify
            return centred_map.copy()

    # Transform to midpoint coordinates
    # ---------------------------------

    # U = (I + M) / 2
    # g = S (U^-1 - I)

    midpoint = TpsaMap([
        (zi + mi) * 0.5
        for zi, mi in zip(identity.components, m.components)
    ])

    midpoint_inv = _checked_inverse(midpoint, rcond=rcond)

    displacement = [
        ui - zi
        for ui, zi in zip(
            midpoint_inv.components[:canonical_dim],
            identity.components[:canonical_dim],
        )
    ]

    g = _apply_symplectic_unit(displacement)

    # Project onto a gradient field
    # -----------------------------

    g_symp = _project_to_gradient(g, lidx=lidx, order=order)

    # Reconstruct the symplectic map
    # ------------------------------

    # Q = I - S g_symp
    # M_symp = 2 Q^-1 - I

    sg = _apply_symplectic_unit(g_symp)

    q_components = list(identity.components)

    for i in range(canonical_dim):
        q_components[i] = identity.components[i] - sg[i]

    # In 5D, momentum was not included in canonical_dim,
    # hence Q[pzeta] remains the identity.

    q = TpsaMap(q_components)
    q_inv = _checked_inverse(q, rcond=rcond)

    result = TpsaMap([
        (2 * qi - zi).truncate(order)
        for qi, zi in zip(
            q_inv.components,
            identity.components,
        )
    ])

    # Enforce exact momentum conservation.
    if lidx.momentum_index is not None:
        pi = lidx.momentum_index
        result[pi] = d.var(pi + 1, order=order)

    if lidx.method == '56d':
        ti = lidx.time_index
        result[ti] = result[ti] + longitudinal_drift

    return result

def _apply_symplectic_unit(
    components: list[Tpsa],
) -> list[Tpsa]:
    """Apply S with canonical (q, p) ordering."""

    return [
        -components[i - 1] if i % 2 else components[i + 1]
        for i in range(len(components))
    ]


def _checked_inverse(
    m: TpsaMap[Tpsa],
    *,
    rcond: float,
) -> TpsaMap[Tpsa]:
    """Invert a map, checking the nominal Jacobian first."""

    jac = m.evaluate_jacobian()
    singular_values = np.linalg.svd(
        jac,
        compute_uv=False,
    )

    if singular_values[-1] <= rcond * max(
        1., singular_values[0]
    ):
        raise ValueError(
            "Singular or ill-conditioned Cayley transform "
            "during symplectification."
        )

    return m.inverse()

def _project_to_gradient(
    g: list[Tpsa],
    *,
    lidx: MapLongitudinalIndices,
    order: int,
) -> list[Tpsa]:
    """Project a polynomial field onto a gradient field.

    Radial integration is performed over the oscillatory
    coordinates. For 56D, the longitudinal component is
    subsequently completed canonically.
    """
    d = g[0].descriptor
    rank = lidx.rank

    canonical_dim = 6 if lidx.method == '56d' else rank

    # Store H outside the TPSA descriptor, as it can contain
    # monomials of total degree order + 1.
    h: dict[tuple[int, ...], float] = defaultdict(float)

    # Radial integration
    # ------------------

    # H = integral_0^1 z . g(t*z) dt
    # Only the first `rank` coordinates are scaled by t

    for i in range(rank):
        for mono, coeff in g[i].monomial_coeffs(tol=0.).items():

            degree = sum(mono[:rank])

            # Maintain the parameter-dependent fixed point.
            # A centred field has no such terms, apart from
            # possible numerical inversion residuals.
            if degree == 0:
                continue

            if sum(mono) > order:
                continue

            # In 56D, H must be independent of zeta.
            if lidx.method == '56d' and mono[lidx.time_index]:
                continue

            power = list(mono)
            power[i] += 1

            h[tuple(power)] += coeff / (degree + 1)

    # Differentiate H to obtain a gradient
    # ------------------------------------

    gradient = [
        d.zero(order=order)
        for _ in range(canonical_dim)
    ]

    for mono, coeff in h.items():
        if coeff == 0.:
            continue

        for i in range(canonical_dim):
            exponent = mono[i]

            if exponent == 0:
                continue

            power = list(mono)
            power[i] -= 1
            power = tuple(power)

            if sum(power) > order:
                continue

            if not d.is_valid_monomial(power):
                continue

            gradient[i].set(power, exponent * coeff)

    # 56D: preserve the on-axis longitudinal drift
    # --------------------------------------------

    # The pzeta derivative of H_perp is already included
    # above. Add dH_0/dpzeta from the original field

    if lidx.method == '56d':
        pi = lidx.momentum_index
        ti = lidx.time_index

        for mono, coeff in g[pi].monomial_coeffs(tol=0.).items():

            # Only retain the on-axis contribution.
            if any(mono[:rank]) or mono[ti]:
                continue

            if sum(mono) > order:
                continue

            gradient[pi].set(mono, coeff)

    return gradient
