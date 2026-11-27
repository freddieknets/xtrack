# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from __future__ import annotations

import numpy as np
from warnings import warn
from typing import TYPE_CHECKING
from dataclasses import dataclass

from madng_tpsa import Descriptor, TpsaMap

from .longitudinal import MapLongitudinalIndices, _parse_lidx
from .helpers import get_descriptor_from_env_or_line
from ..general import _print
from ..line import Line, LineParticleRef, _preserve_track_flags
from ..particles import Particles
from ..twiss.closed_orbit import ClosedOrbitSearchError
from ..tpsa import COORDS, ParticlesTpsa

if TYPE_CHECKING:
    from .longitudinal import Method


@dataclass(slots=True, frozen=True, eq=False)
class OneTurnMap:
    r"""Container holding different representations of the one-turn map.

    Let :math:`M(z, k)` denote the physical one-turn transformation, :math:`c_0`
    the nominal closed orbit, and :math:`c_k(k)` its parametric correction.
    Then we distinguish the following maps:

    .. math::
        \begin{aligned}
            \text{Raw tracked map}	P(z,k) &= M(c_0+z, k) \\
            \text{Nominally centred map}	M_0(z,k) &= P(z,k) - c_0 \\
            \text{Parametrically centred map} M_c(z,k) &= M_0(z+c_k, k) - c_k
        \end{aligned}

    The physical one-turn map (around 0) is never calculated as, due to its
    truncated nature, it might be a considerably worse approximation in the
    region of phase space we are interested in.

    We define the translation map :math:`X(z, k)` to centre the nominal map
    around the parametric closed orbit:

    .. math::
        M_c = X^{-1} \circ M_0 \circ X

    Note that by definition, :math:`P(z, k)` is set in 6D, while :math:`M_0`,
    :math:`M_c`, and :math:`X` are in the chosen dimension (4D, 5D, 56D, or 6D).
    In case of coasting beam (method='56D'), the translation map includes a
    correction of the longitudinal position coordinate to keep it canonical:

    .. math::
        \begin{pmatrix} \vec{z}_\perp' & \zeta' & p_\zeta' \end{pmatrix}
        =
        \begin{pmatrix}
            A_\perp\cdot\vec{z}_\perp + c_k(p_\zeta) &
            \zeta
              + \frac{1}{2}\vec{z}_\perp^T \cdot
                \frac{\partial A^T}{\partial p_\zeta} \cdot S_\perp \cdot A
                \cdot \vec{z}_\perp
              + \frac{\partial c_k^T}{\partial p_\zeta} \cdot S_\perp
                \cdot \vec{z}_\perp &
            p_\zeta
        \end{pmatrix}

    where :math:`A(p_\zeta)` is the linear normalising matrix in 4D and
    :math:`c_k(p_\zeta)` the parametric closed orbit in 4D. Note that to get
    the correction formula for the closed orbit translation, we can use
    :math:`A = I`.
    """
    method: Method
    tracked_map: ParticlesTpsa           # P(z, k) = Mphys(z+c0, k):  Actual output of tracking, including auxiliary coordinates
    retracked_map: ParticlesTpsa | None  # Retracked (c0 optimised)
    reference_orbit: np.ndarray          # c0:  Numerical closed orbit c0
    orbit: TpsaMap                       # c0 + c(k):  Full closed orbit in the dimensions of the map
    orbit_6d: TpsaMap                    # Full closed orbit included coordinates potentially not present in the map
    map: TpsaMap                         # M(z, k) = Mphys(z+c0, k) - c0:  Map relative to the reference orbit
    centred_map: TpsaMap                 # Mcentre = X^(-1) << M << X:  Map relative to the full orbit (and longitudinally corrected)
    translation: TpsaMap                 # X:  Translation to centre and longitudinally correct (in 56D) the map
    inverse_translation: TpsaMap         # X^(-1)
    centring_residual: np.ndarray        # Mcentre(0, 0) = diff  (should be 0)
    longitudinal_drift: float


def get_one_turn_map(
    line,
    *,
    method: Method | None = None,
    particle_ref: Particles | LineParticleRef | None = None,
    co_ref: Particles | None = None,
    co_tracking_iterations: int = 1,
    retrack_around_parametric_co: bool = True,
    momentum_as_parameter: bool = None,
    descriptor: Descriptor | None = None,
    verbose: bool = False,
    _lidx: MapLongitudinalIndices | None = None,
):
    if not isinstance(line, Line):
        raise ValueError("Argument `line` should be an `xt.Line`!")
    particle_ref = _resolve_particle_ref(line, particle_ref)

    _lidx = _parse_lidx(method, _lidx)
    if _lidx.method == '4d' and momentum_as_parameter is None:
        momentum_as_parameter = True

    # Descriptor
    # ----------
    env_desc = get_descriptor_from_env_or_line(line)
    if descriptor is None:
        if env_desc is None:
            descriptor = Descriptor(var_names=COORDS, order=6)
        else:
            descriptor = env_desc
    elif env_desc is not None and descriptor is not env_desc:
        raise ValueError("Provided descriptor not the same as used in the environment knobs!")


    # Numerical (parameter-free) closed orbit
    # ---------------------------------------
    co_method = "6d" if _lidx.rank == 6 else "4d"
    # Note that Xtrack's native method="5d" has a different meaning, namely closing
    # transverse coordinates and energy (allowing zeta to slip), while we mean a
    # transverse map with conserved momentum variable. Hence, even for _lidx.method='5d'
    # we should use find_closed_orbit(method='4d').
    if co_ref is None:
        if line.tracker is None:
            line.build_tracker()
        try:
            co_ref = line.find_closed_orbit(
                particle_ref=particle_ref,
                method=co_method,
                freeze_longitudinal=False
            )
        except ClosedOrbitSearchError as e:
            raise ClosedOrbitSearchError(
                f"Failed to find the closed orbit for "
                f"longitudinal normal form method {_lidx.method!r}.\n"
                f"Please try again with a different method."
            ) from e

    elif not isinstance(co_ref, Particles):
        raise ValueError("`co_ref` needs to be a `Particles` instance!")
    elif co_ref._capacity != 1:
        raise ValueError("`co_ref` needs to be a single particle!")

    c0 = np.array([getattr(co_ref, cc)[0] for cc in COORDS])

    # Tracking
    # --------
    with _preserve_track_flags(line):
        if _lidx.method in ('4d', '5d', '56d'):
            line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = True

        # One-turn map around numerical (parameter-free) closed orbit
        tracked_map = _create_map_from_co(co_ref, descriptor, c0)
        line.track(tracked_map)
        m = tracked_map.to_map() - c0
        if verbose:
            _print(f"R orbit residual:  {m.const_part}")

        # Newton correction to improve residual
        for i in range(co_tracking_iterations - 1):
            rank = _lidx.rank
            R = m.evaluate_jacobian()[:rank, :rank]
            epsilon = m.const_part[:rank]
            correction = np.linalg.solve(np.eye(rank) - R, epsilon)
            c0[:rank] += correction

            tracked_map = _create_map_from_co(co_ref, descriptor, c0)
            line.track(tracked_map)
            m = tracked_map.to_map() - c0
            if verbose:
                _print(f"Corrected orbit residual (iteration {i+2}):  {m.const_part}")

        # Solve for parameter-dependent closed orbit
        ck = _get_parameter_orbit(m, _lidx, momentum_as_parameter=momentum_as_parameter)
        forward, backward = _get_orbit_translation(ck, _lidx)
        orbit = ck.copy()
        orbit.const_part = c0

        retracked_map = None
        if retrack_around_parametric_co:
            # Track again
            retracked_map = _create_map_from_initial_map(co_ref, forward + c0)
            line.track(retracked_map)
            m = retracked_map.to_map() - c0
            # Solve parameter-dependent closed orbit again
            ck = _get_parameter_orbit(m, _lidx, momentum_as_parameter=momentum_as_parameter)
            forward, backward = _get_orbit_translation(ck, _lidx)
            orbit = ck.copy()
            orbit.const_part = c0

    # Centre!
    # -------
    centred_map = backward << (m << forward)

    # Mc6 = backward << (m << forward)
    # options = dict(momentum_as_parameter=True)
    # Mc4 = _lidx.adapt_6d_map(Mc6, **options)
    # M4 = _lidx.adapt_6d_map(m, **options)
    # X4 = _lidx.adapt_6d_map(forward, **options)
    # Xinv4 = _lidx.adapt_6d_map(backward, **options)
    # # Does dimensional reduction commute with composition?
    # error = (Xinv4 << (M4 << X4)) - Mc4
    # print(error.format())

    # Centre exactly around zero
    diff = centred_map.const_part.copy()
    if _lidx.method == "56d":
        longitudinal_drift = diff[4]
        diff[4] = 0.
    else:
        longitudinal_drift = 0.
    # Careful: setting slices of TpsaTensor.const_part does not work!
    const_part = [
        0 if i in _lidx._selector else centred_map.const_part[i]
        for i in range(centred_map.num_vars)
    ]
    centred_map.const_part = const_part
    if verbose:
        _print(f"Final residual of parametric orbit:  {diff}")

    return OneTurnMap(
        method = _lidx.method,
        tracked_map = tracked_map,
        retracked_map = retracked_map,
        reference_orbit = c0,
        orbit = _lidx.adapt_6d_map(orbit, momentum_as_parameter=momentum_as_parameter),
        orbit_6d = orbit,
        map = _lidx.adapt_6d_map(m, momentum_as_parameter=momentum_as_parameter),
        centred_map = _lidx.adapt_6d_map(centred_map, momentum_as_parameter=momentum_as_parameter),
        translation = _lidx.adapt_6d_map(forward, momentum_as_parameter=momentum_as_parameter),
        inverse_translation = _lidx.adapt_6d_map(backward, momentum_as_parameter=momentum_as_parameter),
        centring_residual = diff,
        longitudinal_drift = longitudinal_drift
    )


def _resolve_particle_ref(line, particle_ref):
    if particle_ref is None:
        if line.particle_ref is None:
            raise ValueError("A reference particle is required")
        particle_ref = line.particle_ref.copy()
    elif not isinstance(particle_ref, (Particles, LineParticleRef )):
        raise ValueError("`particle_ref` needs to be a `Particles` instance!")
    elif particle_ref._capacity != 1:
        raise ValueError("`particle_ref` needs to be a single particle!")
    return particle_ref


def _create_map_from_co(co_ref, descriptor, new_co=None):
    # Reference, for p0c etc
    particle_kwargs = co_ref.to_dict()

    # Update coordinates
    if new_co is not None:
        # Identity gets automatically added to constant terms in ParticlesTpsa
        particle_kwargs.update(dict(zip(COORDS, new_co)))

    # Need to remove derived quantities such that they can be recalculated
    for name in ('delta', 'ptau', 'rpp', 'rvv', '_rpp', '_rvv'):
        particle_kwargs.pop(name, None)

    return ParticlesTpsa(descriptor=descriptor, **particle_kwargs)


def _create_map_from_initial_map(co_ref, initial_map):
    particle_kwargs = co_ref.to_dict()

    # Tracking must not mutate the original transformation.
    particle_kwargs.update(initial_map.copy().named_components)

    # Need to remove derived quantities such that they can be recalculated
    for name in ('delta', 'ptau', 'rpp', 'rvv', '_rpp', '_rvv'):
        particle_kwargs.pop(name, None)

    return ParticlesTpsa(descriptor=initial_map.descriptor, **particle_kwargs)


def _get_parameter_orbit(m, _lidx, *, momentum_as_parameter=False):
    d = m.descriptor
    identity = TpsaMap(d)

    # Remove const_part (needed for inverse) and subtract identity: M'(z, k) = M(z,k) - z - c1
    shifted = m - identity
    shifted.const_part = 0.

    # Root of M' is fixed-point: M( c(k), k ) = c(k) + c1
    if _lidx.method == '6d':
        return shifted.inverse().substitute(0.)

    # Longitudinal variables are spectators in the transverse fixed-point problem.
    shifted[4] = identity.zeta.copy()
    shifted[5] = identity.pzeta.copy()
    inverse = shifted.partial_inverse([True, True, True, True, False, False])

    keep_momentum = _lidx.method in ('5d', '56d') or momentum_as_parameter

    if keep_momentum:
        # Let pzeta free
        root = inverse.substitute(x=0., px=0., y=0., py=0., zeta=0.)
    else:
        root = inverse.substitute(0.)

    return TpsaMap([*root.components[:4], d.zero(), d.zero()])


def _get_orbit_translation(ck, _lidx):
    I = TpsaMap(ck.descriptor)
    forward  = I + ck
    backward = I - ck

    if _lidx.method != "56d":
        return forward, backward

    # Translation:
    #     zT   -> zT + ck(pzeta)
    #     zeta -> zeta + g(zt, pzeta)
    #
    # Inverse translation:
    #     zT   -> zT - ck(pzeta)
    #     zeta -> zeta - g(zT - ck(pzeta), pzeta) = zeta - g(zT, pzeta) + g(ck(pzeta), pzeta)
    #
    # with g = (d ck / d pzeta) S4 zT

    cx_p  = ck.x.derivative("pzeta")
    cpx_p = ck.px.derivative("pzeta")
    cy_p  = ck.y.derivative("pzeta")
    cpy_p = ck.py.derivative("pzeta")

    g      = cx_p * I.px  - cpx_p * I.x  + cy_p * I.py  - cpy_p * I.y
    g_at_c = cx_p * ck.px - cpx_p * ck.x + cy_p * ck.py - cpy_p * ck.y

    forward[4]  = I.zeta + g
    backward[4] = I.zeta - g + g_at_c

    return forward, backward


def get_fixed_point_residual(m: TpsaMap, lidx: MapLongitudinalIndices) -> TpsaMap:
    d = m.descriptor
    rank = lidx.rank
    zero_indices = tuple(range(rank))

    # In 56D, also evaluate at zeta = 0.
    if 'zeta' in d.var_names and rank == 4:
        zero_indices += (d.var_names.index('zeta'),)
    residual = TpsaMap([d.zero() for _ in range(d.num_vars)])
    for i in range(rank):
        for mono, coeff in m.components[i].monomial_coeffs(tol=0.).items():
            if any(mono[j] for j in zero_indices):
                continue
            residual.set_coefficient(i, mono, coeff)
    return residual

def get_fixed_point_residual_coefficients(
    m: TpsaMap,
    lidx: MapLongitudinalIndices,
) -> dict[tuple[int, ...], np.ndarray]:
    residual = get_fixed_point_residual(m, lidx)
    coefficients = {}
    for i, component in enumerate(residual.components[:lidx.rank]):
        for mono, coeff in component.monomial_coeffs(tol=0.).items():
            if mono not in coefficients:
                coefficients[mono] = np.zeros(lidx.rank)
            coefficients[mono][i] = coeff
    return coefficients

def remove_fixed_point_residual(m: TpsaMap, lidx: MapLongitudinalIndices,) -> tuple[TpsaMap, TpsaMap]:
    residual = get_fixed_point_residual(m, lidx)
    corrected = m - residual
    if lidx.method == '56d':
        rx_p = residual.x.derivative('pzeta')
        rpx_p = residual.px.derivative('pzeta')
        ry_p = residual.y.derivative('pzeta')
        rpy_p = residual.py.derivative('pzeta')
        corrected[4] = m.zeta - (
              rx_p  * corrected.px
            - rpx_p * corrected.x
            + ry_p  * corrected.py
            - rpy_p * corrected.y
        )
    return corrected, residual
