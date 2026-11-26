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
from ..line import Line, LineParticleRef
from ..particles import Particles
from ..tpsa import COORDS, ParticlesTpsa

if TYPE_CHECKING:
    from .longitudinal import Method


@dataclass(slots=True, frozen=True, eq=False)
class OneTurnMap:
    method: Method
    reference_orbit: np.ndarray
    map: ParticlesTpsa
    orbit: TpsaMap
    translation: TpsaMap
    inverse_translation: TpsaMap
    centred_map: TpsaMap
    centring_residual: np.ndarray


def get_one_turn_map(
    line,
    *,
    method: Method | None = None,
    particle_ref: Particles | LineParticleRef | None = None,
    co_ref: Particles | None = None,
    co_tracking_iterations: int = 1,
    retrack_around_parametric_co: bool = True,
    descriptor: Descriptor | None = None,
    verbose: bool = False,
    _lidx: MapLongitudinalIndices | None = None,
):
    if not isinstance(line, Line):
        raise ValueError("Argument `line` should be an `xt.Line`!")
    if particle_ref is None:
        if line.particle_ref is None:
            raise ValueError("A reference particle is required")
        particle_ref = line.particle_ref.copy()
    elif not isinstance(particle_ref, (Particles, LineParticleRef )):
        raise ValueError("`particle_ref` needs to be a `Particles` instance!")
    elif particle_ref._capacity != 1:
        raise ValueError("`particle_ref` needs to be a single particle!")

    _lidx = _parse_lidx(method, _lidx)

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
    if co_ref is None:
        if line.tracker is None:
            line.build_tracker()
        freeze = _lidx.rank < 6
        try:
            co_ref = line.find_closed_orbit(
                particle_ref=particle_ref,
                freeze_longitudinal=freeze
            )
        except:
            warn("Failed finding closed orbit! Trying in 4D.")
            _lidx = MapLongitudinalIndices.from_method("56d")
            co_ref = line.find_closed_orbit(
                particle_ref=particle_ref,
                freeze_longitudinal=True
            )

    elif not isinstance(co_ref, Particles):
        raise ValueError("`co_ref` needs to be a `Particles` instance!")
    elif co_ref._capacity != 1:
        raise ValueError("`co_ref` needs to be a single particle!")

    # One-turn map around numerical (parameter-free) closed orbit
    # -----------------------------------------------------------
    m = _create_map_from_co(co_ref, descriptor)
    co_ref_arr = m.const_part.copy()
    c0 = co_ref_arr.copy()
    line.track(m)
    m = m - c0
    if verbose:
        _print(f"R orbit residual:  {m.const_part}")

    # Newton correction to improve residual
    for i in range(co_tracking_iterations - 1):
        rank = _lidx.rank
        R = m.evaluate_jacobian()[:rank, :rank]
        epsilon = m.const_part[:rank]
        correction = np.linalg.solve(np.eye(rank) - R, epsilon)
        c0[:rank] += correction

        m = _create_map_from_co(co_ref, descriptor, c0)
        line.track(m)
        m = m - c0
        if verbose:
            _print(f"Corrected orbit residual (iteration {i+2}):  {m.const_part}")

    original_map = m.copy()

    # Solve for parameter-dependent closed orbit
    # ------------------------------------------
    ck = _get_parameter_orbit(m, _lidx.method)
    orbit = ck.copy()
    orbit.const_part = c0
    if verbose:
        _print(f"Orbit:  {orbit.format()}")

    # Centre!
    # -------
    forward, backward = _get_orbit_translation(ck, method)

    if retrack_around_parametric_co:
        m_retrack = _create_map_from_initial_map(co_ref, forward + c0)
        line.track(m_retrack)
        tracked = m_retrack - c0
        m = backward << tracked

    else:
        m = backward << (m << forward)

    if verbose:
        _print(f"Final residual of parametric orbit:  {m.const_part}")

    # Centre exactly around zero
    diff = m.const_part
    m.const_part = 0

    if verbose:
        _print(f"Non-symplectic coefs by degree (tracked map):")
        errors, worst = m.non_symplectic_coeffs_by_degree(norm='relative_frobenius')
        for degree, error in errors.items():
            _print(f"    {degree}: {error:.4e}    worst: {worst[degree]}")

    return OneTurnMap(
        method = _lidx.method,
        reference_orbit = c0,
        map = original_map,             # M0(z, k) = Mphys(z+c0, k) - c0
        orbit = orbit,                  # c0 + c(k)
        translation = forward,          # X
        inverse_translation = backward, # X^(-1)
        centred_map = m,       # Mcentre = X^(-1) << M0 << X
        centring_residual = diff        # Mcentre(0, 0) = diff  (should be 0)
    )


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


def _get_parameter_orbit(m, method):
    d = m.descriptor
    identity = TpsaMap(d)

    # Remove const_part (needed for inverse) and subtract identity: M'(z, k) = M(z,k) - z - c1
    shifted = m - identity
    shifted.const_part = 0

    # Root of M' is fixed-point: M( c(k), k ) = c(k) + c1
    if method == "6d":
        return shifted.inverse().substitute(0.)

    # Invert only the four transverse fixed-point equations.
    inverse = shifted.partial_inverse([True, True, True, True, False, False])

    if method == "4d":
        root = inverse.substitute(0.)

    elif method in ("5d", "56d"):
        # Let pzeta free
        root = inverse.substitute(x=0., px=0., y=0., py=0., zeta=0.)

    # The longitudinal coordinates are not orbit corrections.
    return TpsaMap([*root.components[:4], d.zero(), d.zero()])


def _get_orbit_translation(ck, method):
    I = TpsaMap(ck.descriptor)
    forward  = I + ck
    backward = I - ck

    if method != "56d":
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
