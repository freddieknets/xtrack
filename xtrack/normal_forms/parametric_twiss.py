# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from madng_tpsa import TpsaMap


def extract_linear_family(transported_map, rank: int, momentum_index: int | None = None):
    """Return orbit(k), W(k)=dT/du at u=0. Retain external parameters,
    and for coasting systems retain the conserved momentum as an independent
    variable. No nonlinear-amplitude dependence remains in W.
    """
    d = transported_map.descriptor
    nv = transported_map.num_vars
    projection = TpsaMap([
        d.var(i + 1) if i == momentum_index else d.zero()
        for i in range(nv)
    ])
    orbit_map = transported_map << projection
    jac = transported_map.jacobian() << projection
    orbit = [orbit_map.flat[i] for i in range(nv)]
    w = np.empty((rank, rank), dtype=object)
    for i in range(rank):
        for j in range(rank):
            w[i, j] = jac[i, j]
    return orbit, w


def _twiss_matrices(w):
    """Mais-Ripken contributions per physical plane and eigenmode."""
    rank, _, count = w.shape
    modes = rank // 2
    alpha = np.empty((modes, modes, count), dtype=object)
    beta = np.empty_like(alpha)
    gamma = np.empty_like(alpha)
    for i in range(modes):
        for j in range(modes):
            x, p = 2 * i, 2 * i + 1
            u, v = 2 * j, 2 * j + 1
            for s in range(count):
                xu, xv = w[x, u, s], w[x, v, s]
                pu, pv = w[p, u, s], w[p, v, s]
                alpha[i, j, s] = -(xu * pu + xv * pv)
                beta[i, j, s] = xu * xu + xv * xv
                gamma[i, j, s] = pu * pu + pv * pv
    return alpha, beta, gamma


def _parametric_phases(w):
    """Continuous nominal phase, TPSA perturbation about each nominal phase.

    Fix each mode's gauge using its physical position coordinate (x, y, zeta).
    This requires a nonzero projection of the mode into that coordinate at
    every observation point. For strongly coupled systems, a more sophisticated
    continuously transported phase gauge is needed.
    """
    rank, _, count = w.shape
    nm = rank // 2
    out = np.empty((nm, count), dtype=object)
    for j in range(nm):
        row, col = 2 * j, 2 * j
        a0, b0 = w[row, col, 0], w[row, col + 1, 0]
        principal = np.empty(count, dtype=float)
        raw = []
        for s in range(count):
            a, b = w[row, col, s], w[row, col + 1, s]
            real = a * a0 + b * b0
            imag = b * a0 - a * b0
            principal[s] = np.arctan2(float(imag.const_part), float(real.const_part))
            raw.append((real, imag))
        unwrapped = np.unwrap(principal)
        for s, (real, imag) in enumerate(raw):
            theta0 = principal[s]
            ct, st = np.cos(theta0), np.sin(theta0)
            # atan2(imag, real) expanded about the nominal complex value.
            # The numerator has zero constant coefficient and the denominator
            # is positive at the reference parameters.
            numerator = imag * ct - real * st
            denominator = real * ct + imag * st
            if float(denominator.const_part) <= 1e-24:
                raise ValueError(
                    f"Mode {j} projection vanishes at location {s}; "
                    "cannot extract a phase from this coordinate."
                )
            out[j, s] = real.descriptor.constant(float(unwrapped[s])) + np.arctan(
                numerator / denominator
            )
    return out


@dataclass(slots=True)
class ParametricTwiss:
    names: tuple[str, ...]
    orbit: np.ndarray                 # (map_size, nlocations), Tpsa
    W_matrix: np.ndarray              # (rank, rank, nlocations), Tpsa
    alpha_matrix: np.ndarray          # (physical_plane, mode, nlocations), Tpsa
    beta_matrix: np.ndarray
    gamma_matrix: np.ndarray
    phase_advance: np.ndarray         # (mode, nlocations), Tpsa, in turns
    _phase_matrix: np.ndarray | None = field(default=None, init=False, repr=False)
    _A_matrix: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_maps(cls, transported_maps, *, rank=6, momentum_index=None):
        """Build from an ordered dict of maps, with the start map first.

        The names correspond to the start of the element when maps were saved.
        """
        names = tuple(transported_maps)
        if not names:
            raise ValueError("No transported maps")
        count = len(names)
        nv = next(iter(transported_maps.values())).num_vars
        orbit = np.empty((nv, count), dtype=object)
        w = np.empty((rank, rank, count), dtype=object)
        for s, name in enumerate(names):
            oo, ww = extract_linear_family(
                transported_maps[name],
                rank=rank,
                momentum_index=momentum_index,
            )
            for i in range(nv):
                orbit[i, s] = oo[i]
            w[:, :, s] = ww
        alpha, beta, gamma = _twiss_matrices(w)
        phases = _parametric_phases(w) / (2 * np.pi)
        return cls(names, orbit, w, alpha, beta, gamma, phases)

    @property
    def mux(self):
        return self.phase_advance[0]

    @property
    def muy(self):
        return self.phase_advance[1]

    @property
    def muzeta(self):
        return self.phase_advance[2] if self.phase_advance.shape[0] == 3 else None

    @property
    def tunes(self):
        """Unwrapped phase advance at last observation in turns."""
        return self.phase_advance[:, -1]

    @property
    def phase_matrix(self):
        """Block-diagonal rotation from start to each observation point."""
        if self._phase_matrix is None:
            rank, _, count = self.W_matrix.shape
            rot = np.empty((rank, rank, count), dtype=object)
            for s in range(count):
                d = self.W_matrix[0, 0, s].descriptor
                for i in range(rank):
                    for j in range(rank):
                        rot[i, j, s] = d.zero()
                for k in range(rank // 2):
                    mu = 2 * np.pi * self.phase_advance[k, s]
                    c, ss = np.cos(mu), np.sin(mu)
                    i = 2 * k
                    rot[i, i, s] = c
                    rot[i, i + 1, s] = ss
                    rot[i + 1, i, s] = -ss
                    rot[i + 1, i + 1, s] = c
            self._phase_matrix = rot
        return self._phase_matrix

    @property
    def A_matrix(self):
        """Local phase-gauged normalisation W(s) @ phase_matrix(s)^-1."""
        if self._A_matrix is None:
            rank, _, count = self.W_matrix.shape
            out = np.empty_like(self.W_matrix)
            rotations = self.phase_matrix
            for s in range(count):
                for k in range(rank // 2):
                    j = 2 * k
                    c = rotations[j, j, s]
                    ss = rotations[j, j + 1, s]
                    for i in range(rank):
                        u = self.W_matrix[i, j, s]
                        v = self.W_matrix[i, j + 1, s]
                        out[i, j, s] = u * c + v * ss
                        out[i, j + 1, s] = -u * ss + v * c
            self._A_matrix = out
        return self._A_matrix
