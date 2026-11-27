# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from madng_tpsa import Descriptor, TpsaMap, TpsaTensor

from .longitudinal import MapLongitudinalIndices, _parse_lidx
from .one_turn_map import (
    get_one_turn_map, OneTurnMap, _create_map_from_initial_map, _resolve_particle_ref
)
from .symplectify import symplectify as make_symplectic
from .linear import linear_normalise
from .parametric_twiss import ParametricTwiss
from ..general import _print
from ..line import Line, LineParticleRef, _preserve_track_flags

if TYPE_CHECKING:
    from collections.abc import Mapping
    from .longitudinal import Method
    from .linear import LinearNormalisation
    from ..particles import Particles


class LinearNormalForm:
    __slots__ = {
        '_longitudinal_indices',
        '_particle_ref',
        '_line',
        '_one_turn_map_result',
        '_symplectified_map',
        '_centred_map',
        '_linear_result',
    }
    _longitudinal_indices: MapLongitudinalIndices
    _particle_ref: Particles | None
    _line: Line | None
    _one_turn_map_result: OneTurnMap | None
    _centred_map: TpsaMap
    _symplectified_map: TpsaMap
    _linear_result: LinearNormalisation

    @classmethod
    def from_line(
        cls,
        line: Line,
        *,
        method: Method | None = None,
        momentum_as_parameter: bool = None,
        particle_ref: Particles | LineParticleRef | None = None,
        co_ref: Particles | None = None,
        co_tracking_iterations: int = 2,
        retrack_around_parametric_co: bool = True,
        symplectify: bool = True,
        symplectify_before: bool = False,
        max_parameter_degree: int | None = None,
        parameter_scales: Mapping[str, float] | Literal['auto'] | None = 'auto',
        coordinate_scale: float = 2.**-10,
        descriptor: Descriptor | None = None,
        verbose: bool = False,
        _lidx: MapLongitudinalIndices | None = None,
    ):
        _lidx = _parse_lidx(method, _lidx)
        particle_ref = _resolve_particle_ref(line, particle_ref)

        if verbose:
            print("Tracking one-turn map...")
        else:
            print("Tracking one-turn map...   ", end='')
        one_turn_map = get_one_turn_map(
            line=line,
            particle_ref=particle_ref,
            co_ref=co_ref,
            co_tracking_iterations=co_tracking_iterations,
            retrack_around_parametric_co=retrack_around_parametric_co,
            momentum_as_parameter=momentum_as_parameter,
            descriptor=descriptor,
            verbose=verbose,
            _lidx=_lidx
        )
        if verbose:
            print("Done tracking one-turn map.")
        else:
            print("done.")

        descriptor = one_turn_map.centred_map.descriptor

        obj = cls.from_map(
            one_turn_map.centred_map,
            symplectify=symplectify,
            symplectify_before=symplectify_before,
            max_parameter_degree=max_parameter_degree,
            parameter_scales=parameter_scales,
            coordinate_scale=coordinate_scale,
            verbose=verbose,
            _lidx=_lidx,
        )
        obj._particle_ref = particle_ref
        obj._line = line
        obj._one_turn_map_result = one_turn_map
        return obj

    @classmethod
    def from_map(
        cls,
        one_turn_map: TpsaMap,
        *,
        method: Method | None = None,
        symplectify: bool = True,
        symplectify_before: bool = False,
        max_parameter_degree: int | None = None,
        parameter_scales: Mapping[str, float] | Literal['auto'] | None = 'auto',
        coordinate_scale: float = 2.**-10,
        verbose: bool = False,
        _lidx: MapLongitudinalIndices | None = None,
    ) -> LinearNormalForm:
        """Normalise a centred map.

        symplectify controls correction after normalisation. The optional
        symplectify_before applies a separate correction to the input map.
        """
        _lidx = _parse_lidx(method, _lidx)
        _lidx.check_map(one_turn_map)

        symp_map = one_turn_map
        if symplectify_before:
            symp_map = make_symplectic(one_turn_map, _lidx=_lidx)

        linear = linear_normalise(
            symp_map,
            max_parameter_degree=max_parameter_degree,
            parameter_scales=parameter_scales,
            coordinate_scale=coordinate_scale,
            symplectify_after=symplectify,
            verbose=verbose,
            _lidx=_lidx,
        )
        obj = cls()
        obj._longitudinal_indices = _lidx
        obj._particle_ref = None
        obj._line = None
        obj._one_turn_map_result = None
        obj._centred_map = one_turn_map
        obj._symplectified_map = linear.symplectified_map
        obj._linear_result = linear
        return obj

    @property
    def particle_ref(self) -> Particles:
        return self._particle_ref

    @property
    def tracked_map(self) -> Particles:
        return self._one_turn_map_result.retracked_map or self._one_turn_map_result.tracked_map

    @property
    def method(self) -> Method:
        return self._longitudinal_indices.method

    @property
    def M(self) -> TpsaMap:
        return self._one_turn_map_result.map

    @property
    def X(self) -> TpsaMap:
        return self._one_turn_map_result.translation

    @property
    def Xinv(self) -> TpsaMap:
        return self._one_turn_map_result.inverse_translation

    @property
    def A(self) -> TpsaMap:
        return self._linear_result.normalising_map

    @property
    def Ainv(self) -> TpsaMap:
        return self._linear_result.inverse_normalising_map

    @property
    def R(self) -> TpsaMap:
        return self._linear_result.normalised_map

    @property
    def A_matrix(self) -> TpsaTensor:
        return self._linear_result.normalising_matrix

    @property
    def Ainv_matrix(self) -> TpsaTensor:
        return self._linear_result.inverse_normalising_matrix

    @property
    def R_matrix(self) -> TpsaTensor:
        return self._linear_result.normalised_matrix

    @property
    def closed_orbit(self):
        return self._one_turn_map_result.orbit

    @property
    def centred_map(self):
        return self._centred_map

    @property
    def symplectified_map(self):
        return self._symplectified_map

    @property
    def scaled_normalised_map(self) -> TpsaMap:
        """Full normalised map in the internal numerical coordinates."""
        return self._linear_result.scaled_normalised_map

    @property
    def argument_scales(self) -> dict[str, float]:
        """Original argument = scale * numerical argument."""
        return self._linear_result.argument_scales.copy()

    def twiss(self, line=None, at=None) -> ParametricTwiss:
        if line is None:
            line = self._line
        if line is None or self.tracked_map is None:
            raise ValueError("Transport requires a tracked ParticlesTpsa reference")
        particle_ref = _resolve_particle_ref(line, self.particle_ref)

        c0 = self._one_turn_map_result.reference_orbit
        initial_map = (self.X << self.A) + c0
        m = _create_map_from_initial_map(particle_ref, initial_map)

        names = list(line.element_names)
        if at is None:
            at = names
        elif isinstance(at, str):
            at = [at]
        else:
            at = list(at)
        # Avoid repeated linear searches through the lattice.
        indices = {name: i for i, name in enumerate(names)}
        at_indices = [indices[name] for name in at]
        if any(a >= b for a, b in zip(at_indices, at_indices[1:])):
            raise ValueError("Observation points must be unique and in lattice order.")

        if line.tracker is None:
            line.build_tracker()

        transported_maps = {}
        prev_index = 0
        with _preserve_track_flags(line):
            if self.method in ('4d', '5d', '56d'):
                line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = True
            for name, index in zip(at, at_indices):
                if index > prev_index:
                    line.track(m, ele_start=prev_index, ele_stop=index)
                transported_maps[name] = m.to_map().copy()
                prev_index = index

        return ParametricTwiss.from_maps(
            transported_maps,
            rank=self._longitudinal_indices.rank,
            momentum_index=self._longitudinal_indices.momentum_index,
        )
