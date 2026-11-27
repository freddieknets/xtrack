# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
from warnings import warn
from typing import TYPE_CHECKING

from madng_tpsa import Descriptor, TpsaMap, TpsaTensor

from .longitudinal import MapLongitudinalIndices, _parse_lidx
from .one_turn_map import get_one_turn_map, OneTurnMap
from .translate import translate_to_centre
from .linear import linear_normalise
from .parametric_twiss import ParametricTwiss
from ..general import _print
from ..line import Line
from ..tpsa import COORDS, ParticlesTpsa

if TYPE_CHECKING:
    from collections.abc import Sequence
    from .translate import Centring
    from .longitudinal import Method
    from .linear import LinearNormalisation
    from ..particles import Particles
    from ..line import Line, LineParticleRef


class LinearNormalForm:
    __slots__ = {
        '_longitudinal_indices',
        '_particle_ref',
        '_line',
        '_one_turn_map',
        '_linear_result',
    }
    _longitudinal_indices: MapLongitudinalIndices
    _particle_ref: Particles | None
    _line: Line | None
    _one_turn_map: OneTurnMap
    _linear_result: LinearNormalisation

    @classmethod
    def from_line(
        cls,
        line: Line,
        *,
        method: Method | None = None,
        particle_ref: Particles | LineParticleRef | None = None,
        co_ref: Particles | None = None,
        co_tracking_iterations: int = 2,
        retrack_around_parametric_co: bool = True,
        max_parameter_degree: int | None = None,
        momentum_as_parameter: bool = None,
        descriptor: Descriptor | None = None,
        verbose: bool = False,
        _lidx: MapLongitudinalIndices | None = None,
    ):
        _lidx = _parse_lidx(method, _lidx)
        if verbose:
            print("Tracking one-turn map...")
        else:
            print("Tracking one-turn map...   ", end='')
        local_one_turn_map = get_one_turn_map(
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
        descriptor = local_one_turn_map.centred_map.descriptor

        obj = cls.from_map(
            local_one_turn_map.centred_map,
            max_parameter_degree=max_parameter_degree,
            _lidx=_lidx,
        )

        obj._particle_ref = particle_ref
        obj._line = line
        return obj

    @classmethod
    def from_map(
        cls,
        one_turn_map: TpsaMap,
        *,
        method: Method | None = None,
        max_parameter_degree: int | None = None,
        _lidx: MapLongitudinalIndices | None = None,
    ) -> LinearNormalForm:
        _lidx = _parse_lidx(method, _lidx)
        # Need symplecticity checks + Healy make symplectic
        _lidx.check_map(one_turn_map)

        # TODO: check centring + centre if needed

        linear = linear_normalise(
            one_turn_map,
            max_parameter_degree=max_parameter_degree,
            _lidx=_lidx,
        )
        obj = cls()
        obj._longitudinal_indices = _lidx
        obj._one_turn_map = one_turn_map
        # obj._centring_result = centring
        obj._linear_result = linear
        obj._particle_ref = None
        obj._line = None
        return obj

    @property
    def particle_ref(self) -> Particles:
        return self._particle_ref

    @property
    def tracked_map(self) -> Particles:
        return self._one_turn_map.retracked_map or self._one_turn_map.tracked_map

    @property
    def method(self) -> Method:
        return self._longitudinal_indices.method

    @property
    def M(self) -> TpsaMap:
        return self._one_turn_map.map

    @property
    def X(self) -> TpsaMap:
        return self._one_turn_map.translation

    @property
    def Xinv(self) -> TpsaMap:
        return self._one_turn_map.inverse_translation

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
        return self._one_turn_map.orbit

    @property
    def centred_map(self):
        return self._one_turn_map.centred_map

    @property
    def twiss(self, line=None, at=None) -> ParametricTwiss:
        if line is None:
            line = self._line
        if line is None or self.tracked_map is None:
            raise ValueError("Transport requires a tracked ParticlesTpsa reference")
        initial_map = self.X << self.A
        kwargs = self.tracked_map.to_particles().to_dict()
        kwargs.update({name: getattr(initial_map, name) for name in COORDS})
        m = ParticlesTpsa(**kwargs)  # Descriptor is taken from TPSAs
        if at is None:
            at = list(line.element_names)
        transported_maps = {}
        prev_element = None
        for name in at:
            line.track(m, ele_start=prev_element, ele_stop=name)
            transported_maps[name] = m.to_map().copy()
            prev_element = name
        return ParametricTwiss.from_maps(
            transported_maps,
            rank=self._longitudinal_indices.rank,
            momentum_index=self._longitudinal_indices.momentum_index,
        )
