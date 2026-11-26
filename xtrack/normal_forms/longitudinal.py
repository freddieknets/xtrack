# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from madng_tpsa import Descriptor, symplectic_unit_matrix, TpsaMap

from ..tpsa import COORDS


Method = Literal['4d', '5d', '56d', '6d']


def _parse_lidx(
    method: Method | None = None,
    _lidx: MapLongitudinalIndices | None = None
) -> MapLongitudinalIndices:
    if _lidx is None:
        # 56D means:
        #   x, px, y, py do NOT depend on ζ, pζ
        #   pζ is constant
        #   ζ transforms as ζ + f(x, px, y, py, pζ)
        # TODO: Auto-recognise method!
        if method is None:
            method = '6d'
        _lidx = MapLongitudinalIndices.from_method(method)
    elif method is not None:
        raise ValueError("Cannot provide both `method` and `_lidx`!")
    return _lidx

@dataclass(slots=True, frozen=True, eq=False)
class MapLongitudinalIndices:
    """Lightweight class to store the different longitudinal modes.

    4D  (x, px, y, py)	       Two oscillatory modes
    5D  (x, px, y, py, pζ)	   Momentum retained, longitudinal position omitted
    56D (x, px, y, py, ζ, pζ)  Full canonical map, coasting momentum (still only two oscillatory modes)
    6D  (x, px, y, py, ζ, pζ)  Three oscillatory modes, e.g. synchrotron motion
    """
    rank: int
    time_index: int | None
    momentum_index: int | None
    map_size: int
    method: str

    @classmethod
    def from_method(cls, method: Method | None) -> MapLongitudinalIndices:
        if method is None:
            method = '6d'
        match method:
            case '4d':  return cls(4, None, None, 4, method)
            case '5d':  return cls(4, None,    4, 5, method)
            case '56d': return cls(4,    4,    5, 6, method)
            case '6d':  return cls(6, None, None, 6, method)
            case _: raise ValueError(f'Unknown longitudinal mode: {method!r}')

    @property
    def _selector(self):
        if self.map_size == 5:
            return (0, 1, 2, 3, 5)
        return tuple(range(self.map_size))

    @property
    def var_names(self):
        return tuple(COORDS[i] for i in self._selector)

    def move_descriptor(
        self,
        descriptor: Descriptor,
        *,
        momentum_as_parameter: bool = False
    ) -> Descriptor:
        param_names = descriptor.param_names
        max_orders = tuple(descriptor.max_orders[i] for i in self._selector)
        max_orders += descriptor.max_orders[descriptor.num_vars:]
        if momentum_as_parameter:
            if self.method != '4d':
                raise ValueError("Can only use momentum as parameter in 4D.")
            param = descriptor.var_names[5]
            if param in param_names:
                raise ValueError(
                    f"Cannot use momentum {param} as parameter: already a parameter!"
                )
            param_names += (param,)
            max_orders += (descriptor.max_orders[5],)
        return Descriptor(
            var_names=self.var_names,
            order=descriptor.order,
            param_names=param_names,
            max_orders=max_orders,
        )

    def adapt_6d_map(
        self,
        m6: TpsaMap,
        *,
        momentum_as_parameter: bool = False
    ) -> TpsaMap:
        """
        Reduce a validated 6D/56D map to 4D or 5D.

        For 4D, optionally turn the conserved pzeta
        variable into an external TPSA parameter.
        """
        if self.method == "6d" or self.method == '56d':
            return m6

        src = m6.descriptor
        dst = self.move_descriptor(src, momentum_as_parameter=momentum_as_parameter)
        projection = TpsaMap(src)
        projection[4].clear()      # Remove zeta
        if self.method == "4d":
            projection[5].clear()  # Remove pzeta
        result = TpsaMap([
            tpsa.change_descriptor(dst)
            for nn, tpsa in m6.compose(projection).named_components.items()
            if nn in self.var_names
        ])
        if self.method == "4d" and momentum_as_parameter:
            projection = TpsaMap(src)
            projection[4].clear()  # Remove zeta from reference (we don't want to keep coeffs for ζ pζ etc)
            ref = m6.compose(projection)
            for nn, tpsa in result.named_components.items():
                for monomial, value in getattr(ref, nn).monomial_coeffs(tol=0).items():
                    if monomial[5] > 0:
                        # monomial reference: (x, pₓ, y, pᵧ, ζ, pζ, param1, ..., param_n)
                        # monomial result:    (x, pₓ, y, pᵧ,        param1, ..., param_n, pζ)
                        tpsa.set(monomial[:4] + monomial[6:] + (monomial[5],), value)
        return result

    def check_map(self, m: TpsaMap, *, tol=1e-11):
        if m.num_vars != self.map_size:
            raise ValueError(f"{self.method} requires {self.map_size} variables; got {m.num_vars}")
        if self.method == "56d":
            self.check_56d_map(m, tol=tol)
        elif self.method == "5d":
            self.check_5d_map(m, tol=tol)

        # For 4D and 6D, dimension and symplecticity
        # are the main structural checks.

    def check_5d_map(self, m: TpsaMap, *, tol: float = 1e-11):
        if m.num_vars != 5:
            raise ValueError("5D requires five variables")
        error = m.flat[self.momentum_index] - m.descriptor.var(self.momentum_index + 1)
        if any(abs(v) > tol for v in error.monomial_coeffs(tol=0).values()):
            raise ValueError("5D requires conserved pzeta")

    def check_56d_map(self, m: TpsaMap, *, tol: float = 1e-11):
        if m.num_vars != 6:
            raise ValueError("56D requires six variables")
        d = m.descriptor
        pzeta = d.var(self.momentum_index + 1)
        errors = {
            "pzeta conservation": m.flat[self.momentum_index] - pzeta,
            "zeta translation": m.flat[self.time_index].derivative(self.time_index + 1) - 1,
        }
        for i in range(self.rank):
            errors[f"zeta dependence of component {i}"] = m.flat[i].derivative(self.time_index + 1)
        for name, error in errors.items():
            coeffs = error.monomial_coeffs(tol=0)
            largest = max((abs(v) for v in coeffs.values()), default=0.0)
            if largest > tol:
                raise ValueError(
                    f"Invalid 56D map: {name}, "
                    f"largest residual coefficient={largest:.3e}"
                )

    def check_symplecticity(self, m: TpsaMap, tol: float = 0):
        nn = self.map_size if self.method == '56d' else self.rank
        J = m.jacobian()[:nn, :nn]
        S = symplectic_unit_matrix(nn)
        defect = (J.T @ S @ J - S).truncate(m.truncation_order - 1)
        # print(f"Non-symplecitc coefs by degree: {m.non_symplectic_coeffs_by_degree(norm='relative_frobenius')}")
        # TODO: need a test/assert
        return defect
