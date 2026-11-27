import numpy as np
import pytest

from madng_tpsa import Descriptor, TpsaMap, symplectic_unit_matrix
from xtrack.normal_forms import LinearNormalForm, MapLongitudinalIndices
from xtrack.normal_forms.linear import HomologicalSolver, rescale_map, _jacobian_family
from xtrack.normal_forms.inspect import max_coefficient


@pytest.mark.parametrize('method', ['4d', '5d', '56d', '6d'])
@pytest.mark.parametrize('before', [False, True])
def test_scaled_post_symplectification(method, before):
    lidx = MapLongitudinalIndices.from_method(method)
    d = Descriptor(var_names=lidx.var_names, order=4,
                   param_names=['k'], param_order=3)
    z = list(d.vars())
    x, px = z[:2]
    k = d.param('k')
    p = z[lidx.momentum_index] if lidx.momentum_index is not None else d.zero()
    kick = list(z)
    kick[1] = px - (k + p) * x - (1 + k + p) * x**2
    if method == '56d':
        kick[lidx.time_index] = z[lidx.time_index] + x**2 / 2 + x**3 / 3
    rotation = list(z)
    for i, mu in zip(range(0, lidx.rank, 2), (0.3, 0.7, 0.12)):
        c, s = float(np.cos(mu)), float(np.sin(mu))
        rotation[i] = c*z[i] + s*z[i+1]
        rotation[i+1] = -s*z[i] + c*z[i+1]
    m = TpsaMap(rotation) << TpsaMap(kick)
    m[0] = m[0] + 1e-4 * (1 + k) * x**2
    normal = LinearNormalForm.from_map(
        m, method=method, symplectify_before=before,
        parameter_scales={'k': 2.**-7, **({'pzeta': 2.**-9}
                          if lidx.momentum_index is not None else {})},
    )
    scaled = normal.scaled_normalised_map
    lidx.check_map(scaled, tol=1e-12)
    assert max_coefficient(lidx.symplectic_defect(scaled)) < 1e-11
    assert max_coefficient(lidx.symplectic_defect(normal.symplectified_map)) < 1e-8
    restored = rescale_map(scaled, normal.argument_scales, inverse=True)
    assert max_coefficient(restored - normal.R) == 0.
    reconstructed = normal.A << (normal.R << normal.Ainv)
    assert max_coefficient(reconstructed - normal.symplectified_map) < 1e-8
    assert normal.centred_map is m
    family = _jacobian_family(scaled, lidx)[:lidx.rank, :lidx.rank]
    assert np.all(np.isfinite(family.const_part))


def test_hamiltonian_projection_guard():
    mu = .3
    c, s = np.cos(mu), np.sin(mu)
    R = np.array([[c, s], [-s, c]])
    S = symplectic_unit_matrix(2)
    solver = HomologicalSolver(R)
    q = np.array([[.2, .3], [.4, -.2]])
    K, H = solver.solve(q + 1e-12 * np.eye(2))
    np.testing.assert_allclose(K.T @ S + S @ K, 0., atol=1e-12)
    np.testing.assert_allclose(q + R @ K @ R.T - K, H, atol=1e-12)
    assert solver.max_discarded > 0.
    with pytest.raises(ValueError, match='not sufficiently Hamiltonian'):
        solver.solve(q + 1e-3 * np.eye(2))


def test_automatic_parameter_scaling():
    from xtrack.normal_forms.linear import infer_parameter_scales
    from madng_tpsa import TpsaTensor
    d = Descriptor(var_names=['x', 'px', 'y', 'py'], order=4,
                   param_names=['k', 'h'], param_order=3)
    k, h = d.params()
    family = TpsaTensor([[1 + 100*k + 1e6*k*h, 0*k], [0*k, 1 + 10*h]])
    lidx = MapLongitudinalIndices.from_method('4d')
    scales = infer_parameter_scales(family, lidx, 3)
    assert 100 * scales['k'] <= 1e-2
    assert 10 * scales['h'] <= 1e-2
    assert 1e6 * scales['k'] * scales['h'] <= 1e-2
