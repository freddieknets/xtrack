import numpy as np
import pytest

from madng_tpsa import Descriptor, TpsaMap

from xtrack.tpsa import COORDS
from xtrack.linear_normal_form import healy_symplectify
from xtrack.normal_forms import MapLongitudinalIndices, symplectify
from xtrack.normal_forms.symplectify import nominal_symplecticity_error, domain_symplecticity_error


def max_coefficient(tensor):
    return max(
        (abs(c) for t in tensor.flat for c in t.monomial_coeffs(tol=0.).values()),
        default=0.,
    )


@pytest.mark.parametrize('method', ['4d', '5d', '56d', '6d'])
def test_healy_symplectify(method):

    lidx = MapLongitudinalIndices.from_method(method)

    param_names = ['k', 'pzeta'] if method == '4d' else ['k']
    d = Descriptor(var_names=lidx.var_names, order=4, param_names=param_names, param_order=2)

    components = list(d.vars())
    x, px, _, _ = components[:4]
    k = d.param('k')

    if method == '4d':
        pzeta = d.param('pzeta')
    else:
        pzeta = d.var('pzeta')

    components[1] = px - (1 + k + pzeta) * x**2

    if method in ('56d', '6d'):
        components[4] = d.var('zeta') + x**3 / 3

    exact = TpsaMap(components)

    # Test 1: the reference map is already symplectic.
    defect = lidx.symplectic_defect(exact)
    assert max_coefficient(defect) < 1e-12

    # Test 2: symplectification should leave it unchanged.
    unchanged = symplectify(exact, method=method)

    assert max_coefficient(unchanged - exact) < 1e-12

    # Test 3: deliberately introduce a parametric,
    # nonlinear, non-symplectic perturbation.
    broken = exact.copy()
    broken[0] = broken[0] + 1e-4 * (1 + k) * x**2

    defect = lidx.symplectic_defect(broken)
    assert max_coefficient(defect) > 1e-5

    corrected = symplectify(broken, method=method)

    defect = lidx.symplectic_defect(corrected)
    assert max_coefficient(defect) < 1e-10

    # Test 4: idempotency.
    corrected_again = symplectify(corrected, method=method)
    assert max_coefficient(corrected_again - corrected) < 1e-10

    # Test 5: longitudinal structure.
    lidx.check_map(corrected, tol=1e-12)


def test_healy_linear_consistency():
    d = Descriptor(var_names=COORDS, order=3)

    identity = TpsaMap(d.vars())

    M = np.array([
        [ 0.80,  0.61,  0.,    0.,    0.,    0.],
        [-0.60,  0.80,  0.,    0.,    0.,    0.],
        [ 0.,    0.,    0.60,  0.80,  0.,    0.],
        [ 0.,    0.,   -0.80,  0.60,  0.,    0.],
        [ 0.,    0.,    0.,    0.,    0.99,  0.10],
        [ 0.,    0.,    0.,    0.,   -0.10,  0.99],
    ])

    m = TpsaMap([sum(float(M[i, j]) * identity[j] for j in range(6)) for i in range(6)])

    corrected = symplectify(m, method='6d', force=True)
    np.testing.assert_allclose(
        corrected.evaluate_jacobian(),
        healy_symplectify(M),
        atol=1e-12,
        rtol=0.,
    )

def test_healy_nonlinear_guard():
    d = Descriptor(var_names=('x', 'px', 'y', 'py'), order=4)
    x, px, y, py = d.vars()
    m = TpsaMap([x + 0.1 * x**2, px, y, py])
    lidx = MapLongitudinalIndices.from_method('4d')

    # The nominal matrix is the identity.
    assert nominal_symplecticity_error(m, lidx) == 0.

    radii = dict.fromkeys(d.var_names, 1.)
    # The nominal guard alone would not detect the defect.
    with pytest.raises(ValueError, match='domain-weighted'):
        symplectify(m, method='4d', radii=radii)

    # Forced symplectification should still work.
    corrected = symplectify(m, method='4d', radii=radii, force=True)
    assert domain_symplecticity_error(corrected, lidx, radii) < 1e-12

def test_healy_damping_guard():
    d = Descriptor(var_names=('x', 'px', 'y', 'py'), order=4)
    m = TpsaMap([0.99 * z for z in d.vars()])

    with pytest.raises(ValueError, match='nominal defect'):
        symplectify(m, method='4d')

    with pytest.warns(RuntimeWarning, match='nominal defect'):
        unchanged = symplectify(m, method='4d', on_large_defect='warn')

    assert max_coefficient(unchanged - m) == 0.


def test_healy_small_defect():
    d = Descriptor(var_names=COORDS, order=3)
    lidx = MapLongitudinalIndices.from_method('6d')
    identity = TpsaMap(d.vars())

    symp_tol = 1e-12
    damp_tol = 1e-8

    # Construct an initially symplectic rotation matrix.
    M = np.eye(6)
    for i, mu in enumerate([0.3, 0.4, 0.01]):
        c, s = np.cos(mu), np.sin(mu)
        M[2*i:2*i+2, 2*i:2*i+2] = [[c, s], [-s, c]]

    # Introduce a small non-symplectic perturbation.
    M[0, 0] += 1e-10
    m = TpsaMap([sum(float(M[i, j]) * identity[j] for j in range(6)) for i in range(6)])

    # Verify that the defect lies within the correction window.
    error_before = nominal_symplecticity_error(m, lidx)
    assert symp_tol < error_before < damp_tol

    # Symplectify without bypassing the guard.
    corrected = symplectify(m, method='6d', symp_tol=symp_tol, damp_tol=damp_tol)

    # Verify that the correction was applied successfully.
    error_after = nominal_symplecticity_error(corrected, lidx)
    assert error_after < symp_tol
    assert error_after < error_before

    # Verify consistency with the original linear algorithm.
    np.testing.assert_allclose(corrected.evaluate_jacobian(), healy_symplectify(M), atol=symp_tol, rtol=0.)

    # Verify full map symplecticity.
    assert max_coefficient(lidx.symplectic_defect(corrected)) < symp_tol

    # Verify that the correction itself remains small.
    correction = max_coefficient(corrected - m)
    assert symp_tol < correction < damp_tol

    # Verify that the input map was not modified.
    np.testing.assert_array_equal(m.evaluate_jacobian(), M)
