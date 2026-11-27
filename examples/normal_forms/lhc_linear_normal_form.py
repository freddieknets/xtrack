# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from xtrack.normal_forms import LinearNormalForm

from get_lhc_lattice import generate_lhc_with_params

# radii = {
#     'x':          1e-3,
#     'px':         1e-5,
#     'y':          1e-3,
#     'py':         1e-5,
#     'zeta':       1e-1,
#     'pzeta':      1e-3,
#     'dkqt.12r1':  1e-4,
#     'dksd.b1':    1e-2,
#     'dksf.b1':    1e-2,
#     'dkod.a12b1': 1e-1,
#     'i_mo':       100.,
# }

# radii = {
#     name: radii[name]
#     for name in m.descriptor.var_names + m.descriptor.param_names
# }

# m_symp = healy_symplectify(
#     m,
#     method='56d',
#     radii=radii,
# )
from xtrack.normal_forms import LinearNormalForm
from get_lhc_lattice import generate_lhc_with_params

env, descriptor = generate_lhc_with_params('trim')
line = env.b1

normal = LinearNormalForm.from_line(
    line,
    method='6d',
    co_tracking_iterations=1,
    retrack_around_parametric_co=True,
    verbose=False
)

M = normal.M
Mc = normal.centred_map
Ms = normal.symplectified_map
lidx = normal._longitudinal_indices
P = normal.tracked_map
c = normal.closed_orbit
X = normal.translation
Xinv = normal.inverse_translation
A = normal.A
Ainv = normal.Ainv

print("Parametric orbit:")
print(c.format())

print("Parametric residual of centred map (i.e. Mc(0, k) which should vanish ∀k):")
print(M.substitute(0.).format()) # This substitutes the coordinates but not the parameters
# If these are not zero and verified to be just noise, we can remove them:
# residual = m.substitute(0.)
# residual[:rank] = 0  # for 4D, 5D, and 56D
# m = m - residual

print("Verifying composition, (X^(-1) << (M << X)) - Mc:")
difference = (Xinv << (M << X)) - Mc
print(difference.format())
