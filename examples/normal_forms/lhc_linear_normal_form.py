# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from xtrack.normal_forms import LinearNormalForm
from xtrack.normal_forms.inspect import inspect_symplectic_defect, inspect_linear_family

from get_lhc_lattice import generate_lhc_with_params

env, descriptor = generate_lhc_with_params('trim')
line = env.b1

normal = LinearNormalForm.from_line(
    line,
    method='6d',
    co_tracking_iterations=1,
    retrack_around_parametric_co=True,
    symplectify=True,
    symplectify_before=False,
    parameter_scales='auto',
    coordinate_scale=2.**-10,
    verbose=True,
)

M = normal.M
Mc = normal.centred_map
Ms = normal.symplectified_map
lidx = normal._longitudinal_indices
P = normal.tracked_map
c = normal.closed_orbit
X = normal.X
Xinv = normal.Xinv
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

print("Inspection:")
print(f"Argument scales: {normal.argument_scales}")
inspect_symplectic_defect(normal.scaled_normalised_map, lidx)
inspect_linear_family(normal.scaled_normalised_map, lidx)

tw = normal.twiss()
