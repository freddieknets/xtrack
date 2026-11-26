# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from xtrack.normal_forms import get_one_turn_map


from get_lhc_lattice import generate_lhc_with_params

env, descriptor = generate_lhc_with_params('trim')
line = env.b1

one_turn_map = get_one_turn_map(
    line,
    method='4d',
    co_tracking_iterations=1,
    retrack_around_parametric_co=True,
    verbose=False
)

M0 = one_turn_map.map
M = one_turn_map.centred_map
P = one_turn_map.retracked_map or one_turn_map.tracked_map
c = one_turn_map.orbit
X = one_turn_map.translation
Xinv = one_turn_map.inverse_translation

print("Parametric orbit:")
print(c.format())

print("Parametric residual of centred map (i.e. Mc(0, k) which should vanish ∀k):")
print(M.substitute(0.).format()) # This substitutes the coordinates but not the parameters
# If these are not zero and verified to be just noise, we can remove them:
# residual = m.substitute(0.)
# residual[:rank] = 0  # for 4D, 5D, and 56D
# m = m - residual

print("Verifying composition, (X^(-1) << (M0 << X)) - Mc:")
difference = (Xinv << (M0 << X)) - M
print(difference.format())
