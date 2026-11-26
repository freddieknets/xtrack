# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from xtrack.normal_forms import get_one_turn_map


from get_lhc_lattice import generate_lhc_with_params

env, descriptor = generate_lhc_with_params('trim')
line = env.b1

m1 = get_one_turn_map(
    line,
    co_tracking_iterations=1,
    retrack_around_parametric_co=True,
    verbose=True
)
