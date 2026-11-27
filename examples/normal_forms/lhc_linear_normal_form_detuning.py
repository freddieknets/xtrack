# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

from xtrack.normal_forms import LinearNormalForm

from get_lhc_lattice import generate_lhc_without_params

env, descriptor = generate_lhc_without_params()
line = env.b1

normal = LinearNormalForm.from_line(
    line,
    method='4d',
    co_tracking_iterations=1,
    retrack_around_parametric_co=True,
    symplectify=True,
    symplectify_before=False,
    parameter_scales='auto',
    coordinate_scale=2.**-10,
    verbose=True,
)

tw = normal.twiss()
