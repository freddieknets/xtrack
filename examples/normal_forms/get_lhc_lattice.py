# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import scipy.constants as sc

from madng_tpsa import Descriptor

import xtrack as xt
from xtrack.tpsa import COORDS


def generate_lhc_without_params(ORDER=6):
    env = xt.load("https://acc-models.web.cern.ch/acc-models/lhc/2026/xsuite/lhc.json")
    env.vars.load("https://acc-models.web.cern.ch/acc-models/lhc/2026/strengths/compiled/R2025aRP_A15cmC15cmA10mL200cm_Flat.madx")
    env.vars.load("https://acc-models.web.cern.ch/acc-models/lhc/2026/scenarios/cycle/pp/levelling/1050/settings.madx")
    env.new_particle("beam1", mass0=938272046.0, q0=1, p0c="p0c")
    env.b1.set_particle_ref("beam1")
    env.new_particle("beam2", mass0=938272046.0, q0=1, p0c="p0c")
    env.b2.set_particle_ref("beam2")

    # Crossing angle and separation to create closed orbit
    env['on_x1_h'] = 150    # urad
    env['on_x5_v'] = -150   # urad
    env['on_sep1_v'] = 1.25  # mm
    env['on_sep5_h'] = 1.25  # mm

    descriptor = Descriptor(var_names=COORDS, order=ORDER)

    return env, descriptor


def generate_lhc_with_params(param_type, ORDER=6):
    match param_type:
        case 'trim':
            param_names = ['dkqt.12r1']
            PARAM_ORDER = 4
        case 'tune':
            param_names = ['qx', 'qy']
            PARAM_ORDER = 2
        case 'chrom':
            param_names = ['chrom_x', 'chrom_y']
            PARAM_ORDER = 2
        case 'oct':
            param_names = ['dkod.a12b1', 'i_mo']
            PARAM_ORDER = 2
        case _:
            raise ValueError("Choose `param_type` from ['trim', 'tune', 'chrom', 'oct']")

    env, _ = generate_lhc_without_params()
    descriptor = Descriptor(var_names=COORDS, order=ORDER, param_names=param_names, param_order=PARAM_ORDER)

    match param_type:
        case 'trim':
            # Adjust one specific trim quadrupole
            env['dkqt.12r1'] = 0
            env.ref['mqt.12r1.b1'].k1 += env.ref['dkqt.12r1']
            env.vars['dkqt.12r1'] = descriptor.param('dkqt.12r1')  # Promote the variable to a TPSA parameter

        case 'tune':
            tw = env.b1.twiss()
            env.vars['dqx.b1_op'] = descriptor.param('qx', tw.qx) - tw.qx  # parameter with constant part equal to current expected value
            env.vars['dqy.b1_op'] = descriptor.param('qy', tw.qy) - tw.qy

        case 'chrom':
            env.vars['dqpx.b1_op'] = descriptor.param('chrom_x', env['dqpx.b1_op'])  # parameter with constant part equal to current knob value
            env.vars['dqpy.b1_op'] = descriptor.param('chrom_y', env['dqpy.b1_op'])

        case 'oct':
            brho = env['p0c'] / sc.c
            for beam in [1, 2]:
                env.vars[f'i_oct_b{beam}'] = descriptor.param('i_mo')
                env.vars[f'i_mo.b{beam}']  = env.vars[f'i_oct_b{beam}']
                for i in range(1,9):
                    env.vars[f'kof.a{i}{i%8+1}b{beam}'] = env.ref['kmax_mo'] * env.ref[f'i_mo.b{beam}'] / env.ref['imax_mo'] / brho
                    env.vars[f'kod.a{i}{i%8+1}b{beam}'] = env.ref['kmax_mo'] * env.ref[f'i_mo.b{beam}'] / env.ref['imax_mo'] / brho
            env.vars['dkod.a12b1'] = descriptor.param('dkod.a12b1', 0.0)
            env.vars['kod.a12b1'] += env.vars['dkod.a12b1']

    return env, descriptor
