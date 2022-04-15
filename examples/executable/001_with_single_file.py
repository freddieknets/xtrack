import os
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

class LineMetaData(xo.Struct):
    ele_offsets = xo.Int64[:]
    ele_typeids = xo.Int64[:]

class SimState(xo.Struct):
    particles = xp.Particles.XoStruct
    i_turn = xo.Int64
    size = xo.Int64

class SimConfig(xo.Struct):
    buffer_size = xo.Int64
    line_metadata = xo.Ref(LineMetaData)
    num_turns = xo.Int64
    sim_state = xo.Ref(SimState)

simbuf = xo.ContextCpu().new_buffer()

sim_config = SimConfig(buffer_size=-1, _buffer=simbuf)

# Simulation input
num_turns = 10

line = xt.Line(elements=[
    xt.Drift(length=1.0), xt.Multipole(knl=[1e-6]), xt.Drift(length=1.0)])
tracker = xt.Tracker(line=line, _buffer=simbuf)

particles = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12, x=[1e-3,2e-3,3e-3])

# Assemble data structure
line_metadata = LineMetaData(_buffer=simbuf,
                             ele_offsets=tracker.ele_offsets_dev,
                             ele_typeids=tracker.ele_typeids_dev)

sim_state = SimState(_buffer=simbuf, particles=particles._xobject, i_turn=0)
sim_config.line_metadata = line_metadata
sim_config.num_turns = num_turns
sim_config.sim_state = sim_state
sim_state.size = sim_state._size # store size of sim_state

assert sim_config._offset == 0
assert sim_config._fields[0].offset == 0
assert sim_config._fields[0].name == 'buffer_size'

sim_config.buffer_size = simbuf.capacity

# Write sim buffer to file
with open('sim.bin', 'wb') as fid:
    fid.write(simbuf.buffer.tobytes())

# Generate C executable
if isinstance(simbuf.context, xo.ContextCpu):
    with open('simconfig.h', 'w') as fid:
        fid.write(xo.specialize_source(LineMetaData._gen_c_api(),
                                       specialize_for='cpu_serial'))
        fid.write('\n')
        fid.write(xo.specialize_source(SimState._gen_c_api(),
                                       specialize_for='cpu_serial'))
        fid.write('\n')
        fid.write(xo.specialize_source(SimConfig._gen_c_api(),
                                       specialize_for='cpu_serial'))
else:
    raise NotImplementedError

with open('xtrack.h', 'w') as fid:
    fid.write(tracker.track_kernel.specialized_source)

os.system('clang executable_with_xobjects.c -o exec_with_xobjects')

# Run executable
os.system('./exec_with_xobjects')

# Load output
buffer_out = tracker._buffer.context.new_buffer(capacity=sim_state.size)
with open('sim_state_out.bin', 'rb') as fid:
    state_bytes = fid.read(buffer_out.capacity)
buffer_out.buffer = np.frombuffer(state_bytes, dtype=np.int8)
sim_state_out = SimState._from_buffer(buffer=buffer_out, offset=0)
p_out = xp.Particles(_xobject=sim_state_out.particles)

assert np.all(p_out.s == 20)