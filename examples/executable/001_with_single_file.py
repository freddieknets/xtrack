from functools import partial
import xobjects as xo
import xtrack as xt
import xpart as xp

class LineMetaData(xo.Struct):
    ele_offsets = xo.Int64[:]
    ele_typeids = xo.Int64[:]

class SimConfig(xo.Struct):
    buffer_size = xo.Int64
    line_metadata = xo.Ref(LineMetaData)
    num_turns = xo.Int64
    particles = xo.Ref(xp.Particles.XoStruct)

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
sim_config.line_metadata = line_metadata
sim_config.num_turns = num_turns
sim_config.particles = particles.copy(_buffer=simbuf)._xobject

assert sim_config._offset == 0
assert sim_config._fields[0].offset == 0
assert sim_config._fields[0].name == 'buffer_size'

sim_config.buffer_size = simbuf.capacity

# Write particles buffer to file
with open('sim.bin', 'wb') as fid:
    fid.write(simbuf.buffer.tobytes())