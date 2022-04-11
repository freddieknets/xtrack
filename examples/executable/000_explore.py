import os
import numpy as np

import xtrack as xt
import xpart as xp


p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12, x=[99,99, 99])
p = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12, x=[1e-3,2e-3,3e-3], _buffer=p0._buffer)
p = p0


line = xt.Line(elements=[
    xt.Drift(length=1.0), xt.Multipole(knl=[1e-6]), xt.Drift(length=1.0)])
tracker = xt.Tracker(line=line)

with open('xtrack.h', 'w') as fid:
    fid.write(tracker.track_kernel.specialized_source)

# Write particles buffer to file
with open('part.bin', 'wb') as fid:
    fid.write(p._buffer.buffer.tobytes())

# Write line buffer to file
with open('line.bin', 'wb') as fid:
    fid.write(tracker._buffer.buffer.tobytes())

# write element offsets to file
with open('line_ele_offsets.bin', 'wb') as fid:
    fid.write(tracker.ele_offsets_dev.tobytes())

# write element type_ids to file
with open('line_ele_typeids.bin', 'wb') as fid:
    fid.write(tracker.ele_typeids_dev.tobytes())

with open('conf.txt', 'w') as fid:
    fid.write(f'{len(p._buffer.buffer)}\n')
    fid.write(f'{p._offset}\n')
    fid.write(f'{len(tracker._buffer.buffer)}\n')
    fid.write(f'{len(tracker.ele_offsets_dev)}\n')

os.system('bash compile_it.sh')
os.system('./hello')

buffer_out = p._buffer.context.new_buffer(capacity=p._buffer.capacity)
buffer_out.buffer[:] = np.fromfile('part_out.bin', dtype=np.int8)

p_out = xp.Particles(_buffer=buffer_out, _offset=p._offset)