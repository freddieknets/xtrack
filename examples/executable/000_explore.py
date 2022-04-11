from re import I
import xtrack as xt
import xpart as xp


p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12, x=[10,20,30])
p = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12, x=[1,2,3], _buffer=p0._buffer)

line = xt.Line(elements=[
    xt.Drift(length=1.0), xt.Multipole(knl=[10]), xt.Drift(length=1.0)])
tracker = xt.Tracker(line=line)

with open('xtrack.h', 'w') as fid:
    fid.write(tracker.track_kernel.specialized_source)

# Write particles buffer to file
with open('part.bin', 'wb') as fid:
    fid.write(p._buffer.buffer.tobytes())

# Write line buffer to file
with open('line.bin', 'wb') as fid:
    fid.write(tracker._buffer.buffer.tobytes())

with open('conf.txt', 'w') as fid:
    fid.write(f'{len(p._buffer.buffer)}\n')
    fid.write(f'{p._offset}\n')
    fid.write(f'{len(tracker._buffer.buffer)}\n')
    fid.write(f'{len(tracker.ele_offsets_dev)}\n')