import xtrack as xt
import xpart as xp

p = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12, x=[1,2,3])

line = xt.Line(elements=[xt.Drift(length=1.0)])
tracker = xt.Tracker(line=line)

with open('xtrack.h', 'w') as fid:
    fid.write(tracker.track_kernel.specialized_source)
