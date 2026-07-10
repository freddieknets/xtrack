import xtrack as xt
import numpy as np
env = xt.Environment()
env.new('rb', 'RBend', length_straight=2., angle=0.1)

line = env.new_line(components=['rb'])
line.set_particle_ref('electron', energy0=5e9)
line.configure_bend_model(edge='full')

two = line.twiss4d(betx=1, bety=1, spin=True, spin_x=1)

spin_angle = np.atan2(two.spin_z, two.spin_x)/(2*np.pi)