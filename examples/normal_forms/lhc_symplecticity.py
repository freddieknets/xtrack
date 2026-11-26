# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import time
import matplotlib.pyplot as plt

from madng_tpsa import TpsaMap
from xtrack.tpsa import COORDS, ParticlesTpsa

from get_lhc_lattice import generate_lhc_without_params


env, d = generate_lhc_without_params()
line = env.b1


# ================================
# === 1. Full LHC one-turn map ===
# ================================

# Create a ParticlesTpsa around the numerical closed orbit
ORDER = 6
particle_ref = line.particle_ref
line.build_tracker()
co_ref = line.find_closed_orbit(particle_ref=particle_ref, freeze_longitudinal=False)
particle_kwargs = co_ref.to_dict()
for name in ('delta', 'ptau', 'rpp', 'rvv', '_rpp', '_rvv'):
    particle_kwargs.pop(name, None)
m_init = ParticlesTpsa(descriptor=d, **particle_kwargs)

# Find elements to store the map at
co_ref_arr = m_init.const_part.copy()
c0 = co_ref_arr.copy()
tt = line.get_table()
els = tt.rows[[
    not ttt.startswith('Drift')
    and not ttt.startswith('Limit')
    and not ttt.startswith('Marker')
    for ttt in tt.element_type
]].name

# Track
m1 = m_init.copy()
t0 = time.time()
line.track(m1, multi_element_monitor_at=els)
print(f"Tracked LHC one-turn map in {time.time() - t0}s.")

print(f"Non-symplectic coefs by degree:")
errors, worst = m1.non_symplectic_coeffs_by_degree(norm='relative_frobenius')
for degree, error in errors.items():
    print(f"    {degree}: {error:.4e}    worst: {worst[degree]}")

# Calculate the non-symplectic coefficients by degree for each s
names = []
deg = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
for i, comp in enumerate(line.record_multi_element_last_track._map_series[0]):
    m_s = TpsaMap(comp)
    error, worst = m_s.non_symplectic_coeffs_by_degree(norm='relative_frobenius')
    for j in range(6):
        deg[j].append(error.get(j, 0))
    names.append(line.record_multi_element_last_track.obs_names[i-1])

# Plot
tt = line.get_table()
ss = tt.rows[names].s
_, ax = plt.subplots(1, 1)
for nn, dd in deg.items():
    ax.plot(ss, dd, label=f'Degree {nn}')
ax.set_yscale('log')
ax.set_ylim((1e-16,100))
ax.set_xlabel('s [m]')
ax.set_ylabel('Rel. Frob. norm of difference by degree')
ax.legend()
ax.set_title('Non-symplecticity of LHC one-turn map')
plt.tight_layout()
plt.savefig('lhc_symplecticity.png', dpi=300)


# ================================
# === 2. Single quadrupole map ===
# ================================


# There is one element (a Quadrupole at IP5) that adds a lot of non-symplecticity
el = 'mqxb.a2l5/b1'
n_iterations = 50
m2 = m_init.copy()
print()

# We track many times through this map to see how fast the non-symplectiticy grows
all_maps = [m2.copy()]
t0 = time.time()
for _ in range(n_iterations):
    line[el].track(m2)
    all_maps.append(m2.copy())
print(f"Tracked quadrupole map ({n_iterations}) in {time.time() - t0}s.")

print(f"Final non-symplectic coefs by degree:")
errors, worst = m2.non_symplectic_coeffs_by_degree(norm='relative_frobenius')
for degree, error in errors.items():
    print(f"    {degree}: {error:.4e}    worst: {worst[degree]}")

# Calculate the non-symplectic coefficients by degree for each iteration
deg = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
for m_iter in all_maps:
    error, worst = m_iter.non_symplectic_coeffs_by_degree(norm='relative_frobenius')
    for j in range(6):
        deg[j].append(error.get(j, 0))

# Plot
ss = list(range(n_iterations+1))
_, ax = plt.subplots(1, 1)
for nn, dd in deg.items():
    ax.plot(ss, dd, label=f'Degree {nn}')
ax.set_yscale('log')
ax.set_ylim((1e-16,1e10))
ax.set_xlabel('s [m]')
ax.set_ylabel('Rel. Frob. norm of difference by degree')
ax.legend()
ax.set_title(f'Non-symplecticity of quadrupole map {el}')
plt.tight_layout()
plt.savefig('quad_symplecticity.png', dpi=300)
