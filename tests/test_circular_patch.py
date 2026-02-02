"""Test circular patch load approximation."""

import numpy as np
from plate_bending.solvers.levy_solver import StableLevySolver

# Parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000

print("="*70)
print("CIRCULAR PATCH LOAD VALIDATION")
print("="*70)
print()

# Test with SSSS plate
levy = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)

# 1. Point load approximation (very small R)
result_point = levy.solve('circular', q0, x0=0.5*a, y0=0.5*b, R=0.01*a)
print(f"Very small R (0.01a): W_max = {result_point['W_max']*1e3:.4f} mm")

# 2. Small circular patch
result_small = levy.solve('circular', q0, x0=0.5*a, y0=0.5*b, R=0.05*a)
print(f"Small R (0.05a):      W_max = {result_small['W_max']*1e3:.4f} mm")

# 3. Medium circular patch
result_med = levy.solve('circular', q0, x0=0.5*a, y0=0.5*b, R=0.1*a)
print(f"Medium R (0.1a):      W_max = {result_med['W_max']*1e3:.4f} mm")

# 4. Large circular patch
result_large = levy.solve('circular', q0, x0=0.5*a, y0=0.5*b, R=0.2*a)
print(f"Large R (0.2a):       W_max = {result_large['W_max']*1e3:.4f} mm")

print()
print("Expected: Larger radius = larger deflection (more loaded area)")

# 5. Compare with equivalent rectangular patch
# Circular patch area = pi*R^2
# For R=0.1, area = 0.0314 a^2
# Equivalent square side = sqrt(0.0314) * a = 0.177a
# Square from (0.5-0.0885, 0.5-0.0885) to (0.5+0.0885, 0.5+0.0885)
print()
print("-"*50)
print("Comparison: Circular vs equivalent-area rectangular patch")
print("-"*50)

R = 0.15
area_circ = np.pi * R**2
side = np.sqrt(area_circ)

result_circ = levy.solve('circular', q0, x0=0.5*a, y0=0.5*b, R=R*a)
result_rect = levy.solve('rect_patch', q0,
                         x1=(0.5-side/2)*a, y1=(0.5-side/2)*b,
                         x2=(0.5+side/2)*a, y2=(0.5+side/2)*b)

print(f"Circle R={R}a (area={area_circ:.4f}a^2):")
print(f"  W_max = {result_circ['W_max']*1e3:.4f} mm, W_coef = {result_circ['W_coef']:.6f}")

print(f"Square side={side:.4f}a (area={side**2:.4f}a^2):")
print(f"  W_max = {result_rect['W_max']*1e3:.4f} mm, W_coef = {result_rect['W_coef']:.6f}")

diff = abs(result_circ['W_coef'] - result_rect['W_coef']) / result_rect['W_coef'] * 100
print(f"Difference: {diff:.1f}%")

# 6. Off-center circular patch
print()
print("-"*50)
print("Off-center circular patch (R=0.1a)")
print("-"*50)

result_center = levy.solve('circular', q0, x0=0.5*a, y0=0.5*b, R=0.1*a)
result_edge = levy.solve('circular', q0, x0=0.3*a, y0=0.5*b, R=0.1*a)

print(f"Center (0.5, 0.5): W_max = {result_center['W_max']*1e3:.4f} mm at {result_center['W_max_loc']}")
print(f"Edge (0.3, 0.5):   W_max = {result_edge['W_max']*1e3:.4f} mm at {result_edge['W_max_loc']}")

# For SCSF, check load position effects
print()
print("-"*50)
print("SCSF plate: Circular patch position effects")
print("-"*50)

levy_scsf = StableLevySolver(a, b, h, E, nu, 'C', 'F', 100)

# Near clamped edge (y=0)
result_clamp = levy_scsf.solve('circular', q0, x0=0.5*a, y0=0.2*b, R=0.1*a)
# Center
result_mid = levy_scsf.solve('circular', q0, x0=0.5*a, y0=0.5*b, R=0.1*a)
# Near free edge (y=b)
result_free = levy_scsf.solve('circular', q0, x0=0.5*a, y0=0.8*b, R=0.1*a)

print(f"Near clamped (y=0.2): W_max = {result_clamp['W_max']*1e3:.4f} mm")
print(f"Center (y=0.5):       W_max = {result_mid['W_max']*1e3:.4f} mm")
print(f"Near free (y=0.8):    W_max = {result_free['W_max']*1e3:.4f} mm")

print()
if result_free['W_max'] > result_mid['W_max'] > result_clamp['W_max']:
    print("PASS: Physical behavior correct (load near free edge causes most deflection)")
else:
    print("CHECK: Results need investigation")

print()
print("="*70)
print("CIRCULAR PATCH TESTS COMPLETE")
print("="*70)
