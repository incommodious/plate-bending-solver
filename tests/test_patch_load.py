"""Test patch load support in Levy solver."""

import numpy as np
import matplotlib.pyplot as plt
from plate_bending.solvers.levy_solver import StableLevySolver

# Parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000

print("="*70)
print("PATCH LOAD VALIDATION - LEVY SOLVER")
print("="*70)
print()

# Test 1: Compare uniform vs patch load for SSSS plate
print("Test 1: SSSS plate - uniform vs patch load")
print("-"*50)

levy_ssss = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)

# Uniform load
result_uniform = levy_ssss.solve('uniform', q0)
print(f"  Uniform load:     W_max = {result_uniform['W_max']:.6e} m")
print(f"                    W_coef = {result_uniform['W_coef']:.6f}")

# Full-span patch (should match uniform)
result_full_patch = levy_ssss.solve('rect_patch', q0, x1=0.0, y1=0.0, x2=a, y2=b)
print(f"  Full patch [0,1]x[0,1]: W_max = {result_full_patch['W_max']:.6e} m")
print(f"                    W_coef = {result_full_patch['W_coef']:.6f}")

# Central patch
result_center = levy_ssss.solve('rect_patch', q0, x1=0.25*a, y1=0.25*b, x2=0.75*a, y2=0.75*b)
print(f"  Center patch [0.25,0.75]^2: W_max = {result_center['W_max']:.6e} m")
print(f"                    W_coef = {result_center['W_coef']:.6f}")

# Small central patch
result_small = levy_ssss.solve('rect_patch', q0, x1=0.4*a, y1=0.4*b, x2=0.6*a, y2=0.6*b)
print(f"  Small patch [0.4,0.6]^2: W_max = {result_small['W_max']:.6e} m")
print(f"                    W_coef = {result_small['W_coef']:.6f}")

print()
print("  Expected: W_uniform > W_center_patch > W_small_patch")
print(f"  Actual:   {result_uniform['W_coef']:.4f} > {result_center['W_coef']:.4f} > {result_small['W_coef']:.4f}")
if result_uniform['W_coef'] > result_center['W_coef'] > result_small['W_coef']:
    print("  PASS: Physical ordering correct")
else:
    print("  FAIL: Physical ordering incorrect!")
print()

# Test 2: SCSF plate with patch load
print("Test 2: SCSF plate - patch load")
print("-"*50)

levy_scsf = StableLevySolver(a, b, h, E, nu, 'C', 'F', 100)

result_uniform_scsf = levy_scsf.solve('uniform', q0)
print(f"  Uniform load:     W_coef = {result_uniform_scsf['W_coef']:.6f}")

result_patch_scsf = levy_scsf.solve('rect_patch', q0, x1=0.3*a, y1=0.3*b, x2=0.7*a, y2=0.7*b)
print(f"  Center patch:     W_coef = {result_patch_scsf['W_coef']:.6f}")

# Patch near free edge should cause larger deflection
result_patch_free = levy_scsf.solve('rect_patch', q0, x1=0.3*a, y1=0.6*b, x2=0.7*a, y2=1.0*b)
print(f"  Patch near free edge (y=0.6-1.0): W_coef = {result_patch_free['W_coef']:.6f}")

# Patch near clamped edge
result_patch_clamp = levy_scsf.solve('rect_patch', q0, x1=0.3*a, y1=0.0*b, x2=0.7*a, y2=0.4*b)
print(f"  Patch near clamped edge (y=0-0.4): W_coef = {result_patch_clamp['W_coef']:.6f}")

print()
print("  Expected: W_free_edge > W_center > W_clamped_edge")
print(f"  Actual:   {result_patch_free['W_coef']:.4f} > {result_patch_scsf['W_coef']:.4f} > {result_patch_clamp['W_coef']:.4f}")
if result_patch_free['W_coef'] > result_patch_scsf['W_coef'] > result_patch_clamp['W_coef']:
    print("  PASS: Physical behavior correct")
else:
    print("  FAIL: Physical behavior unexpected!")
print()

# Test 3: Continuity check at patch boundaries
print("Test 3: Continuity at patch boundaries")
print("-"*50)

levy = StableLevySolver(a, b, h, E, nu, 'S', 'S', 50)
result = levy.solve('rect_patch', q0, x1=0.3*a, y1=0.3*b, x2=0.7*a, y2=0.7*b)

# Check solution at y = 0.3 (boundary)
W = result['W']
Y = result['Y']
ny = W.shape[0]

# Find indices near y1=0.3 and y2=0.7
y1_idx = int(0.3 * (ny - 1))
y2_idx = int(0.7 * (ny - 1))

# Check for jumps at patch boundaries
w_at_y1_minus = W[y1_idx-1, ny//2]
w_at_y1 = W[y1_idx, ny//2]
w_at_y1_plus = W[y1_idx+1, ny//2]

jump_y1 = abs(w_at_y1_plus - w_at_y1) / abs(w_at_y1_plus - w_at_y1_minus)

w_at_y2_minus = W[y2_idx-1, ny//2]
w_at_y2 = W[y2_idx, ny//2]
w_at_y2_plus = W[y2_idx+1, ny//2]

jump_y2 = abs(w_at_y2_plus - w_at_y2) / abs(w_at_y2_plus - w_at_y2_minus)

print(f"  At y1=0.3: w[-1]={w_at_y1_minus:.6e}, w[0]={w_at_y1:.6e}, w[+1]={w_at_y1_plus:.6e}")
print(f"  At y2=0.7: w[-1]={w_at_y2_minus:.6e}, w[0]={w_at_y2:.6e}, w[+1]={w_at_y2_plus:.6e}")

# Check for smoothness (no sudden jumps)
max_jump = 0
for i in range(1, ny-1):
    d1 = W[i, ny//2] - W[i-1, ny//2]
    d2 = W[i+1, ny//2] - W[i, ny//2]
    if abs(d1) > 1e-15:
        jump = abs(d2 - d1) / (abs(d1) + abs(d2) + 1e-30)
        max_jump = max(max_jump, jump)

print(f"  Max relative jump in slope: {max_jump:.4f}")
if max_jump < 0.5:
    print("  PASS: Solution appears smooth")
else:
    print("  WARN: Possible discontinuity")
print()

# Test 4: Plot comparison
print("Test 4: Generating comparison plot...")
print("-"*50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# SSSS uniform
ax = axes[0, 0]
ax.contourf(result_uniform['X'], result_uniform['Y'], result_uniform['W']*1e3, levels=20)
ax.set_title(f"SSSS Uniform (W_max={result_uniform['W_max']*1e3:.4f} mm)")
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# SSSS center patch
ax = axes[0, 1]
ax.contourf(result_center['X'], result_center['Y'], result_center['W']*1e3, levels=20)
ax.set_title(f"SSSS Center Patch (W_max={result_center['W_max']*1e3:.4f} mm)")
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.axhline(0.25*b, color='r', linestyle='--', linewidth=1)
ax.axhline(0.75*b, color='r', linestyle='--', linewidth=1)
ax.axvline(0.25*a, color='r', linestyle='--', linewidth=1)
ax.axvline(0.75*a, color='r', linestyle='--', linewidth=1)

# SCSF uniform
ax = axes[1, 0]
ax.contourf(result_uniform_scsf['X'], result_uniform_scsf['Y'], result_uniform_scsf['W']*1e3, levels=20)
ax.set_title(f"SCSF Uniform (W_max={result_uniform_scsf['W_max']*1e3:.4f} mm)")
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# SCSF patch near free edge
ax = axes[1, 1]
ax.contourf(result_patch_free['X'], result_patch_free['Y'], result_patch_free['W']*1e3, levels=20)
ax.set_title(f"SCSF Patch near Free (W_max={result_patch_free['W_max']*1e3:.4f} mm)")
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.axhline(0.6*b, color='r', linestyle='--', linewidth=1)
ax.axvline(0.3*a, color='r', linestyle='--', linewidth=1)
ax.axvline(0.7*a, color='r', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('test_patch_load_results.png', dpi=150)
print(f"  Plot saved to test_patch_load_results.png")
print()

print("="*70)
print("PATCH LOAD TESTS COMPLETE")
print("="*70)
