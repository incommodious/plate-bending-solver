"""
Test Suite for Non-Levy Boundary Conditions (FCFC and CCCF)
============================================================
These boundary conditions require the Ritz solver as they are not Levy-type.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from plate_bending.solvers.ritz_solver import RitzSolver
from plate_bending.solvers.beam_functions import beam_function
from plate_bending.validation.benchmarks import Benchmarks

# Standard test parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000
D = E * h**3 / (12 * (1 - nu**2))

print("="*80)
print("NON-LEVY BOUNDARY CONDITIONS TEST SUITE (FCFC, CCCF, FCCC)")
print("="*80)
print(f"\nParameters: a={a}, b={b}, h={h}, E={E:.1e}, nu={nu}")
print(f"D = {D:.4f} N·m, q0 = {q0} Pa")
print()

results = []

# ==============================================================================
# TEST 1: Beam function boundary condition verification
# ==============================================================================
print("="*80)
print("TEST 1: Beam Function Boundary Conditions")
print("="*80)

xi = np.array([0.0, 1.0])
all_bc_ok = True

for bc in ['FC', 'CF', 'CC', 'FF']:
    phi = beam_function(xi, 1, bc, 0)
    phi_p = beam_function(xi, 1, bc, 1)
    phi_pp = beam_function(xi, 1, bc, 2)
    
    issues = []
    bc0, bc1 = bc[0], bc[1]
    
    # Check at xi=0
    if bc0 == 'C':
        if abs(phi[0]) > 0.01: issues.append(f"phi(0)={phi[0]:.2f} should be 0")
        if abs(phi_p[0]) > 0.01: issues.append(f"phi'(0)={phi_p[0]:.2f} should be 0")
    elif bc0 == 'F':
        # Free edge: curvature should be ~0 (natural BC, not strictly enforced)
        pass  # Natural BCs handled by variational formulation
    
    # Check at xi=1
    if bc1 == 'C':
        if abs(phi[1]) > 0.01: issues.append(f"phi(1)={phi[1]:.2f} should be 0")
        if abs(phi_p[1]) > 0.01: issues.append(f"phi'(1)={phi_p[1]:.2f} should be 0")
    elif bc1 == 'F':
        pass  # Natural BCs
    
    if issues:
        print(f"  {bc}: ❌ {issues}")
        all_bc_ok = False
    else:
        print(f"  {bc}: ✅ Essential BCs satisfied")

test1_pass = all_bc_ok
results.append(('Beam function BCs', test1_pass))
print()

# ==============================================================================
# TEST 2: FCFC Solver - Convergence Study
# ==============================================================================
print("="*80)
print("TEST 2: FCFC Convergence Study")
print("="*80)

print("FCFC: Free at x=0, Clamped at y=0, Free at x=a, Clamped at y=b")
print("  → x-direction: FF (Free-Free)")
print("  → y-direction: CC (Clamped-Clamped)")
print()

fcfc_results = []
for n in [5, 8, 10, 12, 15]:
    ritz = RitzSolver(a, b, h, E, nu, 'FCFC', n, n)
    R = ritz.solve('uniform', q0)
    fcfc_results.append(R['W_coef'])
    W = R['W']
    max_idx = np.unravel_index(np.argmax(np.abs(W)), W.shape)
    print(f"  n={n:2d}: W_coef = {R['W_coef']:.6f}, max at ({R['X'][max_idx]:.2f}, {R['Y'][max_idx]:.2f})")

# Check convergence (relative change < 2% from n=12 to n=15)
conv_change = abs(fcfc_results[-1] - fcfc_results[-2]) / fcfc_results[-2] * 100
print(f"\nConvergence check: {conv_change:.2f}% change from n=12 to n=15 (< 2%: {'PASS' if conv_change < 2 else 'FAIL'})")
test2_pass = conv_change < 2
results.append(('FCFC convergence', test2_pass))
print()

# ==============================================================================
# TEST 3: CCCF Solver - Convergence Study
# ==============================================================================
print("="*80)
print("TEST 3: CCCF Convergence Study")
print("="*80)

print("CCCF: Clamped at x=0, Clamped at y=0, Clamped at x=a, Free at y=b")
print("  → x-direction: CC (Clamped-Clamped)")
print("  → y-direction: CF (Clamped-Free)")
print()

cccf_results = []
for n in [5, 8, 10, 12, 15]:
    ritz = RitzSolver(a, b, h, E, nu, 'CCCF', n, n)
    R = ritz.solve('uniform', q0)
    cccf_results.append(R['W_coef'])
    W = R['W']
    max_idx = np.unravel_index(np.argmax(np.abs(W)), W.shape)
    print(f"  n={n:2d}: W_coef = {R['W_coef']:.6f}, max at ({R['X'][max_idx]:.2f}, {R['Y'][max_idx]:.2f})")

conv_change = abs(cccf_results[-1] - cccf_results[-2]) / cccf_results[-2] * 100
print(f"\nConvergence check: {conv_change:.2f}% change from n=12 to n=15 (< 2%: {'PASS' if conv_change < 2 else 'FAIL'})")
test3_pass = conv_change < 2
results.append(('CCCF convergence', test3_pass))
print()

# ==============================================================================
# TEST 4: Physical Behavior Checks
# ==============================================================================
print("="*80)
print("TEST 4: Physical Behavior Verification")
print("="*80)

# Test CCCF physical behavior
ritz_cccf = RitzSolver(a, b, h, E, nu, 'CCCF', 15, 15)
R_cccf = ritz_cccf.solve('uniform', q0)
W_cccf = R_cccf['W']
ny, nx = W_cccf.shape

# Max deflection should be at free edge (y=b)
max_idx = np.unravel_index(np.argmax(np.abs(W_cccf)), W_cccf.shape)
max_y = R_cccf['Y'][max_idx]
max_at_free = max_y > 0.9 * b  # Max should be near y=b (free edge)
print(f"CCCF: Max deflection at y={max_y:.2f} (should be near y={b:.2f} - free edge)")
print(f"  Max at free edge: {'PASS' if max_at_free else 'FAIL'}")

# Deflection at clamped edges should be zero
W_y0 = np.max(np.abs(W_cccf[0, :]))
W_x0 = np.max(np.abs(W_cccf[:, 0]))
W_xa = np.max(np.abs(W_cccf[:, -1]))
clamped_zero = W_y0 < 1e-6 and W_x0 < 1e-6 and W_xa < 1e-6
print(f"  Deflection at y=0: {W_y0:.2e} (should be ~0)")
print(f"  Deflection at x=0: {W_x0:.2e} (should be ~0)")
print(f"  Deflection at x=a: {W_xa:.2e} (should be ~0)")
print(f"  Clamped edges zero: {'PASS' if clamped_zero else 'FAIL'}")

# Free edge should have non-zero deflection
W_yb = np.max(np.abs(W_cccf[-1, :]))
free_nonzero = W_yb > 0.001  # Should have significant deflection
print(f"  Max deflection at y=b (free): {W_yb*1e3:.4f} mm")
print(f"  Free edge has deflection: {'PASS' if free_nonzero else 'FAIL'}")

test4_pass = max_at_free and clamped_zero and free_nonzero
results.append(('Physical behavior', test4_pass))
print()

# ==============================================================================
# TEST 5: FCFC Physical Behavior
# ==============================================================================
print("="*80)
print("TEST 5: FCFC Physical Behavior")
print("="*80)

ritz_fcfc = RitzSolver(a, b, h, E, nu, 'FCFC', 15, 15)
R_fcfc = ritz_fcfc.solve('uniform', q0)
W_fcfc = R_fcfc['W']
ny, nx = W_fcfc.shape

# Free edges at x=0 and x=a, clamped at y=0 and y=b
W_y0 = np.max(np.abs(W_fcfc[0, :]))
W_yb = np.max(np.abs(W_fcfc[-1, :]))
W_x0 = np.max(np.abs(W_fcfc[:, 0]))
W_xa = np.max(np.abs(W_fcfc[:, -1]))

print(f"FCFC edge deflections:")
print(f"  At y=0 (clamped): max |W| = {W_y0:.2e} mm (should be ~0)")
print(f"  At y=b (clamped): max |W| = {W_yb:.2e} mm (should be ~0)")
print(f"  At x=0 (free):    max |W| = {W_x0*1e3:.4f} mm (should be non-zero)")
print(f"  At x=a (free):    max |W| = {W_xa*1e3:.4f} mm (should be non-zero)")

# Check physics
clamped_y_zero = W_y0 < 1e-6 and W_yb < 1e-6
free_x_nonzero = W_x0 > 0.001 and W_xa > 0.001
print(f"  Clamped y-edges zero: {'PASS' if clamped_y_zero else 'FAIL'}")
print(f"  Free x-edges non-zero: {'PASS' if free_x_nonzero else 'FAIL'}")

test5_pass = clamped_y_zero and free_x_nonzero
results.append(('FCFC physical behavior', test5_pass))
print()

# ==============================================================================
# TEST 6: Comparison with Reference (CCCC)
# ==============================================================================
print("="*80)
print("TEST 6: Sanity Check Against CCCC (All Clamped)")
print("="*80)

# CCCC should have smaller deflection than CCCF (removing one clamp increases flexibility)
ritz_cccc = RitzSolver(a, b, h, E, nu, 'CCCC', 15, 15)
R_cccc = ritz_cccc.solve('uniform', q0)
bench_cccc = Benchmarks.get('CCCC')

print(f"CCCC (all clamped):")
print(f"  Ritz W_coef: {R_cccc['W_coef']:.6f}")
print(f"  Benchmark:   {bench_cccc['W_center_coef']:.6f}")
err_cccc = abs(R_cccc['W_coef'] - bench_cccc['W_center_coef']) / bench_cccc['W_center_coef'] * 100
print(f"  Error: {err_cccc:.2f}%")

# CCCF should be more flexible than CCCC
more_flexible = cccf_results[-1] > R_cccc['W_coef']
print(f"\nCCCF (one free edge) vs CCCC:")
print(f"  CCCC W_coef: {R_cccc['W_coef']:.6f}")
print(f"  CCCF W_coef: {cccf_results[-1]:.6f}")
print(f"  CCCF > CCCC (more flexible with free edge): {'PASS' if more_flexible else 'FAIL'}")

test6_pass = err_cccc < 5 and more_flexible
results.append(('CCCC reference check', test6_pass))
print()

# ==============================================================================
# TEST 7: Patch Load on CCCF
# ==============================================================================
print("="*80)
print("TEST 7: CCCF Patch Load Test")
print("="*80)

# Load near free edge should cause more deflection than load near clamped edge
ritz_cccf = RitzSolver(a, b, h, E, nu, 'CCCF', 12, 12)

# Load near clamped edge (y=0)
R_near_clamp = ritz_cccf.solve('rect_patch', q0, x1=0.3, y1=0.0, x2=0.7, y2=0.3)
# Load at center
R_center = ritz_cccf.solve('rect_patch', q0, x1=0.3, y1=0.35, x2=0.7, y2=0.65)
# Load near free edge (y=b)
R_near_free = ritz_cccf.solve('rect_patch', q0, x1=0.3, y1=0.7, x2=0.7, y2=1.0)

print(f"CCCF patch load deflections:")
print(f"  Near clamped (y=0):  W_coef = {R_near_clamp['W_coef']:.6f}")
print(f"  Center:              W_coef = {R_center['W_coef']:.6f}")
print(f"  Near free (y=b):     W_coef = {R_near_free['W_coef']:.6f}")

# Physical ordering: load near free edge should cause most deflection
ordering_ok = R_near_free['W_coef'] > R_center['W_coef'] > R_near_clamp['W_coef']
print(f"  Physical ordering (free > center > clamped): {'PASS' if ordering_ok else 'FAIL'}")

test7_pass = ordering_ok
results.append(('CCCF patch load physics', test7_pass))
print()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("="*80)
print("SUMMARY")
print("="*80)

passed = sum(1 for _, p in results if p)
total = len(results)

for name, passed_test in results:
    print(f"  [{'PASS' if passed_test else 'FAIL':4}] {name}")

print()
print(f"Total: {passed}/{total} tests passed")

if passed == total:
    print("\n✅ ALL TESTS PASSED - FCFC and CCCF implementations validated!")
else:
    print(f"\n⚠️  {total - passed} test(s) need attention")

print("="*80)
