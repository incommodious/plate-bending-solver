"""
Comprehensive Validation Suite for Plate Bending Solvers
=========================================================

Tests all success criteria:
1. SSSS + uniform matches Timoshenko within 5%
2. SSSF + uniform matches Timoshenko within 5%
3. Levy and Ritz agree within 3%
4. Patch loads physically reasonable
5. All BC configurations work correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.ritz_solver import RitzSolver
from plate_bending.validation.benchmarks import Benchmarks

# Test parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000

print("="*80)
print("COMPREHENSIVE VALIDATION SUITE")
print("="*80)
print()

results = []

# =============================================================================
# TEST 1: SSSS + uniform matches Timoshenko within 5%
# =============================================================================
print("TEST 1: SSSS + uniform vs Timoshenko (< 5%)")
print("-"*60)

levy_ssss = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
L_ssss = levy_ssss.solve('uniform', q0)
bench_ssss = Benchmarks.get('SSSS')['W_center_coef']
err_ssss = abs(L_ssss['W_coef'] - bench_ssss) / bench_ssss * 100

print(f"  Levy:      {L_ssss['W_coef']:.6f}")
print(f"  Benchmark: {bench_ssss:.6f}")
print(f"  Error:     {err_ssss:.2f}%")
test1_pass = err_ssss < 5
print(f"  Result:    {'PASS' if test1_pass else 'FAIL'}")
results.append(('SSSS uniform vs Timoshenko', test1_pass))
print()

# =============================================================================
# TEST 2: SSSF + uniform matches Timoshenko within 5%
# =============================================================================
print("TEST 2: SSSF + uniform vs Timoshenko (< 5%)")
print("-"*60)

levy_sssf = StableLevySolver(a, b, h, E, nu, 'S', 'F', 100)
L_sssf = levy_sssf.solve('uniform', q0)
bench_sssf = Benchmarks.get('SSSF')['W_max_coef']
err_sssf = abs(L_sssf['W_coef'] - bench_sssf) / bench_sssf * 100

print(f"  Levy:      {L_sssf['W_coef']:.6f}")
print(f"  Benchmark: {bench_sssf:.6f}")
print(f"  Error:     {err_sssf:.2f}%")
test2_pass = err_sssf < 5
print(f"  Result:    {'PASS' if test2_pass else 'FAIL'}")
results.append(('SSSF uniform vs Timoshenko', test2_pass))
print()

# =============================================================================
# TEST 3: Levy and Ritz agree within 3%
# =============================================================================
print("TEST 3: Levy vs Ritz agreement (< 3%)")
print("-"*60)

test_cases = [('SSSS', 'S', 'S'), ('SCSF', 'C', 'F')]
all_agree = True

for bc_name, bc_y0, bc_yb in test_cases:
    levy = StableLevySolver(a, b, h, E, nu, bc_y0, bc_yb, 100)
    L = levy.solve('uniform', q0)

    ritz = RitzSolver(a, b, h, E, nu, bc_name, 15, 15)
    R = ritz.solve('uniform', q0)

    diff = abs(L['W_coef'] - R['W_coef']) / L['W_coef'] * 100
    agrees = diff < 3
    all_agree = all_agree and agrees

    print(f"  {bc_name}: Levy={L['W_coef']:.6f}, Ritz={R['W_coef']:.6f}, diff={diff:.2f}% {'PASS' if agrees else 'FAIL'}")

test3_pass = all_agree
print(f"  Result:    {'PASS' if test3_pass else 'PARTIAL'}")
results.append(('Levy vs Ritz agreement', test3_pass))
print()

# =============================================================================
# TEST 4: Patch loads physically reasonable
# =============================================================================
print("TEST 4: Patch loads physically reasonable")
print("-"*60)

# Test 4a: Smaller patch = smaller deflection
levy = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
W_uniform = levy.solve('uniform', q0)['W_coef']
W_patch_large = levy.solve('rect_patch', q0, x1=0.2, y1=0.2, x2=0.8, y2=0.8)['W_coef']
W_patch_small = levy.solve('rect_patch', q0, x1=0.4, y1=0.4, x2=0.6, y2=0.6)['W_coef']

ordering_ok = W_uniform > W_patch_large > W_patch_small
print(f"  Uniform:     W_coef = {W_uniform:.6f}")
print(f"  Large patch: W_coef = {W_patch_large:.6f}")
print(f"  Small patch: W_coef = {W_patch_small:.6f}")
print(f"  Physical ordering (W_uniform > W_large > W_small): {'PASS' if ordering_ok else 'FAIL'}")

# Test 4b: SCSF - load near free edge causes more deflection
levy_scsf = StableLevySolver(a, b, h, E, nu, 'C', 'F', 100)
W_clamp = levy_scsf.solve('rect_patch', q0, x1=0.3, y1=0.0, x2=0.7, y2=0.3)['W_coef']
W_center = levy_scsf.solve('rect_patch', q0, x1=0.3, y1=0.35, x2=0.7, y2=0.65)['W_coef']
W_free = levy_scsf.solve('rect_patch', q0, x1=0.3, y1=0.7, x2=0.7, y2=1.0)['W_coef']

scsf_ordering = W_free > W_center > W_clamp
print(f"  SCSF patch near clamped: {W_clamp:.6f}")
print(f"  SCSF patch at center:    {W_center:.6f}")
print(f"  SCSF patch near free:    {W_free:.6f}")
print(f"  Physical ordering (W_free > W_center > W_clamp): {'PASS' if scsf_ordering else 'FAIL'}")

# Test 4c: Levy and Ritz patch loads agree
levy_patch = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
L_patch = levy_patch.solve('rect_patch', q0, x1=0.3, y1=0.3, x2=0.7, y2=0.7)

ritz_patch = RitzSolver(a, b, h, E, nu, 'SSSS', 15, 15)
R_patch = ritz_patch.solve('rect_patch', q0, x1=0.3, y1=0.3, x2=0.7, y2=0.7)

patch_diff = abs(L_patch['W_coef'] - R_patch['W_coef']) / L_patch['W_coef'] * 100
patch_agree = patch_diff < 5
print(f"  SSSS patch: Levy={L_patch['W_coef']:.6f}, Ritz={R_patch['W_coef']:.6f}, diff={patch_diff:.2f}%")

test4_pass = ordering_ok and scsf_ordering and patch_agree
print(f"  Result:    {'PASS' if test4_pass else 'PARTIAL'}")
results.append(('Patch loads physically reasonable', test4_pass))
print()

# =============================================================================
# TEST 5: All BC configurations work
# =============================================================================
print("TEST 5: All BC configurations")
print("-"*60)

bc_cases = [
    ('SSSS', 'S', 'S'),
    ('SCSC', 'C', 'C'),
    ('SCSS', 'C', 'S'),
    ('SCSF', 'C', 'F'),
    ('SSSF', 'S', 'F'),
    ('SFSF', 'F', 'F'),
]

all_work = True
for bc_name, bc_y0, bc_yb in bc_cases:
    try:
        levy = StableLevySolver(a, b, h, E, nu, bc_y0, bc_yb, 50)
        L = levy.solve('uniform', q0)

        # Check solution is finite and positive
        valid = np.isfinite(L['W_coef']) and L['W_coef'] > 0

        bench = Benchmarks.get(bc_name)
        if bench:
            ref_key = 'W_max_coef' if 'W_max_coef' in bench else 'W_center_coef'
            ref = bench[ref_key]
            err = abs(L['W_coef'] - ref) / ref * 100
            status = f"err={err:.1f}%"
        else:
            status = "no benchmark"

        print(f"  {bc_name}: W_coef={L['W_coef']:.6f} ({status}) {'OK' if valid else 'INVALID'}")
        all_work = all_work and valid

    except Exception as e:
        print(f"  {bc_name}: ERROR - {e}")
        all_work = False

test5_pass = all_work
print(f"  Result:    {'PASS' if test5_pass else 'FAIL'}")
results.append(('All BC configurations work', test5_pass))
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("="*80)
print("SUMMARY")
print("="*80)

passed = sum(1 for _, p in results if p)
total = len(results)

for name, passed_test in results:
    print(f"  [{('PASS' if passed_test else 'FAIL'):4}] {name}")

print()
print(f"Total: {passed}/{total} tests passed")

if passed == total:
    print("\nALL TESTS PASSED - Implementation is validated!")
else:
    print(f"\n{total - passed} test(s) need attention")

print("="*80)
