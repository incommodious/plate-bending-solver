"""Comprehensive validation of the plate bending solvers."""

from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.ritz_solver import RitzSolver

# Parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000

print("="*70)
print("VALIDATION SUMMARY")
print("="*70)

# Test 1: SSSS uniform
levy_ssss = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
L_ssss = levy_ssss.solve('uniform', q0)
ritz_ssss = RitzSolver(a, b, h, E, nu, 'SSSS', M=15, N=15)
R_ssss = ritz_ssss.solve('uniform', q0)
bench_ssss = 0.00406
err_levy = abs(L_ssss['W_coef'] - bench_ssss) / bench_ssss * 100
err_ritz = abs(R_ssss['W_coef'] - bench_ssss) / bench_ssss * 100
diff = abs(L_ssss['W_coef'] - R_ssss['W_coef']) / L_ssss['W_coef'] * 100
print(f"SSSS uniform:")
print(f"  Levy:  {L_ssss['W_coef']:.6f} (vs bench 0.00406: {err_levy:.2f}% error)")
print(f"  Ritz:  {R_ssss['W_coef']:.6f} (vs bench 0.00406: {err_ritz:.2f}% error)")
print(f"  Levy vs Ritz: {diff:.2f}% difference")
print()

# Test 2: SSSF uniform
levy_sssf = StableLevySolver(a, b, h, E, nu, 'S', 'F', 100)
L_sssf = levy_sssf.solve('uniform', q0)
bench_sssf = 0.01286
err_levy = abs(L_sssf['W_coef'] - bench_sssf) / bench_sssf * 100
print(f"SSSF uniform:")
print(f"  Levy:  {L_sssf['W_coef']:.6f} (vs bench 0.01286: {err_levy:.2f}% error)")
print()

# Test 3: SCSF uniform
levy_scsf = StableLevySolver(a, b, h, E, nu, 'C', 'F', 100)
L_scsf = levy_scsf.solve('uniform', q0)
ritz_scsf = RitzSolver(a, b, h, E, nu, 'SCSF', M=15, N=15)
R_scsf = ritz_scsf.solve('uniform', q0)
diff = abs(L_scsf['W_coef'] - R_scsf['W_coef']) / L_scsf['W_coef'] * 100
print(f"SCSF uniform:")
print(f"  Levy:  {L_scsf['W_coef']:.6f}")
print(f"  Ritz:  {R_scsf['W_coef']:.6f}")
print(f"  Levy vs Ritz: {diff:.2f}% difference")
print()

# Test 4: Patch load
levy_patch = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
L_patch = levy_patch.solve('rect_patch', q0, x1=0.25, x2=0.75, y1=0.25, y2=0.75)
ritz_patch = RitzSolver(a, b, h, E, nu, 'SSSS', M=15, N=15)
R_patch = ritz_patch.solve('rect_patch', q0, x1=0.25, x2=0.75, y1=0.25, y2=0.75)
diff = abs(L_patch['W_coef'] - R_patch['W_coef']) / L_patch['W_coef'] * 100
print(f"SSSS rect patch (0.25-0.75):")
print(f"  Levy:  {L_patch['W_coef']:.6f}")
print(f"  Ritz:  {R_patch['W_coef']:.6f}")
print(f"  Levy vs Ritz: {diff:.2f}% difference")
print()

print("="*70)
print("PHYSICAL ORDERING CHECK")
print("="*70)

bcs = [('S', 'S', 'SSSS'), ('C', 'S', 'SCSS'), ('C', 'C', 'SCSC'), ('C', 'F', 'SCSF'), ('S', 'F', 'SSSF')]
results = []
for bc0, bc1, name in bcs:
    levy = StableLevySolver(a, b, h, E, nu, bc0, bc1, 100)
    L = levy.solve('uniform', q0)
    results.append((name, L['W_coef']))

results.sort(key=lambda x: x[1])
print("Ordering (more constraints = less deflection):")
for name, w in results:
    print(f"  {name}: {w:.6f}")
print()
expected = ['SCSC', 'SCSS', 'SSSS', 'SCSF', 'SSSF']
actual = [r[0] for r in results]
if actual == expected:
    print("Physical ordering: CORRECT")
else:
    print(f"Physical ordering: UNEXPECTED (got {actual})")

print()
print("="*70)
print("ALL TESTS COMPLETE")
print("="*70)
