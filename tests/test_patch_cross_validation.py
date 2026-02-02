"""Cross-validate patch loads between Levy and Ritz solvers."""

import numpy as np
from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.ritz_solver import RitzSolver

# Parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000

print("="*70)
print("PATCH LOAD CROSS-VALIDATION: LEVY vs RITZ")
print("="*70)
print()

# Test 1: SSSS with rectangular patch
print("Test 1: SSSS plate, center rectangular patch [0.3,0.7]^2")
print("-"*50)

levy_ssss = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
ritz_ssss = RitzSolver(a, b, h, E, nu, 'SSSS', M=15, N=15)

x1, y1, x2, y2 = 0.3*a, 0.3*b, 0.7*a, 0.7*b

L_result = levy_ssss.solve('rect_patch', q0, x1=x1, y1=y1, x2=x2, y2=y2)
R_result = ritz_ssss.solve('rect_patch', q0, x1=x1, y1=y1, x2=x2, y2=y2)

print(f"  Levy W_coef:  {L_result['W_coef']:.6f}")
print(f"  Ritz W_coef:  {R_result['W_coef']:.6f}")
diff = abs(L_result['W_coef'] - R_result['W_coef']) / L_result['W_coef'] * 100
print(f"  Difference:   {diff:.2f}%")
if diff < 5:
    print("  PASS: Agreement within 5%")
else:
    print("  CHECK: Larger than expected difference")
print()

# Test 2: SSSS uniform (baseline check)
print("Test 2: SSSS plate, uniform load (baseline)")
print("-"*50)

L_unif = levy_ssss.solve('uniform', q0)
R_unif = ritz_ssss.solve('uniform', q0)

print(f"  Levy W_coef:  {L_unif['W_coef']:.6f}")
print(f"  Ritz W_coef:  {R_unif['W_coef']:.6f}")
print(f"  Benchmark:    0.00406")
diff_unif = abs(L_unif['W_coef'] - R_unif['W_coef']) / L_unif['W_coef'] * 100
print(f"  Difference:   {diff_unif:.2f}%")
print()

# Test 3: SCSF with rectangular patch
print("Test 3: SCSF plate, rectangular patch")
print("-"*50)

levy_scsf = StableLevySolver(a, b, h, E, nu, 'C', 'F', 100)
ritz_scsf = RitzSolver(a, b, h, E, nu, 'SCSF', M=15, N=15)

# Center patch
x1, y1, x2, y2 = 0.3*a, 0.3*b, 0.7*a, 0.7*b

L_scsf = levy_scsf.solve('rect_patch', q0, x1=x1, y1=y1, x2=x2, y2=y2)
R_scsf = ritz_scsf.solve('rect_patch', q0, x1=x1, y1=y1, x2=x2, y2=y2)

print(f"  Levy W_coef:  {L_scsf['W_coef']:.6f}")
print(f"  Ritz W_coef:  {R_scsf['W_coef']:.6f}")
diff_scsf = abs(L_scsf['W_coef'] - R_scsf['W_coef']) / L_scsf['W_coef'] * 100
print(f"  Difference:   {diff_scsf:.2f}%")
print()

# Test 4: Circular patch comparison
print("Test 4: SSSS plate, circular patch R=0.15")
print("-"*50)

R_circ = 0.15
xc, yc = 0.5*a, 0.5*b

L_circ = levy_ssss.solve('circular', q0, x0=xc, y0=yc, R=R_circ*a)
R_circ_result = ritz_ssss.solve('circular', q0, x0=xc, y0=yc, R=R_circ*a)

print(f"  Levy W_coef:  {L_circ['W_coef']:.6f}")
print(f"  Ritz W_coef:  {R_circ_result['W_coef']:.6f}")
diff_circ = abs(L_circ['W_coef'] - R_circ_result['W_coef']) / L_circ['W_coef'] * 100
print(f"  Difference:   {diff_circ:.2f}%")
print()

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
print()
print(f"  SSSS uniform:    {diff_unif:.2f}% difference")
print(f"  SSSS rect patch: {diff:.2f}% difference")
print(f"  SCSF rect patch: {diff_scsf:.2f}% difference")
print(f"  SSSS circ patch: {diff_circ:.2f}% difference")
print()

if max(diff_unif, diff) < 5:
    print("VERDICT: Patch load implementations agree well for SSSS")
else:
    print("VERDICT: Further investigation needed")

print("="*70)
