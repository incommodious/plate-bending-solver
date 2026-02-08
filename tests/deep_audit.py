"""
Deep Physics & Math Audit for Plate Bending Solver
===================================================
Comprehensive validation against multiple reference sources.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.fit_solver import FITSolver
from plate_bending.solvers.ritz_solver import RitzSolver
from plate_bending.solvers.beam_functions import beam_function, get_eigenvalue

# Standard test parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000
D = E * h**3 / (12 * (1 - nu**2))

print("="*80)
print("DEEP PHYSICS & MATH AUDIT")
print("="*80)
print(f"\nParameters: a={a}, b={b}, h={h}, E={E:.1e}, nu={nu}")
print(f"D = {D:.4f} N·m")
print(f"q0 = {q0} Pa")
print()

# ==============================================================================
# SECTION 1: Verify flexural rigidity calculation
# ==============================================================================
print("=" * 80)
print("SECTION 1: FLEXURAL RIGIDITY CHECK")
print("=" * 80)
D_manual = E * h**3 / (12 * (1 - nu**2))
print(f"D = E*h^3 / (12*(1-nu^2)) = {E:.1e} * {h}^3 / (12*(1-{nu}^2))")
print(f"  = {E * h**3:.4e} / {12 * (1 - nu**2):.4f}")
print(f"  = {D_manual:.4f} N·m")
print()

# ==============================================================================
# SECTION 2: SSSS Uniform Load - Navier Exact Solution
# ==============================================================================
print("=" * 80)
print("SECTION 2: SSSS UNIFORM LOAD - NAVIER EXACT CHECK")
print("=" * 80)

# Exact Navier solution for center deflection of SSSS plate
# w_center = (16 * q0) / (pi^6 * D) * sum_{m,n odd} sin(m*pi/2)*sin(n*pi/2) / (m*n*(m^2/a^2 + n^2/b^2)^2)
# For a=b: w_center = (16 * q0 * a^4) / (pi^6 * D) * sum = alpha * q0 * a^4 / D
# alpha_center = 0.00406235

w_exact_center = 0
for m in range(1, 200, 2):
    for n in range(1, 200, 2):
        w_exact_center += 1.0 / (m * n * (m**2 + n**2)**2)
w_exact_coef = 16.0 / np.pi**6 * w_exact_center

print(f"Navier analytical center deflection coefficient: {w_exact_coef:.6f}")
print(f"Timoshenko Table 8 reference:                     0.004060")
print(f"Match: {'YES' if abs(w_exact_coef - 0.00406) / 0.00406 < 0.001 else 'NO'}")

# Compare with Levy solver
levy_ssss = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
result_ssss = levy_ssss.solve('uniform', q0)
W = result_ssss['W']
ny, nx = W.shape
W_center_levy = W[ny//2, nx//2]
W_center_coef_levy = abs(W_center_levy) * D / (q0 * a**4)
W_max_coef_levy = result_ssss['W_coef']

print(f"Levy center coef:  {W_center_coef_levy:.6f}")
print(f"Levy max coef:     {W_max_coef_levy:.6f}")
print(f"Max = Center (SSSS): {abs(W_center_coef_levy - W_max_coef_levy) / W_max_coef_levy < 0.01}")

# Compare with FIT (Navier)
fit_ssss = FITSolver(a, b, h, E, nu, 'SSSS', 100)
result_fit = fit_ssss.solve('uniform', q0)
print(f"FIT center coef:   {result_fit['W_center_coef']:.6f}")
print(f"FIT max coef:      {result_fit['W_coef']:.6f}")

# Ritz
ritz_ssss = RitzSolver(a, b, h, E, nu, 'SSSS', 15, 15)
result_ritz = ritz_ssss.solve('uniform', q0)
print(f"Ritz center coef:  {result_ritz['W_center_coef']:.6f}")
print(f"Ritz max coef:     {result_ritz['W_coef']:.6f}")
print()

# ==============================================================================
# SECTION 3: SSSS CENTER MOMENTS - Verify moment computation
# ==============================================================================
print("=" * 80)
print("SECTION 3: SSSS CENTER MOMENTS CHECK")
print("=" * 80)

# Timoshenko: Mx_center = My_center = 0.0479 * q * a^2 for nu=0.3, a/b=1
Mx_center = result_ssss['Mx'][ny//2, nx//2]
My_center = result_ssss['My'][ny//2, nx//2]
Mx_coef = abs(Mx_center) / (q0 * a**2)
My_coef = abs(My_center) / (q0 * a**2)

print(f"Levy Mx_center coef: {Mx_coef:.6f} (Timoshenko: 0.0479)")
print(f"Levy My_center coef: {My_coef:.6f} (Timoshenko: 0.0479)")
print(f"Mx/My ratio: {Mx_coef/My_coef:.4f} (should be ~1.0 for square SSSS)")
print(f"Mx error: {abs(Mx_coef - 0.0479)/0.0479*100:.1f}%")
print(f"My error: {abs(My_coef - 0.0479)/0.0479*100:.1f}%")
print()

# ==============================================================================
# SECTION 4: DETAILED BC COMPARISON - W_center vs W_max
# ==============================================================================
print("=" * 80)
print("SECTION 4: DETAILED BC ANALYSIS - CENTER vs MAX VALUES")
print("=" * 80)

bc_configs = [
    ('SSSS', 'S', 'S'),
    ('SCSC', 'C', 'C'),
    ('SCSS', 'C', 'S'),
    ('SCSF', 'C', 'F'),
    ('SSSF', 'S', 'F'),
    ('SFSF', 'F', 'F'),
]

for bc_name, bc_y0, bc_yb in bc_configs:
    levy = StableLevySolver(a, b, h, E, nu, bc_y0, bc_yb, 100)
    L = levy.solve('uniform', q0)
    
    W = L['W']
    ny, nx = W.shape
    W_center = W[ny//2, nx//2]
    W_max = np.max(np.abs(W))
    W_max_idx = np.unravel_index(np.argmax(np.abs(W)), W.shape)
    X, Y = L['X'], L['Y']
    max_loc = (X[W_max_idx], Y[W_max_idx])
    
    W_center_coef = abs(W_center) * D / (q0 * a**4)
    W_max_coef = W_max * D / (q0 * a**4)
    
    print(f"\n{bc_name}:")
    print(f"  W_center = {W_center*1e3:.4f} mm, coef = {W_center_coef:.6f}")
    print(f"  W_max    = {W_max*1e3:.4f} mm, coef = {W_max_coef:.6f}")
    print(f"  Max location: ({max_loc[0]:.3f}, {max_loc[1]:.3f})")
    
    # Also run Ritz for comparison
    try:
        ritz = RitzSolver(a, b, h, E, nu, bc_name, 15, 15)
        R = ritz.solve('uniform', q0)
        R_W = R['W']
        R_ny, R_nx = R_W.shape
        R_center_coef = abs(R_W[R_ny//2, R_nx//2]) * D / (q0 * a**4)
        R_max_coef = R['W_coef']
        R_max_idx = np.unravel_index(np.argmax(np.abs(R_W)), R_W.shape)
        R_max_loc = (R['X'][R_max_idx], R['Y'][R_max_idx])
        
        print(f"  Ritz center coef: {R_center_coef:.6f}")
        print(f"  Ritz max coef:    {R_max_coef:.6f}")
        print(f"  Ritz max loc:     ({R_max_loc[0]:.3f}, {R_max_loc[1]:.3f})")
        
        diff_center = abs(W_center_coef - R_center_coef) / W_center_coef * 100 if W_center_coef > 0 else 0
        diff_max = abs(W_max_coef - R_max_coef) / W_max_coef * 100 if W_max_coef > 0 else 0
        print(f"  Levy-Ritz diff (center): {diff_center:.2f}%")
        print(f"  Levy-Ritz diff (max):    {diff_max:.2f}%")
    except Exception as e:
        print(f"  Ritz FAILED: {e}")

    # Run FIT for comparison
    try:
        fit = FITSolver(a, b, h, E, nu, bc_name, 100)
        F = fit.solve('uniform', q0)
        F_W = F['W']
        F_ny, F_nx = F_W.shape
        F_center_coef = abs(F_W[F_ny//2, F_nx//2]) * D / (q0 * a**4)
        F_max_coef = F['W_coef']
        print(f"  FIT center coef:  {F_center_coef:.6f}")
        print(f"  FIT max coef:     {F_max_coef:.6f}")
        
        diff_fit = abs(W_max_coef - F_max_coef) / W_max_coef * 100 if W_max_coef > 0 else 0
        print(f"  Levy-FIT diff (max):     {diff_fit:.2f}%")
    except Exception as e:
        print(f"  FIT FAILED: {e}")

print()

# ==============================================================================
# SECTION 5: FREE EDGE BC PHYSICS CHECK
# ==============================================================================
print("=" * 80)
print("SECTION 5: FREE EDGE BOUNDARY CONDITIONS PHYSICS CHECK")
print("=" * 80)

# For free edges, verify:
# 1. Moment at free edge should be ~0
# 2. Deflection at free edge should be non-zero
# 3. Max deflection should be at or near free edge

for bc_name, bc_y0, bc_yb in [('SCSF', 'C', 'F'), ('SSSF', 'S', 'F'), ('SFSF', 'F', 'F')]:
    levy = StableLevySolver(a, b, h, E, nu, bc_y0, bc_yb, 100)
    L = levy.solve('uniform', q0)
    
    W = L['W']
    My = L['My']
    ny, nx = W.shape
    
    print(f"\n{bc_name}:")
    
    # Check deflection at edges
    W_y0 = np.max(np.abs(W[0, :]))
    W_yb = np.max(np.abs(W[-1, :]))
    print(f"  Max |W| at y=0 edge: {W_y0*1e3:.6f} mm")
    print(f"  Max |W| at y=b edge: {W_yb*1e3:.6f} mm")
    
    # Check moments at edges
    My_y0 = np.max(np.abs(My[0, :]))
    My_yb = np.max(np.abs(My[-1, :]))
    My_max = np.max(np.abs(My))
    print(f"  Max |My| at y=0 edge: {My_y0:.4e} (should be {'~0 (F)' if bc_y0 == 'F' else 'nonzero'})")
    print(f"  Max |My| at y=b edge: {My_yb:.4e} (should be {'~0 (F)' if bc_yb == 'F' else 'nonzero'})")
    print(f"  Max |My| overall:     {My_max:.4e}")
    
    if bc_y0 == 'F':
        ratio = My_y0 / My_max if My_max > 0 else 0
        print(f"  My_y0/My_max = {ratio:.4f} (should be << 1 for free edge)")
    if bc_yb == 'F':
        ratio = My_yb / My_max if My_max > 0 else 0
        print(f"  My_yb/My_max = {ratio:.4f} (should be << 1 for free edge)")
    
    # Check that clamped edge has zero deflection and slope
    if bc_y0 == 'C':
        print(f"  W at y=0 (clamped): {np.max(np.abs(W[0,:]))*1e3:.6f} mm (should be ~0)")
        dy = L['Y'][1,0] - L['Y'][0,0]
        dWdy_y0 = np.max(np.abs((W[1,:] - W[0,:]) / dy))
        print(f"  dW/dy at y=0 (clamped): {dWdy_y0:.6e} (should be ~0)")

print()

# ==============================================================================
# SECTION 6: VERIFY LEVY ODE PARTICULAR SOLUTION
# ==============================================================================
print("=" * 80)
print("SECTION 6: PARTICULAR SOLUTION VERIFICATION")
print("=" * 80)

# The particular solution for the Levy ODE with uniform load is:
# Y_p = qm / (D * am^4)
# Verify: Y''''_p - 2*am^2*Y''_p + am^4*Y_p = qm/D
# Since Y_p is constant: Y''''_p = Y''_p = 0
# So am^4 * Y_p = qm/D → Y_p = qm/(D*am^4) ✓

m = 1
am = m * np.pi / a
qm = 4.0 * q0 / (m * np.pi)
Yp = qm / (D * am**4)
print(f"For m=1, am={am:.4f}:")
print(f"  qm = {qm:.4f}")
print(f"  Yp = qm / (D * am^4) = {Yp:.6e}")
print(f"  Verification: am^4 * Yp = {am**4 * Yp:.4f}")
print(f"  Expected:     qm / D   = {qm / D:.4f}")
print(f"  Match: {abs(am**4 * Yp - qm/D) / (qm/D) < 1e-10}")
print()

# ==============================================================================
# SECTION 7: BEAM FUNCTION VERIFICATION
# ==============================================================================
print("=" * 80)
print("SECTION 7: BEAM EIGENFUNCTION VERIFICATION")
print("=" * 80)

# Check that beam functions satisfy boundary conditions
xi = np.linspace(0, 1, 201)

for bc_type in ['SS', 'CC', 'CF', 'CS', 'SF', 'FF']:
    print(f"\n{bc_type} beam functions:")
    
    for n in [1, 2, 3]:
        phi = beam_function(xi, n, bc_type, 0)
        phi_p = beam_function(xi, n, bc_type, 1)
        phi_pp = beam_function(xi, n, bc_type, 2)
        
        # Check values at boundaries
        bc0, bc1 = bc_type[0], bc_type[1]
        
        issues = []
        
        # At xi = 0
        if bc0 == 'S':  # phi(0) = 0, phi''(0) = 0
            if abs(phi[0]) > 1e-6: issues.append(f"n={n}: phi(0)={phi[0]:.2e} (should be 0)")
            if abs(phi_pp[0]) > 1e-2: issues.append(f"n={n}: phi''(0)={phi_pp[0]:.2e} (should be 0)")
        elif bc0 == 'C':  # phi(0) = 0, phi'(0) = 0
            if abs(phi[0]) > 1e-6: issues.append(f"n={n}: phi(0)={phi[0]:.2e} (should be 0)")
            if abs(phi_p[0]) > 1e-2: issues.append(f"n={n}: phi'(0)={phi_p[0]:.2e} (should be 0)")
        elif bc0 == 'F':  # phi''(0) = 0, phi'''(0) = 0
            phi_ppp = beam_function(xi, n, bc_type, 3)
            if bc_type == 'FF' and n <= 2:
                pass  # Rigid body modes
            else:
                if abs(phi_pp[0]) > 1e-2: issues.append(f"n={n}: phi''(0)={phi_pp[0]:.2e} (should be ~0)")
                if abs(phi_ppp[0]) > 1e-2: issues.append(f"n={n}: phi'''(0)={phi_ppp[0]:.2e} (should be ~0)")
        
        # At xi = 1
        if bc1 == 'S':  # phi(1) = 0, phi''(1) = 0
            if abs(phi[-1]) > 1e-6: issues.append(f"n={n}: phi(1)={phi[-1]:.2e} (should be 0)")
            if abs(phi_pp[-1]) > 1e-2: issues.append(f"n={n}: phi''(1)={phi_pp[-1]:.2e} (should be 0)")
        elif bc1 == 'C':  # phi(1) = 0, phi'(1) = 0
            if abs(phi[-1]) > 1e-6: issues.append(f"n={n}: phi(1)={phi[-1]:.2e} (should be 0)")
            if abs(phi_p[-1]) > 1e-2: issues.append(f"n={n}: phi'(1)={phi_p[-1]:.2e} (should be 0)")
        elif bc1 == 'F':  # phi''(1) = 0, phi'''(1) = 0
            phi_ppp = beam_function(xi, n, bc_type, 3)
            if abs(phi_pp[-1]) > 1e-2: issues.append(f"n={n}: phi''(1)={phi_pp[-1]:.2e} (should be ~0)")
            if abs(phi_ppp[-1]) > 1e-2: issues.append(f"n={n}: phi'''(1)={phi_ppp[-1]:.2e} (should be ~0)")
        
        if issues:
            for issue in issues:
                print(f"  ⚠️  {issue}")
        else:
            print(f"  ✅ n={n}: BCs satisfied")

print()

# ==============================================================================
# SECTION 8: STRAIN ENERGY FORMULATION CHECK (Ritz)
# ==============================================================================
print("=" * 80)
print("SECTION 8: RITZ STRAIN ENERGY FORMULATION CHECK")
print("=" * 80)

# The strain energy for plate bending is:
# U = (D/2) ∫∫ [(w_xx + w_yy)^2 - 2(1-nu)(w_xx*w_yy - w_xy^2)] dA
# Expanding:
# U = (D/2) ∫∫ [w_xx^2 + 2*w_xx*w_yy + w_yy^2 - 2(1-nu)*w_xx*w_yy + 2(1-nu)*w_xy^2] dA
# U = (D/2) ∫∫ [w_xx^2 + w_yy^2 + 2*nu*w_xx*w_yy + 2(1-nu)*w_xy^2] dA
#
# This gives stiffness matrix terms:
# K = D * [Ix2*Iy0 + Ix0*Iy2 + nu*(Ix20_mp*Iy20_qn + Ix20_pm*Iy20_nq) + 2(1-nu)*Ix1*Iy1]

print("Verifying Ritz stiffness for SSSS with M=N=1 (single term):")
print("For SSSS: phi_m(x) = sin(m*pi*x/a), psi_n(y) = sin(n*pi*y/b)")
print("For m=p=n=q=1:")

# Analytical values for SS beam functions
# I0 = integral sin^2 = L/2
# I1 = (pi/L)^2 * L/2 (first deriv squared)
# I2 = (pi/L)^4 * L/2 (second deriv squared)
# Ix20 = integral sin''(m*pi*x/a) * sin(p*pi*x/a) = -(m*pi/a)^2 * L/2 for m=p

I0_analytic = a / 2
I1_analytic = (np.pi / a)**2 * a / 2
I2_analytic = (np.pi / a)**4 * a / 2
Ix20_analytic = -(np.pi / a)**2 * a / 2  # For m=p=1

print(f"  I0 = a/2 = {I0_analytic:.6f}")
print(f"  I1 = (pi/a)^2 * a/2 = {I1_analytic:.6f}")
print(f"  I2 = (pi/a)^4 * a/2 = {I2_analytic:.6f}")
print(f"  Ix20 = -(pi/a)^2 * a/2 = {Ix20_analytic:.6f}")

# K_1111 = D * [I2_x*I0_y + I0_x*I2_y + nu*(Ix20*Iy20 + Ix20*Iy20) + 2(1-nu)*I1_x*I1_y]
# For a=b=1:
K_analytic = D * (I2_analytic * I0_analytic + I0_analytic * I2_analytic 
                  + nu * (Ix20_analytic * Ix20_analytic + Ix20_analytic * Ix20_analytic) 
                  + 2 * (1 - nu) * I1_analytic * I1_analytic)

# But Ix20*Iy20 = (-pi^2/a * a/2) * (-pi^2/b * b/2) = (pi^4) / 4
# And 2 * nu * (pi^4)/4 = nu * pi^4 / 2
# Term 1+2: 2 * pi^4/a^3 * a/2 = pi^4/a^2
# Actually let me be more careful
# I2 * I0 = (pi^4/(a^3)) * (a/2) = pi^4 / (2*a^2) for each direction, both same for a=b
# So term1 + term2 = 2 * pi^4/(2*a^2) * (a/2) ... hmm, let me just compute numerically

term1 = I2_analytic * I0_analytic
term2 = I0_analytic * I2_analytic
term3 = nu * 2 * (Ix20_analytic * Ix20_analytic)  # nu * (Ix20_mp*Iy20_qn + Ix20_pm*Iy20_nq)
term4 = 2 * (1 - nu) * I1_analytic * I1_analytic

K_analytic = D * (term1 + term2 + term3 + term4)
print(f"\n  K_1111 (analytical):")
print(f"    Term1 (w_xx^2): D * Ix2 * Iy0 = {D * term1:.4f}")
print(f"    Term2 (w_yy^2): D * Ix0 * Iy2 = {D * term2:.4f}")
print(f"    Term3 (Poisson): D * nu * 2*Ix20*Iy20 = {D * term3:.4f}")
print(f"    Term4 (twist):   D * 2(1-nu) * Ix1 * Iy1 = {D * term4:.4f}")
print(f"    Total K_1111 = {K_analytic:.4f}")

# Compare with Ritz solver
ritz_1 = RitzSolver(a, b, h, E, nu, 'SSSS', 1, 1)
K_ritz = ritz_1._assemble_stiffness()
print(f"\n  Ritz K[0,0] = {K_ritz[0,0]:.4f}")
print(f"  Match: {abs(K_analytic - K_ritz[0,0]) / abs(K_analytic) < 0.01}")
print(f"  Relative diff: {abs(K_analytic - K_ritz[0,0]) / abs(K_analytic) * 100:.4f}%")

# For SSSS, single-term Ritz should give:
# w_center = F_11 / K_1111 where F_11 = q0 * (2/pi)^2 * a * b for m=n=1
F_11 = q0 * (2.0 / np.pi)**2 * a * b
W_11 = F_11 / K_analytic
W_center_coef_1term = abs(W_11) * D / (q0 * a**4)
print(f"\n  Single-term Ritz center coef: {W_center_coef_1term:.6f}")
print(f"  Expected (exact Navier):      0.004062")
print(f"  Note: single-term should overestimate (Ritz is upper bound on energy)")
print()

# ==============================================================================
# SECTION 9: ASPECT RATIO TESTS
# ==============================================================================
print("=" * 80)
print("SECTION 9: ASPECT RATIO TESTS")
print("=" * 80)

# Timoshenko Table 8: SSSS uniform, alpha (center deflection)
# a/b=1.0: alpha=0.00406
# a/b=1.2: alpha=0.00564
# a/b=1.4: alpha=0.00724
# a/b=1.6: alpha=0.00844
# a/b=2.0: alpha=0.01013
# a/b=inf: alpha=0.01302

timoshenko_ssss = {
    1.0: 0.00406,
    1.2: 0.00564,
    1.4: 0.00724,
    1.6: 0.00844,
    2.0: 0.01013,
}

print("\nSSS uniform, center deflection coefficient vs Timoshenko Table 8:")
print(f"{'a/b':>5} {'Levy':>10} {'Timoshenko':>12} {'Error':>8}")
print("-" * 40)

for ab_ratio, ref in timoshenko_ssss.items():
    a_test = ab_ratio
    b_test = 1.0
    levy = StableLevySolver(a_test, b_test, h, E, nu, 'S', 'S', 100)
    L = levy.solve('uniform', q0)
    W = L['W']
    ny, nx = W.shape
    W_center_val = abs(W[ny//2, nx//2])
    D_test = E * h**3 / (12 * (1 - nu**2))
    coef = W_center_val * D_test / (q0 * a_test**4)
    err = abs(coef - ref) / ref * 100
    print(f"{ab_ratio:5.1f} {coef:10.6f} {ref:12.6f} {err:7.2f}%")

print()

# ==============================================================================
# SECTION 10: RECIPROCITY AND SYMMETRY CHECKS
# ==============================================================================
print("=" * 80)
print("SECTION 10: RECIPROCITY AND SYMMETRY CHECKS")
print("=" * 80)

# For SSSS square plate, W(x,y) should equal W(y,x) due to symmetry
levy = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
L = levy.solve('uniform', q0)
W = L['W']

# Check W[i,j] == W[j,i] (transpose symmetry for square SSSS)
diff = np.max(np.abs(W - W.T))
max_w = np.max(np.abs(W))
print(f"SSSS symmetry check max|W-W^T|: {diff:.2e} (max|W|={max_w:.2e})")
print(f"Relative: {diff/max_w:.2e} (should be << 1)")
print()

# ==============================================================================
# SECTION 11: FIT METHOD EQUIVALENCE PROOF
# ==============================================================================
print("=" * 80)
print("SECTION 11: LEVY-FIT EQUIVALENCE CHECK")
print("=" * 80)

print("For non-SSSS Levy-type plates, FIT uses levy_ode method.")
print("Verifying mathematical equivalence:")

for bc_name, bc_y0, bc_yb in [('SCSC', 'C', 'C'), ('SCSF', 'C', 'F'), ('SSSF', 'S', 'F')]:
    levy = StableLevySolver(a, b, h, E, nu, bc_y0, bc_yb, 100)
    L = levy.solve('uniform', q0)
    
    fit = FITSolver(a, b, h, E, nu, bc_name, 100)
    F = fit.solve('uniform', q0)
    
    diff = abs(L['W_coef'] - F['W_coef']) / L['W_coef'] * 100
    print(f"  {bc_name}: Levy={L['W_coef']:.6f}, FIT={F['W_coef']:.6f}, diff={diff:.4f}%")

print()

# ==============================================================================
# SECTION 12: TEST FCFC AND CCCF FEASIBILITY (Ritz only)
# ==============================================================================
print("=" * 80)
print("SECTION 12: NON-LEVY BC FEASIBILITY (FCFC, CCCF)")
print("=" * 80)

print("FCFC: Free at x=0, Clamped at y=0, Free at x=a, Clamped at y=b")
print("  x-direction: FF (Free-Free)")
print("  y-direction: CC (Clamped-Clamped)")
print("  NOT Levy-type → Ritz only")

try:
    ritz_fcfc = RitzSolver(a, b, h, E, nu, 'FCFC', 10, 10)
    R = ritz_fcfc.solve('uniform', q0)
    print(f"  Ritz W_max: {R['W_max']*1e3:.4f} mm, coef: {R['W_coef']:.6f}")
    W = R['W']
    ny, nx = W.shape
    max_idx = np.unravel_index(np.argmax(np.abs(W)), W.shape)
    print(f"  Max location: ({R['X'][max_idx]:.3f}, {R['Y'][max_idx]:.3f})")
    print(f"  W at center: {abs(W[ny//2, nx//2])*1e3:.4f} mm")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("CCCF: Clamped at x=0, Clamped at y=0, Clamped at x=a, Free at y=b")
print("  x-direction: CC (Clamped-Clamped)")
print("  y-direction: CF (Clamped-Free)")
print("  NOT Levy-type → Ritz only")

try:
    ritz_cccf = RitzSolver(a, b, h, E, nu, 'CCCF', 10, 10)
    R = ritz_cccf.solve('uniform', q0)
    print(f"  Ritz W_max: {R['W_max']*1e3:.4f} mm, coef: {R['W_coef']:.6f}")
    W = R['W']
    ny, nx = W.shape
    max_idx = np.unravel_index(np.argmax(np.abs(W)), W.shape)
    print(f"  Max location: ({R['X'][max_idx]:.3f}, {R['Y'][max_idx]:.3f})")
    print(f"  W at center: {abs(W[ny//2, nx//2])*1e3:.4f} mm")
except Exception as e:
    print(f"  FAILED: {e}")

# Convergence study for FCFC and CCCF
print()
print("Convergence study for FCFC:")
for n in [5, 8, 10, 12, 15]:
    try:
        ritz = RitzSolver(a, b, h, E, nu, 'FCFC', n, n)
        R = ritz.solve('uniform', q0)
        print(f"  n={n:2d}: W_coef = {R['W_coef']:.6f}")
    except Exception as e:
        print(f"  n={n:2d}: FAILED: {e}")

print()
print("Convergence study for CCCF:")
for n in [5, 8, 10, 12, 15]:
    try:
        ritz = RitzSolver(a, b, h, E, nu, 'CCCF', n, n)
        R = ritz.solve('uniform', q0)
        print(f"  n={n:2d}: W_coef = {R['W_coef']:.6f}")
    except Exception as e:
        print(f"  n={n:2d}: FAILED: {e}")

print()
print("="*80)
print("DEEP AUDIT COMPLETE")
print("="*80)
