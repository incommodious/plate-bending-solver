"""Test SSSF convergence with increasing terms."""

import numpy as np
from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.ritz_solver import RitzSolver

# Parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000

print("="*70)
print("SSSF CONVERGENCE STUDY")
print("="*70)
print()

# Levy reference
levy = StableLevySolver(a, b, h, E, nu, 'S', 'F', 100)
L = levy.solve('uniform', q0)
print(f"Levy W_coef: {L['W_coef']:.6f}")
print(f"Benchmark:   0.01286")
print()

print("Ritz convergence:")
print("-"*50)
for n in [3, 5, 8, 10, 15, 20, 25]:
    ritz = RitzSolver(a, b, h, E, nu, 'SSSF', M=n, N=n)
    R = ritz.solve('uniform', q0)
    levy_diff = abs(R['W_coef'] - L['W_coef']) / L['W_coef'] * 100
    bench_err = abs(R['W_coef'] - 0.01286) / 0.01286 * 100
    print(f"  n={n:2d}: W_coef = {R['W_coef']:.6f}, vs Levy: {levy_diff:.1f}%, vs bench: {bench_err:.1f}%")
