"""Test the Ritz solver against Levy and benchmarks."""

import numpy as np
from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.ritz_solver import RitzSolver

# Parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000

print("="*70)
print("RITZ SOLVER VALIDATION")
print("="*70)
print()

# Test 1: SSSS (all simply supported)
print("Test 1: SSSS (all simply supported)")
print("-"*50)

levy = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
L = levy.solve('uniform', q0)

ritz = RitzSolver(a, b, h, E, nu, 'SSSS', M=10, N=10)
R = ritz.solve('uniform', q0)

print(f"  Levy W_coef:  {L['W_coef']:.6f}")
print(f"  Ritz W_coef:  {R['W_coef']:.6f}")
print(f"  Expected:     0.00406")
print(f"  Levy error:   {abs(L['W_coef'] - 0.00406) / 0.00406 * 100:.2f}%")
print(f"  Ritz error:   {abs(R['W_coef'] - 0.00406) / 0.00406 * 100:.2f}%")
print(f"  Levy-Ritz:    {abs(L['W_coef'] - R['W_coef']) / L['W_coef'] * 100:.2f}%")
print()

# Test 2: SSSF (simply supported - free)
print("Test 2: SSSF (simply supported at y=0, free at y=b)")
print("-"*50)

levy = StableLevySolver(a, b, h, E, nu, 'S', 'F', 100)
L = levy.solve('uniform', q0)

ritz = RitzSolver(a, b, h, E, nu, 'SSSF', M=10, N=10)
R = ritz.solve('uniform', q0)

print(f"  Levy W_coef:  {L['W_coef']:.6f}")
print(f"  Ritz W_coef:  {R['W_coef']:.6f}")
print(f"  Expected:     0.01286")
print(f"  Levy error:   {abs(L['W_coef'] - 0.01286) / 0.01286 * 100:.2f}%")
print(f"  Ritz error:   {abs(R['W_coef'] - 0.01286) / 0.01286 * 100:.2f}%")
print(f"  Levy-Ritz:    {abs(L['W_coef'] - R['W_coef']) / L['W_coef'] * 100:.2f}%")
print()

# Test 3: SCSF (clamped - free)
print("Test 3: SCSF (clamped at y=0, free at y=b)")
print("-"*50)

levy = StableLevySolver(a, b, h, E, nu, 'C', 'F', 100)
L = levy.solve('uniform', q0)

ritz = RitzSolver(a, b, h, E, nu, 'SCSF', M=10, N=10)
R = ritz.solve('uniform', q0)

print(f"  Levy W_coef:  {L['W_coef']:.6f}")
print(f"  Ritz W_coef:  {R['W_coef']:.6f}")
print(f"  Benchmark:    0.01377 (Szilard - may have different convention)")
print(f"  Levy-Ritz:    {abs(L['W_coef'] - R['W_coef']) / L['W_coef'] * 100:.2f}%")
print()

# Test 4: Convergence study for SSSS
print("Test 4: Ritz convergence study for SSSS")
print("-"*50)

for n in [3, 5, 8, 10, 12, 15]:
    ritz = RitzSolver(a, b, h, E, nu, 'SSSS', M=n, N=n)
    R = ritz.solve('uniform', q0)
    error = abs(R['W_coef'] - 0.00406) / 0.00406 * 100
    print(f"  n={n:2d}: W_coef = {R['W_coef']:.6f}, error = {error:.2f}%")
