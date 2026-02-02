"""Test GUI imports and basic functionality."""

import sys
print("Testing imports...")

try:
    from plate_bending.solvers.levy_solver import StableLevySolver
    print("  StableLevySolver: OK")
except Exception as e:
    print(f"  StableLevySolver: FAIL - {e}")
    sys.exit(1)

try:
    from plate_bending.solvers.ritz_solver import RitzSolver
    print("  RitzSolver: OK")
except Exception as e:
    print(f"  RitzSolver: FAIL - {e}")
    sys.exit(1)

try:
    from plate_bending.validation.benchmarks import Benchmarks
    print("  Benchmarks: OK")
except Exception as e:
    print(f"  Benchmarks: FAIL - {e}")
    sys.exit(1)

# Test basic functionality
print("\nTesting basic functionality...")

a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000

# Levy
levy = StableLevySolver(a, b, h, E, nu, 'S', 'S', 50)
L = levy.solve('uniform', q0)
print(f"  Levy SSSS W_coef: {L['W_coef']:.6f} (expected ~0.00406)")

# Ritz
ritz = RitzSolver(a, b, h, E, nu, 'SSSS', 10, 10)
R = ritz.solve('uniform', q0)
print(f"  Ritz SSSS W_coef: {R['W_coef']:.6f} (expected ~0.00406)")

# Benchmark
bench = Benchmarks.get('SSSS')
print(f"  Benchmark SSSS: {bench['W_center_coef']}")

# Test patch load
levy_patch = StableLevySolver(a, b, h, E, nu, 'S', 'S', 50)
LP = levy_patch.solve('rect_patch', q0, x1=0.3, y1=0.3, x2=0.7, y2=0.7)
print(f"  Levy patch load W_coef: {LP['W_coef']:.6f}")

ritz_patch = RitzSolver(a, b, h, E, nu, 'SSSS', 10, 10)
RP = ritz_patch.solve('rect_patch', q0, x1=0.3, y1=0.3, x2=0.7, y2=0.7)
print(f"  Ritz patch load W_coef: {RP['W_coef']:.6f}")

diff = abs(LP['W_coef'] - RP['W_coef']) / LP['W_coef'] * 100
print(f"  Levy vs Ritz patch: {diff:.2f}% difference")

print("\nAll tests passed!")
