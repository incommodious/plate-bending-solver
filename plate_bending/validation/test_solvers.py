"""
Test Suite for Plate Bending Solvers
=====================================

Validates solvers against classical benchmarks.
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.fit_solver import FITSolver
from plate_bending.validation.benchmarks import Benchmarks


def test_ssss_uniform():
    """Test SSSS (all simply supported) with uniform load."""
    print("\n" + "="*70)
    print("TEST: SSSS with uniform load")
    print("="*70)

    a, b, h = 1.0, 1.0, 0.01
    E, nu = 2.1e11, 0.3
    q0 = 10000
    n_terms = 100

    bench = Benchmarks.get('SSSS')
    expected = bench['W_center_coef']

    solver = StableLevySolver(a, b, h, E, nu, 'S', 'S', n_terms)
    results = solver.solve('uniform', q0)

    actual = results['W_coef']
    error = abs(actual - expected) / expected * 100

    print(f"  Expected W_coef: {expected:.6f}")
    print(f"  Actual W_coef:   {actual:.6f}")
    print(f"  Error:           {error:.2f}%")
    print(f"  Source:          {bench['source']}")

    status = "PASS" if error < 5 else "FAIL"
    print(f"  Result:          {status}")

    return error < 5


def test_scsf_uniform():
    """Test SCSF (clamped y=0, free y=b) with uniform load."""
    print("\n" + "="*70)
    print("TEST: SCSF with uniform load")
    print("="*70)

    a, b, h = 1.0, 1.0, 0.01
    E, nu = 2.1e11, 0.3
    q0 = 10000
    n_terms = 100

    bench = Benchmarks.get('SCSF')
    expected = bench['W_max_coef']

    solver = StableLevySolver(a, b, h, E, nu, 'C', 'F', n_terms)
    results = solver.solve('uniform', q0)

    actual = results['W_coef']
    error = abs(actual - expected) / expected * 100

    print(f"  Expected W_coef: {expected:.6f}")
    print(f"  Actual W_coef:   {actual:.6f}")
    print(f"  Error:           {error:.2f}%")
    print(f"  Source:          {bench['source']}")

    status = "PASS" if error < 5 else "FAIL"
    print(f"  Result:          {status}")

    return error < 5


def test_sssf_uniform():
    """Test SSSF (simply supported y=0, free y=b) with uniform load."""
    print("\n" + "="*70)
    print("TEST: SSSF with uniform load")
    print("="*70)

    a, b, h = 1.0, 1.0, 0.01
    E, nu = 2.1e11, 0.3
    q0 = 10000
    n_terms = 100

    bench = Benchmarks.get('SSSF')
    expected = bench['W_max_coef']

    solver = StableLevySolver(a, b, h, E, nu, 'S', 'F', n_terms)
    results = solver.solve('uniform', q0)

    actual = results['W_coef']
    error = abs(actual - expected) / expected * 100

    print(f"  Expected W_coef: {expected:.6f}")
    print(f"  Actual W_coef:   {actual:.6f}")
    print(f"  Error:           {error:.2f}%")
    print(f"  Source:          {bench['source']}")

    status = "PASS" if error < 5 else "FAIL"
    print(f"  Result:          {status}")

    return error < 5


def test_scsc_uniform():
    """Test SCSC (clamped both edges) with uniform load."""
    print("\n" + "="*70)
    print("TEST: SCSC with uniform load")
    print("="*70)

    a, b, h = 1.0, 1.0, 0.01
    E, nu = 2.1e11, 0.3
    q0 = 10000
    n_terms = 100

    bench = Benchmarks.get('SCSC')
    expected = bench['W_center_coef']

    solver = StableLevySolver(a, b, h, E, nu, 'C', 'C', n_terms)
    results = solver.solve('uniform', q0)

    actual = results['W_coef']
    error = abs(actual - expected) / expected * 100

    print(f"  Expected W_coef: {expected:.6f}")
    print(f"  Actual W_coef:   {actual:.6f}")
    print(f"  Error:           {error:.2f}%")
    print(f"  Source:          {bench['source']}")

    status = "PASS" if error < 5 else "FAIL"
    print(f"  Result:          {status}")

    return error < 5


def test_levy_vs_fit():
    """Compare Levy and FIT methods for SSSS case."""
    print("\n" + "="*70)
    print("TEST: Levy vs FIT comparison (SSSS)")
    print("="*70)

    a, b, h = 1.0, 1.0, 0.01
    E, nu = 2.1e11, 0.3
    q0 = 10000

    levy = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
    levy_results = levy.solve('uniform', q0)

    fit = FITSolver(a, b, h, E, nu, 'SSSS', 50)
    fit_results = fit.solve('uniform', q0)

    levy_val = levy_results['W_coef']
    fit_val = fit_results['W_coef']
    diff = abs(levy_val - fit_val) / levy_val * 100

    print(f"  Levy W_coef:     {levy_val:.6f}")
    print(f"  FIT W_coef:      {fit_val:.6f}")
    print(f"  Difference:      {diff:.2f}%")

    status = "PASS" if diff < 3 else "FAIL"
    print(f"  Result:          {status}")

    return diff < 3


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print(" PLATE BENDING SOLVER VALIDATION SUITE")
    print(" Testing sign corrections for free edge boundary conditions")
    print("="*70)

    results = []

    results.append(("SSSS uniform", test_ssss_uniform()))
    results.append(("SCSC uniform", test_scsc_uniform()))
    results.append(("SCSF uniform", test_scsf_uniform()))
    results.append(("SSSF uniform", test_sssf_uniform()))
    results.append(("Levy vs FIT", test_levy_vs_fit()))

    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, passed_test in results:
        status = "PASS" if passed_test else "FAIL"
        print(f"  {name:20s} {status}")

    print(f"\n  Total: {passed}/{total} passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
    else:
        print(f"\n  {total - passed} test(s) FAILED")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
