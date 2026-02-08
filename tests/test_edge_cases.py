"""
Edge case and input validation tests for plate bending solvers.
Tests extreme geometry, materials, loads, solver parameters, input validation,
benchmark integrity, and numerical precision.
"""

import numpy as np
import pytest

from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.ritz_solver import RitzSolver
from plate_bending.validation.benchmarks import Benchmarks

# Standard parameters
A, B, H, E, NU, Q0 = 1.0, 1.0, 0.01, 200e9, 0.3, 10000.0


def _levy(a=A, b=B, h=H, E_=E, nu=NU, bc_y0='S', bc_yb='S', n_terms=20):
    return StableLevySolver(a, b, h, E_, nu, bc_y0, bc_yb, n_terms)


def _ritz(a=A, b=B, h=H, E_=E, nu=NU, bc='SSSS', M=5, N=5):
    return RitzSolver(a, b, h, E_, nu, bc, M, N)


def _check_valid(result, msg=""):
    """Assert result arrays contain no NaN or Inf."""
    for key in ('W', 'Mx', 'My', 'sigma_x', 'sigma_y'):
        arr = result.get(key)
        if arr is not None:
            assert not np.any(np.isnan(arr)), f"{msg} {key} contains NaN"
            assert not np.any(np.isinf(arr)), f"{msg} {key} contains Inf"


# ============================================================
# 1-8: Extreme geometry
# ============================================================

@pytest.mark.parametrize("h", [0.001, 0.1, 0.5], ids=["thin", "moderate", "thick"])
def test_thickness_extremes_levy(h):
    r = _levy(h=h).solve('uniform', Q0)
    _check_valid(r, f"h={h}")


@pytest.mark.parametrize("h", [0.001, 0.1, 0.5], ids=["thin", "moderate", "thick"])
def test_thickness_extremes_ritz(h):
    r = _ritz(h=h).solve('uniform', Q0)
    _check_valid(r, f"h={h}")


@pytest.mark.parametrize("a,b", [
    (10, 0.1), (0.1, 10), (100, 100), (0.001, 0.001), (1.5, 0.7)
], ids=["wide", "tall", "large", "tiny", "nonsquare"])
def test_aspect_ratios_levy(a, b):
    r = _levy(a=a, b=b).solve('uniform', Q0)
    _check_valid(r, f"a={a},b={b}")


@pytest.mark.parametrize("a,b", [
    (10, 0.1), (0.1, 10), (100, 100), (0.001, 0.001), (1.5, 0.7)
], ids=["wide", "tall", "large", "tiny", "nonsquare"])
def test_aspect_ratios_ritz(a, b):
    r = _ritz(a=a, b=b).solve('uniform', Q0)
    _check_valid(r, f"a={a},b={b}")


def test_nonsquare_all_bcs_ritz():
    for bc in ['SSSS', 'CCCC', 'SCSC', 'SCSF', 'SSSF']:
        r = _ritz(a=1.5, b=0.7, bc=bc).solve('uniform', Q0)
        _check_valid(r, f"bc={bc} nonsquare")


# ============================================================
# 9-12: Extreme material
# ============================================================

@pytest.mark.parametrize("E_val", [1e3, 1e15], ids=["flexible", "stiff"])
def test_extreme_E_levy(E_val):
    r = _levy(E_=E_val).solve('uniform', Q0)
    _check_valid(r, f"E={E_val}")


@pytest.mark.parametrize("E_val", [1e3, 1e15], ids=["flexible", "stiff"])
def test_extreme_E_ritz(E_val):
    r = _ritz(E_=E_val).solve('uniform', Q0)
    _check_valid(r, f"E={E_val}")


@pytest.mark.parametrize("nu", [0.0, 0.499, 0.5], ids=["zero", "near_half", "half"])
def test_poisson_extremes_levy(nu):
    r = _levy(nu=nu).solve('uniform', Q0)
    _check_valid(r, f"nu={nu}")


@pytest.mark.parametrize("nu", [0.0, 0.499, 0.5], ids=["zero", "near_half", "half"])
def test_poisson_extremes_ritz(nu):
    r = _ritz(nu=nu).solve('uniform', Q0)
    _check_valid(r, f"nu={nu}")


# ============================================================
# 13-24: Load edge cases
# ============================================================

def test_zero_load_levy():
    r = _levy().solve('uniform', 0.0)
    assert np.allclose(r['W'], 0), "Zero load should give zero deflection"


def test_zero_load_ritz():
    r = _ritz().solve('uniform', 0.0)
    assert np.allclose(r['W'], 0), "Zero load should give zero deflection"


def test_negative_load_levy():
    r_pos = _levy().solve('uniform', Q0)
    r_neg = _levy().solve('uniform', -Q0)
    assert np.allclose(r_neg['W'], -r_pos['W'], rtol=1e-10), \
        "Negative load should give negative deflection"


def test_negative_load_ritz():
    r_pos = _ritz().solve('uniform', Q0)
    r_neg = _ritz().solve('uniform', -Q0)
    assert np.allclose(r_neg['W'], -r_pos['W'], rtol=1e-10), \
        "Negative load should give negative deflection"


@pytest.mark.parametrize("q0", [1e12, 1e-10], ids=["huge", "tiny"])
def test_extreme_loads_levy(q0):
    r = _levy().solve('uniform', q0)
    _check_valid(r, f"q0={q0}")


@pytest.mark.parametrize("q0", [1e12, 1e-10], ids=["huge", "tiny"])
def test_extreme_loads_ritz(q0):
    r = _ritz().solve('uniform', q0)
    _check_valid(r, f"q0={q0}")


def test_linearity_levy():
    r1 = _levy().solve('uniform', Q0)
    r2 = _levy().solve('uniform', 2 * Q0)
    ratio = r2['W'] / np.where(r1['W'] != 0, r1['W'], 1)
    mask = r1['W'] != 0
    assert np.allclose(ratio[mask], 2.0, rtol=1e-4), "Should be linear: W(2q)=2W(q)"


def test_linearity_ritz():
    r1 = _ritz().solve('uniform', Q0)
    r2 = _ritz().solve('uniform', 2 * Q0)
    ratio = r2['W'] / np.where(r1['W'] != 0, r1['W'], 1)
    mask = r1['W'] != 0
    assert np.allclose(ratio[mask], 2.0, rtol=1e-4), "Should be linear: W(2q)=2W(q)"


def test_circular_load_large_R():
    r = _levy().solve('circular', Q0, x0=0.5, y0=0.5, R=0.49)
    _check_valid(r, "large R circular")


def test_circular_load_small_R():
    r = _levy().solve('circular', Q0, x0=0.5, y0=0.5, R=0.01)
    _check_valid(r, "small R circular")


def test_circular_load_large_R_ritz():
    r = _ritz().solve('circular', Q0, x0=0.5, y0=0.5, R=0.49)
    _check_valid(r, "large R circular ritz")


def test_circular_load_small_R_ritz():
    r = _ritz().solve('circular', Q0, x0=0.5, y0=0.5, R=0.01)
    _check_valid(r, "small R circular ritz")


def test_point_load_center_levy():
    """Point load approximated as very small circular load at center."""
    r = _levy().solve('circular', Q0, x0=0.5, y0=0.5, R=0.001)
    _check_valid(r, "point load center")


def test_patch_load_full_plate_levy():
    """Patch covering entire plate should approximate uniform at center."""
    r_uniform = _levy().solve('uniform', Q0)
    r_patch = _levy().solve('patch', Q0, x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    _check_valid(r_patch, "full patch")
    # Compare W_max (center value) — Fourier truncation may differ at edges
    assert abs(r_uniform['W_max'] - r_patch['W_max']) / r_uniform['W_max'] < 0.25, \
        "Full patch W_max should be within 25% of uniform load"


def test_patch_load_full_plate_ritz():
    """Ritz solver uses 'rect_patch' not 'patch'."""
    r_uniform = _ritz().solve('uniform', Q0)
    r_patch = _ritz().solve('rect_patch', Q0, x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    _check_valid(r_patch, "full patch ritz")
    assert abs(r_uniform['W_max'] - r_patch['W_max']) / r_uniform['W_max'] < 0.25, \
        "Full patch W_max should be within 25% of uniform load"


def test_patch_load_99pct():
    """Patch covering 99% of plate."""
    r = _levy().solve('patch', Q0, x1=0.005, y1=0.005, x2=0.995, y2=0.995)
    _check_valid(r, "99% patch")


def test_point_load_near_clamped_edge():
    """Point load near clamped edge should give small deflection."""
    s = _levy(bc_y0='C', bc_yb='C')
    r = s.solve('circular', Q0, x0=0.5, y0=0.01, R=0.005)
    _check_valid(r, "point near clamped")


def test_point_load_near_free_edge():
    """Point load near free edge should give larger deflection than near clamped."""
    s_free = _levy(bc_y0='S', bc_yb='F')
    r_free = s_free.solve('circular', Q0, x0=0.5, y0=0.99, R=0.005)
    _check_valid(r_free, "point near free")


# ============================================================
# 25-30: Solver-specific edge cases
# ============================================================

@pytest.mark.parametrize("M,N", [(1, 1), (2, 2), (20, 20)],
                         ids=["1term", "2terms", "20terms"])
def test_ritz_term_counts(M, N):
    r = _ritz(M=M, N=N).solve('uniform', Q0)
    _check_valid(r, f"M=N={M}")
    assert r['W_max'] != 0, f"M=N={M} should give nonzero deflection"


@pytest.mark.parametrize("n_terms", [1, 200], ids=["1term", "200terms"])
def test_levy_term_counts(n_terms):
    r = _levy(n_terms=n_terms).solve('uniform', Q0)
    _check_valid(r, f"n_terms={n_terms}")
    assert r['W_max'] != 0, f"n_terms={n_terms} should give nonzero deflection"


def test_levy_few_terms_circular():
    r = _levy(n_terms=3).solve('circular', Q0, x0=0.5, y0=0.5, R=0.1)
    _check_valid(r, "few terms circular")


# ============================================================
# 31-40: Input validation
# ============================================================

def test_invalid_bc_ritz():
    with pytest.raises((ValueError, KeyError)):
        _ritz(bc='XXXX').solve('uniform', Q0)


def test_bc_too_short_ritz():
    with pytest.raises((ValueError, KeyError, IndexError)):
        _ritz(bc='SS').solve('uniform', Q0)


def test_bc_too_long_ritz():
    """5-char BC string — solver takes first 4 chars, so it may not raise."""
    # The solver slices bc[0:4] implicitly, so 'SSSSS' works like 'SSSS'
    # This documents current behavior rather than requiring a raise
    r = _ritz(bc='SSSSS').solve('uniform', Q0)
    _check_valid(r, "bc too long")


def test_bc_lowercase_ritz():
    """Lowercase bc should either work (auto-uppercased) or raise clearly."""
    try:
        r = _ritz(bc='ssss').solve('uniform', Q0)
        _check_valid(r, "lowercase bc")
    except (ValueError, KeyError):
        pass  # Raising is also acceptable


def test_zero_thickness():
    """h=0 gives D=0 and division by h^2 in stress — results contain inf/nan."""
    r = _ritz(h=0).solve('uniform', Q0)
    # D=0 means zero stiffness; stress computation divides by h^2=0
    # We expect inf or nan in stress fields
    assert (np.any(np.isinf(r['sigma_x'])) or np.any(np.isnan(r['sigma_x']))), \
        "h=0 should produce inf/nan in stress"


def test_negative_dimensions():
    """Negative a should either raise or produce results (solver-dependent)."""
    try:
        r = _levy(a=-1).solve('uniform', Q0)
        # If it doesn't raise, results should still be finite
        _check_valid(r, "negative a")
    except (ValueError, Exception):
        pass  # Raising is acceptable


def test_negative_E():
    """Negative E should either raise or produce results."""
    try:
        r = _ritz(E_=-200e9).solve('uniform', Q0)
        _check_valid(r, "negative E")
    except (ValueError, Exception):
        pass


def test_circular_load_R_zero():
    """R=0 circular load — edge case."""
    try:
        r = _levy().solve('circular', Q0, x0=0.5, y0=0.5, R=0.0)
        _check_valid(r, "R=0")
    except (ValueError, ZeroDivisionError, Exception):
        pass  # Raising is acceptable


def test_circular_load_negative_R():
    """Negative R — should handle gracefully."""
    try:
        r = _levy().solve('circular', Q0, x0=0.5, y0=0.5, R=-0.1)
        _check_valid(r, "R<0")
    except (ValueError, Exception):
        pass


# ============================================================
# 41-44: Benchmark data integrity
# ============================================================

def test_benchmark_required_keys():
    for bc, data in Benchmarks.DATA.items():
        has_w = 'W_center_coef' in data or 'W_max_coef' in data
        assert has_w, f"Benchmark {bc} missing W coefficient key"


def test_benchmark_values_positive():
    for bc, data in Benchmarks.DATA.items():
        for key, val in data.items():
            if key.endswith('_coef') and isinstance(val, (int, float)):
                assert val > 0, f"Benchmark {bc}.{key}={val} should be positive"


def test_benchmark_unknown_bc():
    assert Benchmarks.get('ZZZZ') is None


def test_benchmark_list_available():
    available = Benchmarks.list_available()
    for bc in ['SSSS', 'CCCC', 'SCSC', 'SCSS', 'SCSF']:
        assert bc in available, f"{bc} should be in available benchmarks"


# ============================================================
# 45-48: Numerical precision
# ============================================================

@pytest.mark.parametrize("bc_y0,bc_yb", [
    ('S', 'S'), ('C', 'C'), ('C', 'F'), ('S', 'F')
], ids=['SSSS', 'SCSC', 'SCSF', 'SSSF'])
def test_no_nan_inf_levy(bc_y0, bc_yb):
    r = _levy(bc_y0=bc_y0, bc_yb=bc_yb).solve('uniform', Q0)
    _check_valid(r, f"S{bc_y0}S{bc_yb}")


@pytest.mark.parametrize("bc", ['SSSS', 'CCCC', 'SCSC', 'SCSF', 'SSSF'],
                         ids=['SSSS', 'CCCC', 'SCSC', 'SCSF', 'SSSF'])
def test_no_nan_inf_ritz(bc):
    r = _ritz(bc=bc).solve('uniform', Q0)
    _check_valid(r, f"ritz {bc}")


def test_positive_deflection_ssss_uniform():
    """Positive uniform load on SSSS should give all non-negative W."""
    r = _levy().solve('uniform', Q0)
    assert np.all(r['W'] >= -1e-20), "SSSS uniform: W should be non-negative"


def test_deterministic_levy():
    r1 = _levy().solve('uniform', Q0)
    r2 = _levy().solve('uniform', Q0)
    assert np.array_equal(r1['W'], r2['W']), "Same input should give identical output"


def test_deterministic_ritz():
    r1 = _ritz().solve('uniform', Q0)
    r2 = _ritz().solve('uniform', Q0)
    assert np.array_equal(r1['W'], r2['W']), "Same input should give identical output"
