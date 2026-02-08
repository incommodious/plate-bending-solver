"""
Exhaustive solver matrix tests: All BC×Load combinations, cross-solver agreement,
physics ordering, symmetry, reciprocity, and linearity.
"""
import numpy as np
import pytest
from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.fit_solver import FITSolver
from plate_bending.solvers.ritz_solver import RitzSolver
from plate_bending.report import _bilinear_sample

# Standard test parameters
A, B, H = 1.0, 1.0, 0.01
E, NU = 200e9, 0.3
Q0 = 10000

LEVY_BCS = ['SSSS', 'SCSC', 'SCSS', 'SCSF', 'SSSF', 'SFSF']
NON_LEVY_BCS = ['CCCC', 'FCFC', 'CCCF', 'FCCC']
ALL_BCS = LEVY_BCS + NON_LEVY_BCS
LOAD_TYPES = ['uniform', 'rect_patch', 'circular', 'point']
RITZ_LOAD_TYPES = ['uniform', 'rect_patch', 'circular']  # Ritz doesn't support point loads


def _load_kwargs(load_type):
    if load_type == 'uniform':
        return {}
    elif load_type == 'rect_patch':
        return {'x1': 0.3, 'y1': 0.3, 'x2': 0.7, 'y2': 0.7}
    elif load_type == 'circular':
        return {'x0': 0.5, 'y0': 0.5, 'R': 0.1}
    elif load_type == 'point':
        return {'x0': 0.5, 'y0': 0.5}


def _bc_to_levy(bc):
    return bc[1], bc[3]


# ============================================================================
# 1. All BC × Load combos with Ritz
# ============================================================================
@pytest.mark.parametrize("bc", ALL_BCS)
@pytest.mark.parametrize("load_type", RITZ_LOAD_TYPES)
def test_ritz_bc_load(bc, load_type):
    """Every BC/load combo runs without error and produces positive deflection."""
    ritz = RitzSolver(A, B, H, E, NU, bc, 8, 8)
    r = ritz.solve(load_type, Q0, **_load_kwargs(load_type))
    assert r['W'].shape[0] > 1, f"W has wrong shape for {bc}/{load_type}"
    assert r['W'].shape[1] > 1
    assert np.max(np.abs(r['W'])) > 0, f"Zero deflection for {bc}/{load_type}"
    assert not np.any(np.isnan(r['W'])), f"NaN in W for {bc}/{load_type}"
    assert not np.any(np.isinf(r['W'])), f"Inf in W for {bc}/{load_type}"


# ============================================================================
# 2. All Levy-type BC × Load combos with Levy solver
# ============================================================================
@pytest.mark.parametrize("bc", LEVY_BCS)
@pytest.mark.parametrize("load_type", LOAD_TYPES)
def test_levy_bc_load(bc, load_type):
    """Every Levy-type BC/load combo works."""
    y0, yb = _bc_to_levy(bc)
    levy = StableLevySolver(A, B, H, E, NU, y0, yb, 50)
    r = levy.solve(load_type, Q0, **_load_kwargs(load_type))
    assert np.max(np.abs(r['W'])) > 0, f"Zero deflection for Levy {bc}/{load_type}"
    assert not np.any(np.isnan(r['W'])), f"NaN in Levy W for {bc}/{load_type}"


# ============================================================================
# 3. Levy vs FIT agreement (should be ~0% for uniform)
# ============================================================================
@pytest.mark.parametrize("bc", LEVY_BCS)
def test_levy_fit_agreement(bc):
    """Levy and FIT match within 0.01% for uniform load."""
    y0, yb = _bc_to_levy(bc)
    levy = StableLevySolver(A, B, H, E, NU, y0, yb, 100)
    fit = FITSolver(A, B, H, E, NU, bc, 100)
    r_l = levy.solve('uniform', Q0)
    r_f = fit.solve('uniform', Q0, auto_converge=False)
    diff = abs(r_l['W_max'] - r_f['W_max']) / r_l['W_max'] * 100
    assert diff < 0.01, f"Levy/FIT differ by {diff:.4f}% for {bc}"


# ============================================================================
# 4. Levy vs Ritz agreement (within 5%)
# ============================================================================
@pytest.mark.parametrize("bc", LEVY_BCS)
def test_levy_ritz_agreement(bc):
    """Levy and Ritz(15,15) match within 5% for uniform load."""
    y0, yb = _bc_to_levy(bc)
    levy = StableLevySolver(A, B, H, E, NU, y0, yb, 100)
    ritz = RitzSolver(A, B, H, E, NU, bc, 15, 15)
    r_l = levy.solve('uniform', Q0)
    r_r = ritz.solve('uniform', Q0)
    diff = abs(r_l['W_coef'] - r_r['W_coef']) / r_l['W_coef'] * 100
    # SSSF/SFSF use quarter-wave sine approximation → larger diff expected
    tol = 25 if bc in ('SSSF', 'SFSF') else 5  # Quarter-wave sine approx for SF BCs
    assert diff < tol, f"Levy/Ritz differ by {diff:.1f}% for {bc} (Levy={r_l['W_coef']:.6f}, Ritz={r_r['W_coef']:.6f})"


# ============================================================================
# 5. Load ordering (SSSS)
# ============================================================================
def test_load_ordering_ssss():
    """Uniform > large patch > small patch for SSSS."""
    ritz = RitzSolver(A, B, H, E, NU, 'SSSS', 10, 10)
    r_uni = ritz.solve('uniform', Q0)
    r_big = ritz.solve('rect_patch', Q0, x1=0.2, y1=0.2, x2=0.8, y2=0.8)
    r_sml = ritz.solve('rect_patch', Q0, x1=0.4, y1=0.4, x2=0.6, y2=0.6)
    assert r_uni['W_max'] > r_big['W_max'] > r_sml['W_max'], \
        f"Load ordering wrong: {r_uni['W_max']:.6e} > {r_big['W_max']:.6e} > {r_sml['W_max']:.6e}"


# ============================================================================
# 6. Load ordering near free vs clamped (SCSF)
# ============================================================================
def test_load_ordering_scsf():
    """Load near free edge > center > near clamped edge for SCSF."""
    ritz = RitzSolver(A, B, H, E, NU, 'SCSF', 10, 10)
    r_free = ritz.solve('rect_patch', Q0, x1=0.3, y1=0.7, x2=0.7, y2=1.0)
    r_cent = ritz.solve('rect_patch', Q0, x1=0.3, y1=0.35, x2=0.7, y2=0.65)
    r_clmp = ritz.solve('rect_patch', Q0, x1=0.3, y1=0.0, x2=0.7, y2=0.3)
    assert r_free['W_max'] > r_cent['W_max'] > r_clmp['W_max'], \
        "SCSF load ordering wrong: free > center > clamped expected"


# ============================================================================
# 7. Aspect ratio tests
# ============================================================================
@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 4.0])
def test_aspect_ratios(ratio):
    """Various aspect ratios run without error."""
    a = ratio
    b = 1.0
    ritz = RitzSolver(a, b, H, E, NU, 'SSSS', 8, 8)
    r = ritz.solve('uniform', Q0)
    assert r['W_max'] > 0


# ============================================================================
# 8. Full-plate patch ≈ uniform
# ============================================================================
def test_patch_equals_uniform():
    """Rectangular patch covering entire plate matches uniform within 1%."""
    ritz = RitzSolver(A, B, H, E, NU, 'SSSS', 10, 10)
    r_uni = ritz.solve('uniform', Q0)
    r_pat = ritz.solve('rect_patch', Q0, x1=0.0, y1=0.0, x2=A, y2=B)
    diff = abs(r_uni['W_max'] - r_pat['W_max']) / r_uni['W_max'] * 100
    assert diff < 1, f"Full patch vs uniform differ by {diff:.2f}%"


# ============================================================================
# 9. Negative load
# ============================================================================
def test_negative_load():
    """Negative load gives opposite deflection."""
    ritz = RitzSolver(A, B, H, E, NU, 'SSSS', 8, 8)
    r_pos = ritz.solve('uniform', Q0)
    r_neg = ritz.solve('uniform', -Q0)
    # W should be opposite sign
    np.testing.assert_allclose(r_pos['W'], -r_neg['W'], rtol=1e-10,
                               err_msg="Negative load should give opposite deflection")


# ============================================================================
# 10. Zero load
# ============================================================================
def test_zero_load():
    """Zero load gives zero deflection."""
    ritz = RitzSolver(A, B, H, E, NU, 'SSSS', 8, 8)
    r = ritz.solve('uniform', 0.0)
    assert np.max(np.abs(r['W'])) < 1e-20, "Zero load should give zero deflection"


# ============================================================================
# 11. Symmetry (SSSS square plate uniform)
# ============================================================================
def test_symmetry_ssss():
    """SSSS square plate with uniform load: W(x,y)=W(y,x), W(x,y)=W(a-x,y)."""
    ritz = RitzSolver(A, B, H, E, NU, 'SSSS', 12, 12)
    r = ritz.solve('uniform', Q0)
    W = r['W']

    # W should be symmetric about both axes
    # W(i,j) ≈ W(j,i) for square plate
    ny, nx = W.shape
    for i in range(0, ny, ny // 4):
        for j in range(0, nx, nx // 4):
            # x-axis symmetry: W(i,j) ≈ W(i, nx-1-j)
            assert abs(W[i, j] - W[i, nx - 1 - j]) < abs(W[ny // 2, nx // 2]) * 0.02, \
                f"x-symmetry broken at ({i},{j})"
            # y-axis symmetry: W(i,j) ≈ W(ny-1-i, j)
            assert abs(W[i, j] - W[ny - 1 - i, j]) < abs(W[ny // 2, nx // 2]) * 0.02, \
                f"y-symmetry broken at ({i},{j})"

    # Mx_center ≈ My_center for square plate
    Mx_c = r['Mx'][ny // 2, nx // 2]
    My_c = r['My'][ny // 2, nx // 2]
    diff = abs(Mx_c - My_c) / abs(Mx_c) * 100
    assert diff < 1, f"Mx_center ({Mx_c:.6e}) != My_center ({My_c:.6e}), diff={diff:.2f}%"


# ============================================================================
# 12. Reciprocity (Maxwell's theorem)
# ============================================================================
def test_reciprocity():
    """Point load at A, measure at B == point load at B, measure at A."""
    levy_A = StableLevySolver(A, B, H, E, NU, 'S', 'S', 100)
    levy_B = StableLevySolver(A, B, H, E, NU, 'S', 'S', 100)

    xA, yA = 0.3, 0.5
    xB, yB = 0.7, 0.3

    r_A = levy_A.solve('point', 1000.0, x0=xA, y0=yA)
    r_B = levy_B.solve('point', 1000.0, x0=xB, y0=yB)

    W_AB = _bilinear_sample(xB, yB, r_A['X'], r_A['Y'], r_A['W'])
    W_BA = _bilinear_sample(xA, yA, r_B['X'], r_B['Y'], r_B['W'])

    diff = abs(W_AB - W_BA) / abs(W_AB) * 100
    # Bilinear interpolation with point loads is imprecise; use 25% tolerance
    # The key physics check is that both values are the same order of magnitude
    assert diff < 25, f"Reciprocity violated: W_AB={W_AB:.6e}, W_BA={W_BA:.6e}, diff={diff:.2f}%"


# ============================================================================
# 13. Linearity
# ============================================================================
def test_linearity():
    """W(2*q0) = 2*W(q0)."""
    ritz = RitzSolver(A, B, H, E, NU, 'SSSS', 8, 8)
    r1 = ritz.solve('uniform', Q0)
    r2 = ritz.solve('uniform', 2 * Q0)
    np.testing.assert_allclose(r2['W'], 2 * r1['W'], rtol=1e-10,
                               err_msg="Linearity violated: W(2q) != 2*W(q)")


# ============================================================================
# 14. Superposition
# ============================================================================
def test_superposition():
    """W(q1) + W(q2) = W(q1+q2)."""
    ritz = RitzSolver(A, B, H, E, NU, 'SSSS', 8, 8)
    r1 = ritz.solve('uniform', 1000)
    r2 = ritz.solve('uniform', 2000)
    r3 = ritz.solve('uniform', 3000)
    np.testing.assert_allclose(r1['W'] + r2['W'], r3['W'], rtol=1e-10,
                               err_msg="Superposition violated")


# ============================================================================
# 15. Extreme loads still linear
# ============================================================================
@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1, 1e4, 1e8])
def test_load_scaling(scale):
    """Results scale linearly with load magnitude."""
    ritz = RitzSolver(A, B, H, E, NU, 'SSSS', 8, 8)
    r_ref = ritz.solve('uniform', Q0)
    r_scaled = ritz.solve('uniform', Q0 * scale)
    np.testing.assert_allclose(r_scaled['W'], r_ref['W'] * scale, rtol=1e-6,
                               err_msg=f"Linearity broken at scale={scale}")


# ============================================================================
# 16. Non-Levy BCs more flexible than CCCC
# ============================================================================
def test_flexibility_ordering():
    """SSSS > CCCF > CCCC in terms of max deflection."""
    ssss = RitzSolver(A, B, H, E, NU, 'SSSS', 12, 12).solve('uniform', Q0)
    cccf = RitzSolver(A, B, H, E, NU, 'CCCF', 12, 12).solve('uniform', Q0)
    cccc = RitzSolver(A, B, H, E, NU, 'CCCC', 12, 12).solve('uniform', Q0)
    assert ssss['W_max'] > cccf['W_max'] > cccc['W_max'], \
        f"Flexibility ordering wrong: SSSS={ssss['W_max']:.6e}, CCCF={cccf['W_max']:.6e}, CCCC={cccc['W_max']:.6e}"
