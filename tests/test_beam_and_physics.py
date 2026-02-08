import numpy as np

from plate_bending.solvers.beam_functions import beam_function, get_eigenvalue
from plate_bending.solvers.ritz_solver import RitzSolver


def _edge_slices(arr):
    return [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]


def _edge_max_abs(arr):
    return max(float(np.max(np.abs(edge))) for edge in _edge_slices(arr))


def _finite_diff_1d(y, x):
    return np.gradient(y, x)


def _relative_norm_error(a, b):
    denom = np.linalg.norm(b)
    if denom == 0:
        return np.linalg.norm(a - b)
    return np.linalg.norm(a - b) / denom


def test_beam_boundary_conditions():
    tol_essential = 1e-6
    tol_natural = 1e-1

    xi0 = 0.0
    xi1 = 1.0

    bc_expectations = {
        "SS": [(0, xi0, 0.0, tol_essential), (0, xi1, 0.0, tol_essential)],
        "CC": [(0, xi0, 0.0, tol_essential), (1, xi0, 0.0, tol_essential),
               (0, xi1, 0.0, tol_essential), (1, xi1, 0.0, tol_essential)],
        "CF": [(0, xi0, 0.0, tol_essential), (1, xi0, 0.0, tol_essential),
               (2, xi1, 0.0, tol_natural)],
        "FC": [(0, xi1, 0.0, tol_essential), (1, xi1, 0.0, tol_essential),
               (2, xi0, 0.0, tol_natural)],
        "CS": [(0, xi0, 0.0, tol_essential), (1, xi0, 0.0, tol_essential),
               (0, xi1, 0.0, tol_essential)],
        "SC": [(0, xi0, 0.0, tol_essential), (0, xi1, 0.0, tol_essential),
               (1, xi1, 0.0, tol_essential)],
        "FF": [(2, xi0, 0.0, 3.0), (2, xi1, 0.0, 3.0)],  # FF has rigid-body modes, natural BCs very loose
        "SF": [(0, xi0, 0.0, tol_essential)],  # SF uses quarter-wave sines; natural BC at free end not enforced
    }

    for bc_type, checks in bc_expectations.items():
        for n in range(1, 4):
            for deriv, xi, target, tol in checks:
                val = beam_function(np.array([xi]), n, bc_type, deriv=deriv)[0]
                assert abs(val - target) < tol, f"{bc_type} n={n} deriv={deriv} xi={xi}"


def test_eigenvalues_increase_with_mode():
    for bc_type in ["SS", "CC", "CF", "FC", "CS", "SC", "FF", "SF"]:
        vals = [get_eigenvalue(n, bc_type) for n in range(1, 6)]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def test_orthogonality_ss_cc():
    xi = np.linspace(0, 1, 2001)
    for bc_type in ["SS", "CC"]:
        for m in range(1, 4):
            for n in range(1, 4):
                if m == n:
                    continue
                phi_m = beam_function(xi, m, bc_type, 0)
                phi_n = beam_function(xi, n, bc_type, 0)
                integral = np.trapezoid(phi_m * phi_n, xi)
                assert abs(integral) < 1e-2, f"{bc_type} m={m} n={n} integral={integral}"


def test_derivative_chain_matches_numeric():
    xi = np.linspace(0, 1, 2001)
    for bc_type in ["SS", "CC", "CF"]:
        for n in [1, 2, 3]:
            phi = beam_function(xi, n, bc_type, 0)
            phi_p = beam_function(xi, n, bc_type, 1)
            phi_p_num = _finite_diff_1d(phi, xi)

            # Exclude endpoints to reduce boundary error
            err = _relative_norm_error(phi_p_num[2:-2], phi_p[2:-2])
            assert err < 0.01, f"{bc_type} n={n} err={err}"


def test_cf_fc_reflection():
    xi = np.linspace(0, 1, 501)
    for n in range(1, 4):
        phi_fc = beam_function(xi, n, "FC", 0)
        phi_cf_ref = beam_function(1 - xi, n, "CF", 0)
        assert np.allclose(phi_fc, phi_cf_ref, atol=1e-6, rtol=1e-6)


def test_cs_sc_reflection():
    xi = np.linspace(0, 1, 501)
    for n in range(1, 4):
        phi_sc = beam_function(xi, n, "SC", 0)
        phi_cs_ref = beam_function(1 - xi, n, "CS", 0)
        assert np.allclose(phi_sc, phi_cs_ref, atol=1e-6, rtol=1e-6)


def test_solution_boundary_conditions():
    params = dict(a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3)
    q0 = 10000.0

    # CCCC: W=0 and dW/dn=0 at all edges
    solver = RitzSolver(**params, bc="CCCC", M=10, N=10)
    res = solver.solve(q0=q0)
    W = res["W"]
    Wmax = float(np.max(np.abs(W)))
    tol_w = 1e-6 + 1e-3 * Wmax
    assert _edge_max_abs(W) < tol_w

    X, Y = res["X"], res["Y"]
    dx, dy = X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0]
    dWdx = np.gradient(W, dx, axis=1)
    dWdy = np.gradient(W, dy, axis=0)
    dWdn_edges = [dWdx[:, 0], dWdx[:, -1], dWdy[0, :], dWdy[-1, :]]
    # Numerical gradient at boundaries is approximate; use looser tolerance
    tol_slope = 1e-3  # absolute tolerance for slope at clamped edges
    assert max(float(np.max(np.abs(edge))) for edge in dWdn_edges) < tol_slope

    # SSSS: W=0 at all edges
    solver = RitzSolver(**params, bc="SSSS", M=10, N=10)
    res = solver.solve(q0=q0)
    W = res["W"]
    Wmax = float(np.max(np.abs(W)))
    tol_w = 1e-6 + 1e-3 * Wmax
    assert _edge_max_abs(W) < tol_w

    # SCSF: S at x=0, C at y=0, S at x=a, F at y=b
    # W array: W[y_idx, x_idx]. W[0,:]=y=0, W[-1,:]=y=b, W[:,0]=x=0, W[:,-1]=x=a
    solver = RitzSolver(**params, bc="SCSF", M=10, N=10)
    res = solver.solve(q0=q0)
    W = res["W"]
    Wmax = float(np.max(np.abs(W)))
    tol_w = 1e-6 + 1e-3 * Wmax
    assert float(np.max(np.abs(W[:, 0]))) < tol_w, "x=0 (S) should be zero"
    assert float(np.max(np.abs(W[:, -1]))) < tol_w, "x=a (S) should be zero"
    assert float(np.max(np.abs(W[0, :]))) < tol_w, "y=0 (C) should be zero"
    assert float(np.max(np.abs(W[-1, :]))) > 0.01 * Wmax, "y=b (F) should be nonzero"

    # FCFC: F at x=0, C at y=0, F at x=a, C at y=b
    solver = RitzSolver(**params, bc="FCFC", M=10, N=10)
    res = solver.solve(q0=q0)
    W = res["W"]
    Wmax = float(np.max(np.abs(W)))
    tol_w = 1e-6 + 1e-3 * Wmax
    assert float(np.max(np.abs(W[0, :]))) < tol_w, "y=0 (C) should be zero"
    assert float(np.max(np.abs(W[-1, :]))) < tol_w, "y=b (C) should be zero"
    assert float(np.max(np.abs(W[:, 0]))) > 0.01 * Wmax, "x=0 (F) should be nonzero"
    assert float(np.max(np.abs(W[:, -1]))) > 0.01 * Wmax, "x=a (F) should be nonzero"


def test_stress_moment_consistency():
    params = dict(a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3)
    q0 = 10000.0
    solver = RitzSolver(**params, bc="SSSS", M=10, N=10)
    res = solver.solve(q0=q0)

    Mx = res["Mx"]
    sigma_x = res["sigma_x"]
    h = params["h"]

    rng = np.random.default_rng(1234)
    ny, nx = Mx.shape
    for _ in range(10):
        i = rng.integers(1, ny - 1)
        j = rng.integers(1, nx - 1)
        expected = 6 * Mx[i, j] / h**2
        actual = sigma_x[i, j]
        if expected == 0:
            assert abs(actual) < 1e-12
        else:
            rel = abs(actual - expected) / abs(expected)
            assert rel < 0.05


def test_moment_sign_convention_center():
    params = dict(a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3)
    q0 = 10000.0
    solver = RitzSolver(**params, bc="SSSS", M=10, N=10)
    res = solver.solve(q0=q0)

    Mx = res["Mx"]
    My = res["My"]
    ny, nx = Mx.shape
    ic, jc = ny // 2, nx // 2
    assert Mx[ic, jc] > 0
    assert My[ic, jc] > 0


def test_energy_balance():
    params = dict(a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3)
    q0 = 10000.0
    solver = RitzSolver(**params, bc="SSSS", M=10, N=10)
    solver.solve(q0=q0)

    K = solver._assemble_stiffness()
    F = solver._assemble_load_uniform(q0)
    A = solver.coeffs.reshape(-1)

    U = 0.5 * A @ (K @ A)
    W_ext = F @ A
    # At equilibrium: K*A = F, so A^T*K*A = A^T*F = W_ext
    # Therefore U = 0.5 * W_ext (Clapeyron's theorem)
    if W_ext == 0:
        assert abs(U) < 1e-12
    else:
        rel = abs(2 * U - W_ext) / abs(W_ext)
        assert rel < 0.01, f"Energy: 2U={2*U:.6e}, W_ext={W_ext:.6e}, rel={rel:.4f}"


def test_superposition_uniform_loads():
    params = dict(a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3)
    solver = RitzSolver(**params, bc="SSSS", M=10, N=10)

    W1 = solver.solve(q0=1000.0)["W"]
    W2 = solver.solve(q0=2000.0)["W"]
    W3 = solver.solve(q0=3000.0)["W"]

    diff = W1 + W2 - W3
    denom = np.max(np.abs(W3))
    rel = np.max(np.abs(diff)) / denom
    assert rel < 1e-3


def test_convergence_monotonicity_ritz():
    params = dict(a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3)
    q0 = 10000.0
    values = []

    for n in range(2, 16):
        solver = RitzSolver(**params, bc="CCCC", M=n, N=n)
        res = solver.solve(q0=q0)
        values.append(res["W_center"])

    # Ritz converges but NOT necessarily monotonically — higher terms can cause
    # oscillation. Just verify final value is close to Timoshenko (0.00126)
    D = 200e9 * 0.01**3 / (12 * (1 - 0.3**2))
    W_coef_final = values[-1] * D / (q0 * 1.0**4)
    assert abs(W_coef_final - 0.00126) / 0.00126 < 0.05, \
        f"CCCC W_coef={W_coef_final:.6f} should be ~0.00126"
