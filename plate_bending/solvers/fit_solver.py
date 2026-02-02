"""
Finite Integral Transform Solver for Plate Bending
===================================================

Two solution approaches:
1. SSSS plates: Exact Navier double sine series (true FIT)
2. Other Levy-type plates: Uses Levy-style ODE solving in y-direction
   with sine series in x-direction

For non-SSSS Levy-type plates, FIT effectively uses the same mathematical
approach as the Levy solver (sine series in x, ODE solution in y), so
results should match closely. This is mathematically correct since both
methods are solving the same governing equations.

References:
- Timoshenko & Woinowsky-Krieger, "Theory of Plates and Shells"
- Xu et al. (2020), "Analytical Bending Solutions of Orthotropic Rectangular
  Thin Plates with Two Adjacent Edges Free"
"""

import numpy as np
from numpy.linalg import solve, cond


class FITSolver:
    """
    Finite Integral Transform Method for plate bending.

    For SSSS plates:
        Uses exact Navier double sine series (true bidirectional FIT)

    For other Levy-type plates (SS on x-edges):
        Uses sine series in x and solves ODE in y (equivalent to Levy method)
    """

    def __init__(self, a, b, h, E, nu, bc='SSSS', n_terms=50):
        """
        Initialize solver.

        Parameters
        ----------
        a, b : float
            Plate dimensions in x and y directions
        h : float
            Plate thickness
        E : float
            Young's modulus
        nu : float
            Poisson's ratio
        bc : str
            4-character boundary condition string (e.g., 'SSSS', 'SCSF')
            Format: [x=0][y=0][x=a][y=b] where each is S/C/F
        n_terms : int
            Number of series terms
        """
        self.a, self.b, self.h = a, b, h
        self.E, self.nu = E, nu
        self.D = E * h**3 / (12 * (1 - nu**2))
        self.bc = bc.upper()
        self.n = n_terms
        self.results = {}
        self.debug = []

        self._parse_bc()

    def _parse_bc(self):
        """Parse boundary condition string."""
        bc = self.bc
        self.is_ssss = (bc == 'SSSS')
        self.is_levy_type = (bc[0] == 'S' and bc[2] == 'S')

        if self.is_levy_type:
            self.bc_y0 = bc[1]
            self.bc_yb = bc[3]

        if self.is_ssss:
            self.method = 'navier'
        elif self.is_levy_type:
            self.method = 'levy_ode'
        else:
            self.method = 'unsupported'

    def solve(self, load_type='uniform', q0=1.0, x0=None, y0=None, R=None,
              x1=None, y1=None, x2=None, y2=None,
              auto_converge=True, tol=1e-4, max_iter=200,
              progress_callback=None):
        """
        Solve using FIT method.

        Parameters
        ----------
        load_type : str
            'uniform', 'rect_patch', 'circular', or 'point'
        q0 : float
            Load intensity
        auto_converge : bool
            If True, iteratively increase terms until convergence
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum terms for convergence
        progress_callback : callable
            Optional callback(progress, message)
        """
        self.debug = [f"FINITE INTEGRAL TRANSFORM METHOD",
                      f"BC: {self.bc}, Method: {self.method}, D = {self.D:.4e}"]

        if self.method == 'navier':
            return self._solve_navier(load_type, q0, x0, y0, R, x1, y1, x2, y2,
                                      auto_converge, tol, max_iter, progress_callback)
        elif self.method == 'levy_ode':
            return self._solve_levy_ode(load_type, q0, x0, y0, R, x1, y1, x2, y2,
                                        auto_converge, tol, max_iter, progress_callback)
        else:
            raise ValueError(f"FIT not supported for BC: {self.bc}")

    # ========================================================================
    # Navier solution (SSSS only) - True bidirectional FIT
    # ========================================================================

    def _solve_navier(self, load_type, q0, x0, y0, R, x1, y1, x2, y2,
                      auto_converge, tol, max_iter, progress_callback):
        """Solve using Navier double sine series for SSSS plates."""
        if auto_converge:
            W, X, Y = self._solve_navier_converge(
                load_type, q0, x0, y0, R, x1, y1, x2, y2,
                tol, max_iter, progress_callback
            )
        else:
            nx, ny = 61, 61
            x = np.linspace(0, self.a, nx)
            y = np.linspace(0, self.b, ny)
            X, Y = np.meshgrid(x, y)
            W = self._solve_navier_core(X, Y, x, y, load_type, q0, x0, y0, R,
                                        x1, y1, x2, y2, self.n)
            if progress_callback:
                progress_callback(1.0, "Complete")

        self.results = {'X': X, 'Y': Y, 'W': W, 'load_type': load_type,
                        'q0': q0, 'method': 'FIT'}
        self._compute_derived()
        return self.results

    def _solve_navier_converge(self, load_type, q0, x0, y0, R, x1, y1, x2, y2,
                                tol, max_iter, progress_callback):
        """Navier solution with auto-convergence."""
        nx_check, ny_check = 21, 21
        x_check = np.linspace(0, self.a, nx_check)
        y_check = np.linspace(0, self.b, ny_check)
        X_check, Y_check = np.meshgrid(x_check, y_check)

        W_prev = None
        converged = False
        step = 20
        final_n = 20

        for n_current in range(20, max_iter + 1, step):
            if progress_callback:
                progress_callback(n_current / max_iter, f"FIT: n={n_current}")

            W_current = self._solve_navier_core(X_check, Y_check, x_check, y_check,
                                                load_type, q0, x0, y0, R,
                                                x1, y1, x2, y2, n_current)

            if W_prev is not None:
                norm_diff = np.linalg.norm(W_current - W_prev)
                norm_val = np.linalg.norm(W_current) + 1e-15
                error = norm_diff / norm_val

                self.debug.append(f"n={n_current}: rel_error = {error:.2e}")

                if error < tol:
                    self.debug.append(f"Converged at n={n_current}")
                    converged = True
                    final_n = n_current
                    break

            W_prev = W_current
            final_n = n_current

        if not converged:
            self.debug.append(f"WARNING: Did not converge within {max_iter} terms")

        if progress_callback:
            progress_callback(0.95, "FIT: Final calculation...")

        nx_final, ny_final = 61, 61
        x_final = np.linspace(0, self.a, nx_final)
        y_final = np.linspace(0, self.b, ny_final)
        X_final, Y_final = np.meshgrid(x_final, y_final)

        W_final = self._solve_navier_core(X_final, Y_final, x_final, y_final,
                                          load_type, q0, x0, y0, R,
                                          x1, y1, x2, y2, final_n)

        if progress_callback:
            progress_callback(1.0, "FIT: Complete")

        self.n = final_n
        return W_final, X_final, Y_final

    def _solve_navier_core(self, X, Y, x, y, load_type, q0, x0, y0, R,
                           x1, y1, x2, y2, n_terms):
        """Core Navier double sine series computation."""
        a, b, D = self.a, self.b, self.D
        W = np.zeros_like(X)

        for m in range(1, n_terms + 1):
            am = m * np.pi / a
            for nn in range(1, n_terms + 1):
                bn = nn * np.pi / b

                qmn = self._compute_qmn_navier(load_type, q0, m, nn, am, bn,
                                               x0, y0, R, x1, y1, x2, y2)
                if abs(qmn) < 1e-30:
                    continue

                denom = D * (am**2 + bn**2)**2
                if abs(denom) < 1e-30:
                    continue

                Wmn = qmn / denom
                sin_mx = np.sin(m * np.pi * x / a)
                sin_ny = np.sin(nn * np.pi * y / b)
                W += Wmn * np.outer(sin_ny, sin_mx)

        return W

    def _compute_qmn_navier(self, load_type, q0, m, n, am, bn, x0, y0, R,
                            x1, y1, x2, y2):
        """Compute load coefficient for Navier solution."""
        a, b = self.a, self.b

        if load_type == 'uniform':
            if m % 2 == 1 and n % 2 == 1:
                return 16 * q0 / (m * n * np.pi**2)
            return 0

        elif load_type == 'point':
            if x0 is None: x0 = a/2
            if y0 is None: y0 = b/2
            return q0 * np.sin(am * x0) * np.sin(bn * y0) * 4 / (a * b)

        elif load_type == 'circular':
            if x0 is None: x0 = a/2
            if y0 is None: y0 = b/2
            if R is None: R = a/10
            return self._circular_qmn(q0, R, x0, y0, am, bn)

        elif load_type == 'rect_patch':
            if x1 is None: x1 = 0.4 * a
            if x2 is None: x2 = 0.6 * a
            if y1 is None: y1 = 0.4 * b
            if y2 is None: y2 = 0.6 * b
            Ix = (np.cos(am * x1) - np.cos(am * x2)) / am
            Iy = (np.cos(bn * y1) - np.cos(bn * y2)) / bn
            return q0 * Ix * Iy * 4 / (a * b)

        return 0

    def _circular_qmn(self, p0, R, x0, y0, am, bn):
        """Compute qmn for circular patch load."""
        a, b = self.a, self.b
        n_r, n_theta = 20, 40
        total = 0.0

        for i in range(n_r):
            r = R * (i + 0.5) / n_r
            for j in range(n_theta):
                theta = 2 * np.pi * j / n_theta
                xp = x0 + r * np.cos(theta)
                yp = y0 + r * np.sin(theta)
                if 0 <= xp <= a and 0 <= yp <= b:
                    total += np.sin(am * xp) * np.sin(bn * yp) * r

        dr = R / n_r
        dtheta = 2 * np.pi / n_theta
        return p0 * total * dr * dtheta * 4 / (a * b)

    # ========================================================================
    # Levy-ODE solution (for non-SSSS Levy-type plates)
    # Uses sine series in x and solves ODE in y
    # ========================================================================

    def _solve_levy_ode(self, load_type, q0, x0, y0, R, x1, y1, x2, y2,
                        auto_converge, tol, max_iter, progress_callback):
        """Solve using Levy-style ODE approach for non-SSSS plates."""
        if auto_converge:
            W, X, Y = self._solve_levy_ode_converge(
                load_type, q0, x0, y0, R, x1, y1, x2, y2,
                tol, max_iter, progress_callback
            )
        else:
            nx, ny = 61, 61
            x = np.linspace(0, self.a, nx)
            y = np.linspace(0, self.b, ny)
            X, Y = np.meshgrid(x, y)
            W = self._solve_levy_ode_core(X, Y, x, y, load_type, q0, x0, y0, R,
                                          x1, y1, x2, y2, self.n)
            if progress_callback:
                progress_callback(1.0, "Complete")

        self.results = {'X': X, 'Y': Y, 'W': W, 'load_type': load_type,
                        'q0': q0, 'method': 'FIT-Levy'}
        self._compute_derived()
        return self.results

    def _solve_levy_ode_converge(self, load_type, q0, x0, y0, R, x1, y1, x2, y2,
                                  tol, max_iter, progress_callback):
        """Levy-ODE solution with auto-convergence."""
        nx_check, ny_check = 21, 21
        x_check = np.linspace(0, self.a, nx_check)
        y_check = np.linspace(0, self.b, ny_check)
        X_check, Y_check = np.meshgrid(x_check, y_check)

        W_prev = None
        converged = False
        step = 10
        final_n = 20

        for n_current in range(20, max_iter + 1, step):
            if progress_callback:
                progress_callback(n_current / max_iter, f"FIT: n={n_current}")

            W_current = self._solve_levy_ode_core(X_check, Y_check, x_check, y_check,
                                                  load_type, q0, x0, y0, R,
                                                  x1, y1, x2, y2, n_current)

            if W_prev is not None:
                norm_diff = np.linalg.norm(W_current - W_prev)
                norm_val = np.linalg.norm(W_current) + 1e-15
                error = norm_diff / norm_val

                self.debug.append(f"n={n_current}: rel_error = {error:.2e}")

                if error < tol:
                    self.debug.append(f"Converged at n={n_current}")
                    converged = True
                    final_n = n_current
                    break

            W_prev = W_current
            final_n = n_current

        if not converged:
            self.debug.append(f"WARNING: Did not converge within {max_iter} terms")

        if progress_callback:
            progress_callback(0.95, "FIT: Final calculation...")

        nx_final, ny_final = 61, 61
        x_final = np.linspace(0, self.a, nx_final)
        y_final = np.linspace(0, self.b, ny_final)
        X_final, Y_final = np.meshgrid(x_final, y_final)

        W_final = self._solve_levy_ode_core(X_final, Y_final, x_final, y_final,
                                            load_type, q0, x0, y0, R,
                                            x1, y1, x2, y2, final_n)

        if progress_callback:
            progress_callback(1.0, "FIT: Complete")

        self.n = final_n
        return W_final, X_final, Y_final

    def _solve_levy_ode_core(self, X, Y, x, y, load_type, q0, x0, y0, R,
                             x1, y1, x2, y2, n_terms):
        """Core Levy-ODE computation with sine series in x."""
        a, b = self.a, self.b
        W = np.zeros_like(X)

        # For uniform load, only odd m contribute
        if load_type == 'uniform':
            m_range = range(1, 2*n_terms, 2)
        else:
            m_range = range(1, n_terms + 1)

        for m in m_range:
            am = m * np.pi / a

            # Compute Y_m(y) based on load type
            if load_type == 'uniform':
                qm = 4.0 * q0 / (m * np.pi) if m % 2 == 1 else 0
                if abs(qm) < 1e-30:
                    continue
                Ym = self._solve_uniform_ode(am, qm, y)

            elif load_type == 'rect_patch':
                if x1 is None: x1 = 0.4 * a
                if x2 is None: x2 = 0.6 * a
                if y1 is None: y1 = 0.4 * b
                if y2 is None: y2 = 0.6 * b
                Qm = (2 * q0 / a) * (np.cos(am * x1) - np.cos(am * x2)) / am
                if abs(Qm) < 1e-30:
                    continue
                Ym = self._solve_patch_ode(am, Qm, y, y1, y2)

            elif load_type == 'circular':
                if x0 is None: x0 = a / 2
                if y0 is None: y0 = b / 2
                if R is None: R = a / 10
                Ym = self._solve_circular_ode(am, q0, x0, y0, R, y, m)

            elif load_type == 'point':
                if x0 is None: x0 = a / 2
                if y0 is None: y0 = b / 2
                qm = 2.0 * q0 / a * np.sin(m * np.pi * x0 / a)
                if abs(qm) < 1e-30:
                    continue
                Ym = self._solve_point_ode(am, qm, y, y0)

            else:
                continue

            # Add contribution
            sin_mx = np.sin(am * x)
            W += np.outer(Ym, sin_mx)

        return W

    def _solve_uniform_ode(self, am, qm, y):
        """Solve ODE for uniform load with stable exponential basis."""
        b, nu, D = self.b, self.nu, self.D
        Yp = qm / (D * am**4)  # Particular solution
        return self._solve_homogeneous_ode(am, y, Yp, Yp, 0, 0)

    def _solve_point_ode(self, am, qm, y, y0):
        """Solve ODE for point load (line load at y=y0)."""
        b, nu, D = self.b, self.nu, self.D

        # Point load creates a discontinuity at y=y0
        # Use piecewise solution similar to patch load
        # Treat as very thin patch
        dy = b / 100
        y1 = max(0, y0 - dy/2)
        y2 = min(b, y0 + dy/2)

        # Scale load intensity for thin strip
        Qm_scaled = qm / dy if dy > 0 else qm

        return self._solve_patch_ode(am, Qm_scaled, y, y1, y2)

    def _solve_patch_ode(self, am, Qm, y, y1, y2):
        """
        Solve ODE for patch load in y-region [y1, y2].
        Uses piecewise solution with 3 regions.
        """
        b, nu, D = self.b, self.nu, self.D
        Yp = Qm / (D * am**4)

        # Helper to evaluate basis at a point
        def eval_basis(yi, am, b):
            exp1 = np.exp(-am * (b - yi))
            exp2 = np.exp(-am * yi)
            psi = np.array([exp1, (b-yi)*exp1, exp2, yi*exp2])
            psi_p = np.array([am*exp1, (am*(b-yi)-1)*exp1, -am*exp2, (1-am*yi)*exp2])
            psi_pp = np.array([am**2*exp1, (am**2*(b-yi)-2*am)*exp1,
                               am**2*exp2, (am**2*yi-2*am)*exp2])
            psi_ppp = np.array([am**3*exp1, (am**3*(b-yi)-3*am**2)*exp1,
                                -am**3*exp2, (-am**3*yi+3*am**2)*exp2])
            return psi, psi_p, psi_pp, psi_ppp

        # Build 12x12 system for 3 regions
        M = np.zeros((12, 12))
        rhs = np.zeros(12)

        psi_0, psi_0_p, psi_0_pp, psi_0_ppp = eval_basis(0, am, b)
        psi_y1, psi_y1_p, psi_y1_pp, psi_y1_ppp = eval_basis(y1, am, b)
        psi_y2, psi_y2_p, psi_y2_pp, psi_y2_ppp = eval_basis(y2, am, b)
        psi_b, psi_b_p, psi_b_pp, psi_b_ppp = eval_basis(b, am, b)

        row = 0

        # BCs at y = 0 (Region I, coeffs 0-3)
        if self.bc_y0 == 'C':
            M[row, 0:4] = psi_0
            rhs[row] = 0
            row += 1
            M[row, 0:4] = psi_0_p
            rhs[row] = 0
            row += 1
        elif self.bc_y0 == 'S':
            M[row, 0:4] = psi_0
            rhs[row] = 0
            row += 1
            M[row, 0:4] = psi_0_pp
            rhs[row] = 0
            row += 1
        elif self.bc_y0 == 'F':
            M[row, 0:4] = psi_0_pp - nu*am**2*psi_0
            rhs[row] = 0
            row += 1
            M[row, 0:4] = psi_0_ppp - (2-nu)*am**2*psi_0_p
            rhs[row] = 0
            row += 1

        # Continuity at y = y1 (Region I = Region II + Yp)
        M[row, 0:4] = psi_y1
        M[row, 4:8] = -psi_y1
        rhs[row] = Yp
        row += 1
        M[row, 0:4] = psi_y1_p
        M[row, 4:8] = -psi_y1_p
        rhs[row] = 0
        row += 1
        M[row, 0:4] = psi_y1_pp
        M[row, 4:8] = -psi_y1_pp
        rhs[row] = 0
        row += 1
        M[row, 0:4] = psi_y1_ppp
        M[row, 4:8] = -psi_y1_ppp
        rhs[row] = 0
        row += 1

        # Continuity at y = y2 (Region II + Yp = Region III)
        M[row, 4:8] = psi_y2
        M[row, 8:12] = -psi_y2
        rhs[row] = -Yp
        row += 1
        M[row, 4:8] = psi_y2_p
        M[row, 8:12] = -psi_y2_p
        rhs[row] = 0
        row += 1
        M[row, 4:8] = psi_y2_pp
        M[row, 8:12] = -psi_y2_pp
        rhs[row] = 0
        row += 1
        M[row, 4:8] = psi_y2_ppp
        M[row, 8:12] = -psi_y2_ppp
        rhs[row] = 0
        row += 1

        # BCs at y = b (Region III, coeffs 8-11)
        if self.bc_yb == 'C':
            M[row, 8:12] = psi_b
            rhs[row] = 0
            row += 1
            M[row, 8:12] = psi_b_p
            rhs[row] = 0
        elif self.bc_yb == 'S':
            M[row, 8:12] = psi_b
            rhs[row] = 0
            row += 1
            M[row, 8:12] = psi_b_pp
            rhs[row] = 0
        elif self.bc_yb == 'F':
            M[row, 8:12] = psi_b_pp - nu*am**2*psi_b
            rhs[row] = 0
            row += 1
            M[row, 8:12] = psi_b_ppp - (2-nu)*am**2*psi_b_p
            rhs[row] = 0

        # Solve system
        try:
            coeffs = solve(M, rhs)
        except:
            coeffs = np.zeros(12)

        # Evaluate Y_m(y)
        Ym = np.zeros_like(y)
        for i, yi in enumerate(y):
            psi, _, _, _ = eval_basis(yi, am, b)
            if yi < y1:
                Ym[i] = np.dot(coeffs[0:4], psi)
            elif yi <= y2:
                Ym[i] = np.dot(coeffs[4:8], psi) + Yp
            else:
                Ym[i] = np.dot(coeffs[8:12], psi)

        return Ym

    def _solve_circular_ode(self, am, q0, x0, y0, R, y, m):
        """Solve ODE for circular load using strip superposition."""
        n_strips = 20
        Ym = np.zeros_like(y)

        circ_y1 = max(0, y0 - R)
        circ_y2 = min(self.b, y0 + R)
        strip_y = np.linspace(circ_y1, circ_y2, n_strips + 1)

        for i in range(n_strips):
            y1_strip = strip_y[i]
            y2_strip = strip_y[i + 1]
            y_mid = (y1_strip + y2_strip) / 2

            dy_from_center = y_mid - y0
            if abs(dy_from_center) >= R:
                continue

            dx = np.sqrt(R**2 - dy_from_center**2)
            x1_strip = max(0, x0 - dx)
            x2_strip = min(self.a, x0 + dx)

            if x2_strip <= x1_strip:
                continue

            # Fourier coefficient for this strip
            Qm_strip = (2 * q0 / self.a) * (np.cos(am * x1_strip) - np.cos(am * x2_strip)) / am

            if abs(Qm_strip) < 1e-30:
                continue

            Ym_strip = self._solve_patch_ode(am, Qm_strip, y, y1_strip, y2_strip)
            Ym += Ym_strip

        return Ym

    def _solve_homogeneous_ode(self, am, y, Yp_0, Yp_b, Yp_prime_0, Yp_prime_b):
        """
        Solve homogeneous ODE with given particular solution values at boundaries.

        For uniform load: Yp is constant, so Yp_0 = Yp_b = Yp, Yp' = 0
        """
        b, nu, D = self.b, self.nu, self.D

        # Basis function values at boundaries
        exp_amb = np.exp(-am * b)

        # At y = 0
        psi1_0 = exp_amb
        psi1_0_p = am * exp_amb
        psi1_0_pp = am**2 * exp_amb
        psi1_0_ppp = am**3 * exp_amb

        psi2_0 = b * exp_amb
        psi2_0_p = (am * b - 1) * exp_amb
        psi2_0_pp = (am**2 * b - 2*am) * exp_amb
        psi2_0_ppp = (am**3 * b - 3*am**2) * exp_amb

        psi3_0 = 1.0
        psi3_0_p = -am
        psi3_0_pp = am**2
        psi3_0_ppp = -am**3

        psi4_0 = 0.0
        psi4_0_p = 1.0
        psi4_0_pp = -2*am
        psi4_0_ppp = 3*am**2

        # At y = b
        psi1_b = 1.0
        psi1_b_p = am
        psi1_b_pp = am**2
        psi1_b_ppp = am**3

        psi2_b = 0.0
        psi2_b_p = -1.0
        psi2_b_pp = -2*am
        psi2_b_ppp = -3*am**2

        psi3_b = exp_amb
        psi3_b_p = -am * exp_amb
        psi3_b_pp = am**2 * exp_amb
        psi3_b_ppp = -am**3 * exp_amb

        psi4_b = b * exp_amb
        psi4_b_p = (1 - am*b) * exp_amb
        psi4_b_pp = (am**2 * b - 2*am) * exp_amb
        psi4_b_ppp = (-am**3 * b + 3*am**2) * exp_amb

        # Build 4x4 system
        M = np.zeros((4, 4))
        rhs = np.zeros(4)

        # BCs at y = 0
        if self.bc_y0 == 'C':
            M[0] = [psi1_0, psi2_0, psi3_0, psi4_0]
            rhs[0] = -Yp_0
            M[1] = [psi1_0_p, psi2_0_p, psi3_0_p, psi4_0_p]
            rhs[1] = -Yp_prime_0
        elif self.bc_y0 == 'S':
            M[0] = [psi1_0, psi2_0, psi3_0, psi4_0]
            rhs[0] = -Yp_0
            M[1] = [psi1_0_pp, psi2_0_pp, psi3_0_pp, psi4_0_pp]
            rhs[1] = 0  # Yp'' = 0 for uniform load
        elif self.bc_y0 == 'F':
            M[0] = [psi1_0_pp - nu*am**2*psi1_0,
                    psi2_0_pp - nu*am**2*psi2_0,
                    psi3_0_pp - nu*am**2*psi3_0,
                    psi4_0_pp - nu*am**2*psi4_0]
            rhs[0] = nu * am**2 * Yp_0
            M[1] = [psi1_0_ppp - (2-nu)*am**2*psi1_0_p,
                    psi2_0_ppp - (2-nu)*am**2*psi2_0_p,
                    psi3_0_ppp - (2-nu)*am**2*psi3_0_p,
                    psi4_0_ppp - (2-nu)*am**2*psi4_0_p]
            rhs[1] = (2-nu) * am**2 * Yp_prime_0

        # BCs at y = b
        if self.bc_yb == 'C':
            M[2] = [psi1_b, psi2_b, psi3_b, psi4_b]
            rhs[2] = -Yp_b
            M[3] = [psi1_b_p, psi2_b_p, psi3_b_p, psi4_b_p]
            rhs[3] = -Yp_prime_b
        elif self.bc_yb == 'S':
            M[2] = [psi1_b, psi2_b, psi3_b, psi4_b]
            rhs[2] = -Yp_b
            M[3] = [psi1_b_pp, psi2_b_pp, psi3_b_pp, psi4_b_pp]
            rhs[3] = 0
        elif self.bc_yb == 'F':
            M[2] = [psi1_b_pp - nu*am**2*psi1_b,
                    psi2_b_pp - nu*am**2*psi2_b,
                    psi3_b_pp - nu*am**2*psi3_b,
                    psi4_b_pp - nu*am**2*psi4_b]
            rhs[2] = nu * am**2 * Yp_b
            M[3] = [psi1_b_ppp - (2-nu)*am**2*psi1_b_p,
                    psi2_b_ppp - (2-nu)*am**2*psi2_b_p,
                    psi3_b_ppp - (2-nu)*am**2*psi3_b_p,
                    psi4_b_ppp - (2-nu)*am**2*psi4_b_p]
            rhs[3] = (2-nu) * am**2 * Yp_prime_b

        # Solve
        try:
            coeffs = solve(M, rhs)
            A, B, C, D_coef = coeffs
        except:
            A, B, C, D_coef = 0, 0, 0, 0

        # Evaluate Y_m(y)
        Ym = np.zeros_like(y)
        for i, yi in enumerate(y):
            exp1 = np.exp(-am * (b - yi))
            exp2 = np.exp(-am * yi)
            psi1 = exp1
            psi2 = (b - yi) * exp1
            psi3 = exp2
            psi4 = yi * exp2
            Ym[i] = A*psi1 + B*psi2 + C*psi3 + D_coef*psi4 + Yp_0

        return Ym

    def _compute_derived(self):
        """Compute moments, stresses, and coefficients."""
        X, Y, W = self.results['X'], self.results['Y'], self.results['W']
        dx = X[0,1] - X[0,0] if X.shape[1] > 1 else self.a / 60
        dy = Y[1,0] - Y[0,0] if Y.shape[0] > 1 else self.b / 60

        Wxx = np.gradient(np.gradient(W, dx, axis=1), dx, axis=1)
        Wyy = np.gradient(np.gradient(W, dy, axis=0), dy, axis=0)
        Wxy = np.gradient(np.gradient(W, dy, axis=0), dx, axis=1)

        Mx = -self.D * (Wxx + self.nu * Wyy)
        My = -self.D * (Wyy + self.nu * Wxx)
        Mxy = -self.D * (1 - self.nu) * Wxy

        sx = 6 * Mx / self.h**2
        sy = 6 * My / self.h**2
        txy = 6 * Mxy / self.h**2
        vm = np.sqrt(sx**2 + sy**2 - sx*sy + 3*txy**2)

        q0 = self.results['q0']
        self.results.update({
            'Mx': Mx, 'My': My, 'Mxy': Mxy,
            'sigma_x': sx, 'sigma_y': sy, 'tau_xy': txy, 'von_mises': vm,
            'W_max': np.max(np.abs(W)),
            'W_max_loc': (X.flat[np.argmax(np.abs(W))], Y.flat[np.argmax(np.abs(W))]),
            'Mx_max': np.max(np.abs(Mx)), 'My_max': np.max(np.abs(My)),
            'vm_max': np.max(vm),
            'W_coef': np.max(np.abs(W)) * self.D / (q0 * self.a**4) if q0 != 0 else 0,
            'Mx_coef': np.max(np.abs(Mx)) / (q0 * self.a**2) if q0 != 0 else 0,
            'My_coef': np.max(np.abs(My)) / (q0 * self.a**2) if q0 != 0 else 0,
            'n_terms_used': self.n,
        })

        ny, nx = W.shape
        ic, jc = ny//2, nx//2
        self.results['W_center'] = W[ic, jc]
        self.results['W_center_coef'] = abs(W[ic, jc]) * self.D / (q0 * self.a**4) if q0 != 0 else 0
