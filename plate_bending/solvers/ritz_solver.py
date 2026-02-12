"""
Rayleigh-Ritz Solver for Plate Bending
======================================

Approximates plate deflection as a double series:
    w(x,y) = sum_m sum_n A_mn * X_m(x) * Y_n(y)

where X_m and Y_n are beam characteristic functions satisfying the
boundary conditions on each edge.

This provides independent validation of the Levy method.
"""

import numpy as np
from scipy.integrate import dblquad, fixed_quad
from .beam_functions import beam_function, get_eigenvalue, compute_beam_integrals, compute_load_integral


class RitzSolver:
    """
    Rayleigh-Ritz solver for rectangular plate bending.

    Supports:
    - SSSS, SCSF, SSSF, SCSC and other Levy-type boundary conditions
    - Uniform, rectangular patch, and circular patch loads
    """

    def __init__(self, a, b, h, E, nu, bc='SCSF', M=10, N=10):
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
            4-character boundary condition string:
            Position 0: x=0, Position 1: y=0, Position 2: x=a, Position 3: y=b
            Characters: S=Simply supported, C=Clamped, F=Free
        M, N : int
            Number of terms in x and y directions
        """
        self.a, self.b, self.h = a, b, h
        self.E, self.nu = E, nu
        self.D = E * h**3 / (12 * (1 - nu**2))
        self.bc = bc.upper()
        self.M, self.N = M, N
        self.ndof = M * N
        self.results = {}
        self.debug = []
        self._K = None
        self._F = None

        # Parse boundary conditions
        self.bc_x0 = self.bc[0]  # x = 0
        self.bc_y0 = self.bc[1]  # y = 0
        self.bc_xa = self.bc[2]  # x = a
        self.bc_yb = self.bc[3]  # y = b

        # Determine beam BC types for x and y directions
        self.bc_x = self.bc_x0 + self.bc_xa  # e.g., 'SS' for simply supported
        self.bc_y = self.bc_y0 + self.bc_yb  # e.g., 'CF' for clamped-free

        # Precompute beam function integrals
        self._compute_integrals()

    def _compute_integrals(self):
        """Precompute beam function integrals for stiffness matrix."""
        self.debug.append(f"Computing beam integrals: bc_x={self.bc_x}, bc_y={self.bc_y}")

        # X-direction integrals
        self.Ix0, self.Ix1, self.Ix2 = compute_beam_integrals(self.M, self.bc_x, self.a)

        # Y-direction integrals
        self.Iy0, self.Iy1, self.Iy2 = compute_beam_integrals(self.N, self.bc_y, self.b)

        # Mixed integrals for Poisson coupling: integral[phi''*psi] and integral[phi*psi'']
        self._compute_mixed_integrals()

    def _compute_mixed_integrals(self):
        """Compute mixed integrals for Poisson coupling."""
        from scipy.integrate import fixed_quad

        n_quad = 50

        # Ix20: integral[X_m'' * X_p] for x-direction
        self.Ix20 = np.zeros((self.M, self.M))
        for m in range(1, self.M + 1):
            for p in range(1, self.M + 1):
                def f(xi):
                    return beam_function(xi, m, self.bc_x, 2) * beam_function(xi, p, self.bc_x, 0)
                val, _ = fixed_quad(f, 0, 1, n=n_quad)
                self.Ix20[m-1, p-1] = val / self.a  # Scale: (1/L^2) * L = 1/L

        # Iy20: integral[Y_n'' * Y_q] for y-direction
        self.Iy20 = np.zeros((self.N, self.N))
        for n in range(1, self.N + 1):
            for q in range(1, self.N + 1):
                def f(eta):
                    return beam_function(eta, n, self.bc_y, 2) * beam_function(eta, q, self.bc_y, 0)
                val, _ = fixed_quad(f, 0, 1, n=n_quad)
                self.Iy20[n-1, q-1] = val / self.b

    def _assemble_stiffness(self):
        """
        Assemble global stiffness matrix K.

        From strain energy U = (D/2) * integral[(w_xx + w_yy)^2 - 2(1-nu)(w_xx*w_yy - w_xy^2)] dA
        Expanding:
        U = (D/2) * integral[w_xx^2 + w_yy^2 + 2*nu*w_xx*w_yy + 2(1-nu)*w_xy^2] dA

        K_mnpq = D * [Ix2[m,p]*Iy0[n,q] + Ix0[m,p]*Iy2[n,q]
                      + nu*(Ix20[m,p]*Iy20[q,n] + Ix20[p,m]*Iy20[n,q])
                      + 2(1-nu)*Ix1[m,p]*Iy1[n,q]]
        """
        K = np.zeros((self.ndof, self.ndof))
        D, nu = self.D, self.nu

        for m in range(1, self.M + 1):
            for n in range(1, self.N + 1):
                i = (m - 1) * self.N + (n - 1)

                for p in range(1, self.M + 1):
                    for q in range(1, self.N + 1):
                        j = (p - 1) * self.N + (q - 1)

                        # Skip lower triangle (symmetric)
                        if j < i:
                            continue

                        # Term 1: w_xx^2 contribution = X_m''*X_p'' * Y_n*Y_q
                        term1 = self.Ix2[m-1, p-1] * self.Iy0[n-1, q-1]

                        # Term 2: w_yy^2 contribution = X_m*X_p * Y_n''*Y_q''
                        term2 = self.Ix0[m-1, p-1] * self.Iy2[n-1, q-1]

                        # Term 3: Poisson coupling 2*nu*w_xx*w_yy
                        # = 2*nu * (X_m''*Y_n) * (X_p*Y_q'')
                        # = 2*nu * Ix20[m,p] * Iy20[q,n]  (note: Iy20[q,n] = integral[Y_q''*Y_n])
                        # By symmetry, also includes the (m,p) <-> (p,m) and (n,q) <-> (q,n) terms
                        term3 = nu * (self.Ix20[m-1, p-1] * self.Iy20[q-1, n-1] +
                                      self.Ix20[p-1, m-1] * self.Iy20[n-1, q-1])

                        # Term 4: twist contribution 2(1-nu)*w_xy^2 = 2(1-nu)*X_m'*X_p' * Y_n'*Y_q'
                        term4 = 2 * (1 - nu) * self.Ix1[m-1, p-1] * self.Iy1[n-1, q-1]

                        K[i, j] = D * (term1 + term2 + term3 + term4)
                        K[j, i] = K[i, j]  # Symmetric

        return K

    def _assemble_load_uniform(self, q0):
        """Load vector for uniform pressure q0."""
        F = np.zeros(self.ndof)

        for m in range(1, self.M + 1):
            for n in range(1, self.N + 1):
                i = (m - 1) * self.N + (n - 1)

                # F_mn = q0 * integral[X_m(x) * Y_n(y)] dA
                Ix = compute_load_integral(m, self.bc_x, 0, 1, self.a)
                Iy = compute_load_integral(n, self.bc_y, 0, 1, self.b)

                F[i] = q0 * Ix * Iy

        return F

    def _assemble_load_rect_patch(self, q0, x1, y1, x2, y2):
        """Load vector for rectangular patch load."""
        F = np.zeros(self.ndof)

        # Normalized coordinates
        xi1, xi2 = x1 / self.a, x2 / self.a
        eta1, eta2 = y1 / self.b, y2 / self.b

        for m in range(1, self.M + 1):
            for n in range(1, self.N + 1):
                i = (m - 1) * self.N + (n - 1)

                # F_mn = q0 * integral_patch[X_m(x) * Y_n(y)] dA
                Ix = compute_load_integral(m, self.bc_x, xi1, xi2, self.a)
                Iy = compute_load_integral(n, self.bc_y, eta1, eta2, self.b)

                F[i] = q0 * Ix * Iy

        return F

    def _assemble_load_circular(self, q0, xc, yc, R):
        """Load vector for circular patch load."""
        F = np.zeros(self.ndof)

        n_r = 20  # Radial points
        n_theta = 40  # Angular points

        for m in range(1, self.M + 1):
            for n in range(1, self.N + 1):
                i = (m - 1) * self.N + (n - 1)

                # Numerical integration over circle
                total = 0.0
                for ir in range(n_r):
                    r = R * (ir + 0.5) / n_r
                    for it in range(n_theta):
                        theta = 2 * np.pi * it / n_theta
                        x = xc + r * np.cos(theta)
                        y = yc + r * np.sin(theta)

                        # Check bounds
                        if x < 0 or x > self.a or y < 0 or y > self.b:
                            continue

                        xi = x / self.a
                        eta = y / self.b

                        phi_m = beam_function(xi, m, self.bc_x, 0)
                        psi_n = beam_function(eta, n, self.bc_y, 0)

                        total += r * phi_m * psi_n

                dr = R / n_r
                dtheta = 2 * np.pi / n_theta
                F[i] = q0 * total * dr * dtheta

        return F

    def solve(self, load_type='uniform', q0=1.0, x0=None, y0=None,
              x1=None, y1=None, x2=None, y2=None, R=None):
        """
        Solve for deflection.

        Parameters
        ----------
        load_type : str
            'uniform', 'rect_patch', or 'circular'
        q0 : float
            Load magnitude
        x0, y0 : float
            Center of circular load
        x1, y1, x2, y2 : float
            Bounds of rectangular patch
        R : float
            Radius of circular patch

        Returns
        -------
        dict
            Results including W, moments, stresses, coefficients
        """
        self.debug = [f"RITZ METHOD",
                      f"BC: {self.bc}, D = {self.D:.4e}",
                      f"Terms: M={self.M}, N={self.N}"]

        # Assemble stiffness
        K = self._assemble_stiffness()
        self.debug.append(f"Stiffness matrix assembled, cond = {np.linalg.cond(K):.2e}")

        # Assemble load vector
        if load_type == 'uniform':
            F = self._assemble_load_uniform(q0)
        elif load_type == 'rect_patch':
            if x1 is None: x1 = 0.4 * self.a
            if x2 is None: x2 = 0.6 * self.a
            if y1 is None: y1 = 0.4 * self.b
            if y2 is None: y2 = 0.6 * self.b
            F = self._assemble_load_rect_patch(q0, x1, y1, x2, y2)
        elif load_type == 'circular':
            if R is None: R = min(self.a, self.b) / 10
            if x0 is None: x0 = self.a / 2
            if y0 is None: y0 = self.b / 2
            F = self._assemble_load_circular(q0, x0, y0, R)
        else:
            raise ValueError(f"Unknown load type: {load_type}")

        # Store K and F for diagnostics / appendix display
        self._K = K.copy()
        self._F = F.copy()

        # Solve K*A = F
        try:
            A = np.linalg.solve(K, F)
        except np.linalg.LinAlgError:
            # Fall back to least squares
            A, _, _, _ = np.linalg.lstsq(K, F, rcond=None)

        # Store coefficients
        self.coeffs = A.reshape((self.M, self.N))

        # Compute deflection field
        nx, ny = 61, 61
        x = np.linspace(0, self.a, nx)
        y = np.linspace(0, self.b, ny)
        X, Y = np.meshgrid(x, y)
        W = self._compute_deflection_field(X, Y)

        self.results = {'X': X, 'Y': Y, 'W': W, 'load_type': load_type,
                        'q0': q0, 'method': 'Ritz', 'n_terms': (self.M, self.N)}
        self._compute_derived()
        return self.results

    def _compute_deflection_field(self, X, Y):
        """Evaluate w(x,y) on grid."""
        W = np.zeros_like(X)

        # Precompute beam functions on grid
        xi = X / self.a
        eta = Y / self.b

        for m in range(1, self.M + 1):
            phi_m = beam_function(xi, m, self.bc_x, 0)

            for n in range(1, self.N + 1):
                psi_n = beam_function(eta, n, self.bc_y, 0)
                A_mn = self.coeffs[m-1, n-1]

                W += A_mn * phi_m * psi_n

        return W

    def _compute_derived(self):
        """Compute moments, stresses, validation coefficients."""
        X, Y, W = self.results['X'], self.results['Y'], self.results['W']
        dx, dy = X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0]

        # Numerical derivatives
        Wxx = np.gradient(np.gradient(W, dx, axis=1), dx, axis=1)
        Wyy = np.gradient(np.gradient(W, dy, axis=0), dy, axis=0)
        Wxy = np.gradient(np.gradient(W, dy, axis=0), dx, axis=1)

        # Moments
        Mx = -self.D * (Wxx + self.nu * Wyy)
        My = -self.D * (Wyy + self.nu * Wxx)
        Mxy = -self.D * (1 - self.nu) * Wxy

        # Bending stresses
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
            'W_coef': np.max(np.abs(W)) * self.D / (q0 * self.a**4),
            'Mx_coef': np.max(np.abs(Mx)) / (q0 * self.a**2),
            'My_coef': np.max(np.abs(My)) / (q0 * self.a**2),
        })

        # Center values
        ny, nx = W.shape
        ic, jc = ny // 2, nx // 2
        self.results['W_center'] = W[ic, jc]
        self.results['W_center_coef'] = abs(W[ic, jc]) * self.D / (q0 * self.a**4)

    def convergence_study(self, load_type='uniform', q0=1.0, n_list=None):
        """
        Perform convergence study.

        Parameters
        ----------
        load_type : str
            Load type
        q0 : float
            Load magnitude
        n_list : list
            List of term counts to test

        Returns
        -------
        list of tuple
            (n, W_max, W_coef) for each n
        """
        if n_list is None:
            n_list = [3, 5, 8, 10, 12, 15]

        results = []
        for n in n_list:
            self.M = n
            self.N = n
            self.ndof = n * n
            self._compute_integrals()

            res = self.solve(load_type, q0)
            results.append((n, res['W_max'], res['W_coef']))
            self.debug.append(f"n={n}: W_coef={res['W_coef']:.6f}")

        return results
