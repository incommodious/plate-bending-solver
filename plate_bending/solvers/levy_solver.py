"""
Numerically Stable Levy Solution for Plate Bending
===================================================

Uses exponentially-scaled basis functions to avoid overflow.

Instead of {cosh(ay), sinh(ay), y*cosh(ay), y*sinh(ay)} which overflow,
we use:
    {e^(-a(b-y)), (b-y)*e^(-a(b-y)), e^(-ay), y*e^(-ay)}

These are all O(1) throughout the domain [0, b].

Sign corrections applied for free edge boundary conditions (2026-01-31).
"""

import numpy as np
from numpy.linalg import solve, cond
from scipy.special import jv


class StableLevySolver:
    """
    Levy solution using exponentially-scaled basis functions.

    The general solution becomes:
        Y(y) = A*e^(-a(b-y)) + B*(b-y)*e^(-a(b-y)) + C*e^(-ay) + D*y*e^(-ay) + Yp

    Supports boundary conditions:
        S = Simply supported (w=0, M=0)
        C = Clamped (w=0, dw/dy=0)
        F = Free (M=0, V=0)
    """

    def __init__(self, a, b, h, E, nu, bc_y0='C', bc_yb='F', n_terms=100):
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
        bc_y0 : str
            Boundary condition at y=0 ('S', 'C', or 'F')
        bc_yb : str
            Boundary condition at y=b ('S', 'C', or 'F')
        n_terms : int
            Number of Fourier terms
        """
        self.a, self.b, self.h = a, b, h
        self.E, self.nu = E, nu
        self.D = E * h**3 / (12 * (1 - nu**2))
        self.bc_y0 = bc_y0.upper()
        self.bc_yb = bc_yb.upper()
        self.bc = f"S{bc_y0}S{bc_yb}"
        self.n_terms = n_terms
        self.results = {}
        self.debug = []

    def alpha(self, m):
        """Wave number for mode m."""
        return m * np.pi / self.a

    def solve(self, load_type='uniform', q0=1.0, x0=None, y0=None, R=None,
              x1=None, y1=None, x2=None, y2=None):
        """
        Solve using stable Levy formulation.

        Parameters
        ----------
        load_type : str
            'uniform', 'point', 'circular', or 'rect_patch'
        q0 : float
            Load magnitude (pressure for uniform/patch, force for point)
        x0, y0 : float
            Load center position (for point, circular, rect_patch)
        R : float
            Radius for circular patch
        x1, y1, x2, y2 : float
            Bounds for rectangular patch

        Returns
        -------
        dict
            Results including W, Mx, My, stresses, coefficients
        """
        self.debug = [f"LEVY METHOD (Stable Formulation)",
                      f"BC: {self.bc}, D = {self.D:.4e}"]

        nx, ny = 61, 61
        x = np.linspace(0, self.a, nx)
        y = np.linspace(0, self.b, ny)
        X, Y = np.meshgrid(x, y)
        W = np.zeros_like(X)

        # Convergence tracking
        W_old = np.zeros_like(X)

        # For uniform load, only odd m contribute
        if load_type == 'uniform':
            m_range = range(1, 2*self.n_terms, 2)
        else:
            m_range = range(1, self.n_terms + 1)

        for m in m_range:
            am = self.alpha(m)

            # Load coefficient
            if load_type == 'uniform':
                qm = 4.0 * q0 / (m * np.pi) if m % 2 == 1 else 0
            elif load_type == 'point':
                if x0 is None: x0 = self.a / 2
                qm = 2.0 * q0 / self.a * np.sin(m * np.pi * x0 / self.a)
            elif load_type == 'circular':
                if x0 is None: x0 = self.a / 2
                if y0 is None: y0 = self.b / 2
                if R is None: R = self.a / 10
                # Will be handled specially below with numerical integration
                qm = 1.0  # Placeholder - actual contribution computed in _solve_circular_load
            elif load_type == 'rect_patch':
                if x1 is None: x1 = 0.4 * self.a
                if x2 is None: x2 = 0.6 * self.a
                qm = self._rect_patch_x_coef(q0, x1, x2, m)
            else:
                qm = 4.0 * q0 / (m * np.pi)

            if abs(qm) < 1e-30:
                continue

            # Solve for Ym(y)
            if load_type == 'rect_patch' and y1 is not None and y2 is not None:
                # Piecewise solution for patch load
                Ym, cond_num = self._solve_patch_load_ode(am, qm, y, y1, y2)
            elif load_type == 'circular' and y0 is not None and R is not None:
                # Use numerical integration for circular load
                Ym, cond_num = self._solve_circular_load(am, q0, x0, y0, R, y, m)
            else:
                Ym, cond_num = self._solve_stable_ode(am, qm, y, m)

            if cond_num > 1e10:
                self.debug.append(f"  m={m}: cond={cond_num:.2e} (high but stable)")

            # Add contribution
            sin_mx = np.sin(am * x)
            W += np.outer(Ym, sin_mx)

            # Check convergence
            rel_change = np.max(np.abs(W - W_old)) / (np.max(np.abs(W)) + 1e-30)
            if m > 1 and rel_change < 1e-8:
                self.debug.append(f"Converged at m={m}, rel_change={rel_change:.2e}")
                break
            W_old = W.copy()

        self.results = {'X': X, 'Y': Y, 'W': W, 'load_type': load_type,
                        'q0': q0, 'method': 'Levy'}
        self._compute_derived()
        return self.results

    def _rect_patch_x_coef(self, q0, x1, x2, m):
        """Fourier coefficient for rectangular patch load (x-direction only)."""
        am = m * np.pi / self.a
        # Integral of sin(am*x) from x1 to x2
        return (2 * q0 / self.a) * (np.cos(am * x1) - np.cos(am * x2)) / am

    def _solve_stable_ode(self, am, qm, y, m):
        """
        Solve ODE using stable exponential basis (uniform load case).

        Basis functions:
            psi_1(y) = e^(-a(b-y))      - decays toward y=0, equals 1 at y=b
            psi_2(y) = (b-y)*e^(-a(b-y)) - decays toward y=0, equals 0 at y=b
            psi_3(y) = e^(-ay)          - decays toward y=b, equals 1 at y=0
            psi_4(y) = y*e^(-ay)        - decays toward y=b, equals 0 at y=0
        """
        b = self.b
        nu = self.nu
        D = self.D

        # Particular solution for uniform load
        Yp = qm / (D * am**4)

        # Evaluate basis functions and derivatives at boundaries
        # At y = 0:
        exp_amb = np.exp(-am * b)  # This is small for large am*b

        # psi_1(0) = e^(-ab), psi_1'(0) = a*e^(-ab), psi_1''(0) = a^2*e^(-ab), psi_1'''(0) = a^3*e^(-ab)
        psi1_0 = exp_amb
        psi1_0_p = am * exp_amb
        psi1_0_pp = am**2 * exp_amb
        psi1_0_ppp = am**3 * exp_amb

        # psi_2(0) = b*e^(-ab), psi_2'(0) = -e^(-ab) + ab*e^(-ab) = (ab-1)*e^(-ab)
        psi2_0 = b * exp_amb
        psi2_0_p = (am * b - 1) * exp_amb
        psi2_0_pp = (am**2 * b - 2*am) * exp_amb
        psi2_0_ppp = (am**3 * b - 3*am**2) * exp_amb

        # psi_3(0) = 1, psi_3'(0) = -a, psi_3''(0) = a^2, psi_3'''(0) = -a^3
        psi3_0 = 1.0
        psi3_0_p = -am
        psi3_0_pp = am**2
        psi3_0_ppp = -am**3

        # psi_4(0) = 0, psi_4'(0) = 1, psi_4''(0) = -2a, psi_4'''(0) = 3a^2
        psi4_0 = 0.0
        psi4_0_p = 1.0
        psi4_0_pp = -2*am
        psi4_0_ppp = 3*am**2

        # At y = b:
        exp_amb_b = np.exp(-am * b)

        # psi_1(b) = 1, psi_1'(b) = a, psi_1''(b) = a^2, psi_1'''(b) = a^3
        psi1_b = 1.0
        psi1_b_p = am
        psi1_b_pp = am**2
        psi1_b_ppp = am**3

        # psi_2(b) = 0, psi_2'(b) = -1, psi_2''(b) = -2a, psi_2'''(b) = -3a^2
        # Derivation: psi2 = (b-y)*e^(-am*(b-y)), let u=b-y
        # psi2'' = d^2/dy^2[(am*u - 1)*e^(-am*u)] = (am^2*u - 2*am)*e^(-am*u)
        # At y=b: u=0, so psi2''(b) = -2*am
        psi2_b = 0.0
        psi2_b_p = -1.0
        psi2_b_pp = -2*am  # FIXED: was incorrectly +2*am
        psi2_b_ppp = -3*am**2

        # psi_3(b) = e^(-ab), psi_3'(b) = -a*e^(-ab), psi_3''(b) = a^2*e^(-ab), psi_3'''(b) = -a^3*e^(-ab)
        psi3_b = exp_amb_b
        psi3_b_p = -am * exp_amb_b
        psi3_b_pp = am**2 * exp_amb_b
        psi3_b_ppp = -am**3 * exp_amb_b

        # psi_4(b) = b*e^(-ab), psi_4'(b) = (1 - ab)*e^(-ab)
        psi4_b = b * exp_amb_b
        psi4_b_p = (1 - am*b) * exp_amb_b
        psi4_b_pp = (am**2 * b - 2*am) * exp_amb_b
        psi4_b_ppp = (-am**3 * b + 3*am**2) * exp_amb_b

        # Build 4x4 system M*[A,B,C,D]^T = rhs
        M = np.zeros((4, 4))
        rhs = np.zeros(4)

        # === BCs at y = 0 ===
        if self.bc_y0 == 'C':  # Clamped: Y(0) = 0, Y'(0) = 0
            M[0] = [psi1_0, psi2_0, psi3_0, psi4_0]
            rhs[0] = -Yp
            M[1] = [psi1_0_p, psi2_0_p, psi3_0_p, psi4_0_p]
            rhs[1] = 0

        elif self.bc_y0 == 'S':  # Simply supported: Y(0) = 0, Y''(0) = 0
            M[0] = [psi1_0, psi2_0, psi3_0, psi4_0]
            rhs[0] = -Yp
            M[1] = [psi1_0_pp, psi2_0_pp, psi3_0_pp, psi4_0_pp]
            rhs[1] = 0

        elif self.bc_y0 == 'F':  # Free: M_y = 0, V_y = 0
            # ==================================================================
            # CORRECTED SIGN (2026-01-31)
            # ==================================================================
            # Free edge condition from plate theory:
            #   M_y = -D(w_yy + nu*w_xx) = 0
            # For Levy: w = Y(y)*sin(am*x), so w_xx = -am^2*Y*sin(am*x)
            #   M_y = -D(Y'' - nu*am^2*Y)*sin(am*x) = 0
            # Therefore: Y''(0) - nu*am^2*Y(0) = 0  (MINUS sign, not plus!)
            #
            # Shear (Kirchhoff):
            #   V_y = -D(w_yyy + (2-nu)*w_xxy) = 0
            # For Levy: w_xxy = -am^2*Y'*sin(am*x)
            #   V_y = -D(Y''' - (2-nu)*am^2*Y')*sin(am*x) = 0
            # Therefore: Y'''(0) - (2-nu)*am^2*Y'(0) = 0  (MINUS sign!)
            # ==================================================================

            # Moment: Y''(0) - nu*am^2*Y(0) = 0
            M[0] = [psi1_0_pp - nu*am**2*psi1_0,
                    psi2_0_pp - nu*am**2*psi2_0,
                    psi3_0_pp - nu*am**2*psi3_0,
                    psi4_0_pp - nu*am**2*psi4_0]
            rhs[0] = nu * am**2 * Yp  # Note: Yp'' = 0, so RHS = nu*am^2*Yp

            # Shear: Y'''(0) - (2-nu)*am^2*Y'(0) = 0
            M[1] = [psi1_0_ppp - (2-nu)*am**2*psi1_0_p,
                    psi2_0_ppp - (2-nu)*am**2*psi2_0_p,
                    psi3_0_ppp - (2-nu)*am**2*psi3_0_p,
                    psi4_0_ppp - (2-nu)*am**2*psi4_0_p]
            rhs[1] = 0  # Yp' = 0

        # === BCs at y = b ===
        if self.bc_yb == 'C':  # Clamped
            M[2] = [psi1_b, psi2_b, psi3_b, psi4_b]
            rhs[2] = -Yp
            M[3] = [psi1_b_p, psi2_b_p, psi3_b_p, psi4_b_p]
            rhs[3] = 0

        elif self.bc_yb == 'S':  # Simply supported
            M[2] = [psi1_b, psi2_b, psi3_b, psi4_b]
            rhs[2] = -Yp
            M[3] = [psi1_b_pp, psi2_b_pp, psi3_b_pp, psi4_b_pp]
            rhs[3] = 0

        elif self.bc_yb == 'F':  # Free
            # ==================================================================
            # CORRECTED SIGN (2026-01-31)
            # ==================================================================
            # Same derivation as above:
            # Moment: Y''(b) - nu*am^2*Y(b) = 0
            # Shear: Y'''(b) - (2-nu)*am^2*Y'(b) = 0
            # ==================================================================

            # Moment: Y''(b) - nu*am^2*Y(b) = 0
            M[2] = [psi1_b_pp - nu*am**2*psi1_b,
                    psi2_b_pp - nu*am**2*psi2_b,
                    psi3_b_pp - nu*am**2*psi3_b,
                    psi4_b_pp - nu*am**2*psi4_b]
            rhs[2] = nu * am**2 * Yp

            # Shear: Y'''(b) - (2-nu)*am^2*Y'(b) = 0
            M[3] = [psi1_b_ppp - (2-nu)*am**2*psi1_b_p,
                    psi2_b_ppp - (2-nu)*am**2*psi2_b_p,
                    psi3_b_ppp - (2-nu)*am**2*psi3_b_p,
                    psi4_b_ppp - (2-nu)*am**2*psi4_b_p]
            rhs[3] = 0

        # Solve
        cond_num = cond(M)
        try:
            coeffs = solve(M, rhs)
            A, B, C, D_coef = coeffs
        except:
            A, B, C, D_coef = 0, 0, 0, 0

        # Evaluate Ym(y)
        ny = len(y)
        Ym = np.zeros(ny)
        for i, yi in enumerate(y):
            exp1 = np.exp(-am * (b - yi))  # For psi_1, psi_2
            exp2 = np.exp(-am * yi)        # For psi_3, psi_4

            psi1 = exp1
            psi2 = (b - yi) * exp1
            psi3 = exp2
            psi4 = yi * exp2

            Ym[i] = A*psi1 + B*psi2 + C*psi3 + D_coef*psi4 + Yp

        return Ym, cond_num

    def _solve_patch_load_ode(self, am, Qm, y, y1, y2):
        """
        Solve Levy ODE with rectangular patch load in [y1, y2].

        Uses piecewise solution:
        - Region I:   0 <= y < y1  (no load)
        - Region II:  y1 <= y <= y2  (loaded)
        - Region III: y2 < y <= b  (no load)

        Returns Ym(y) evaluated at all y points.
        """
        b = self.b
        nu = self.nu
        D = self.D

        # Particular solution magnitude in loaded region
        Yp = Qm / (D * am**4)

        # Helper to evaluate basis functions at a point
        def eval_basis(yi, am, b):
            exp1 = np.exp(-am * (b - yi))
            exp2 = np.exp(-am * yi)

            psi = np.array([exp1, (b-yi)*exp1, exp2, yi*exp2])
            psi_p = np.array([
                am * exp1,
                (am*(b-yi) - 1) * exp1,
                -am * exp2,
                (1 - am*yi) * exp2
            ])
            psi_pp = np.array([
                am**2 * exp1,
                (am**2*(b-yi) - 2*am) * exp1,
                am**2 * exp2,
                (am**2*yi - 2*am) * exp2
            ])
            psi_ppp = np.array([
                am**3 * exp1,
                (am**3*(b-yi) - 3*am**2) * exp1,
                -am**3 * exp2,
                (-am**3*yi + 3*am**2) * exp2
            ])
            return psi, psi_p, psi_pp, psi_ppp

        # Build 12x12 system
        M = np.zeros((12, 12))
        rhs = np.zeros(12)

        # Get basis at all boundaries
        psi_0, psi_0_p, psi_0_pp, psi_0_ppp = eval_basis(0, am, b)
        psi_y1, psi_y1_p, psi_y1_pp, psi_y1_ppp = eval_basis(y1, am, b)
        psi_y2, psi_y2_p, psi_y2_pp, psi_y2_ppp = eval_basis(y2, am, b)
        psi_b, psi_b_p, psi_b_pp, psi_b_ppp = eval_basis(b, am, b)

        row = 0

        # === Boundary conditions at y = 0 (Region I, coeffs 0-3) ===
        if self.bc_y0 == 'C':  # Clamped
            M[row, 0:4] = psi_0
            rhs[row] = 0  # No Yp in region I
            row += 1
            M[row, 0:4] = psi_0_p
            rhs[row] = 0
            row += 1
        elif self.bc_y0 == 'S':  # Simply supported
            M[row, 0:4] = psi_0
            rhs[row] = 0
            row += 1
            M[row, 0:4] = psi_0_pp
            rhs[row] = 0
            row += 1
        elif self.bc_y0 == 'F':  # Free (corrected signs)
            M[row, 0:4] = psi_0_pp - nu*am**2*psi_0
            rhs[row] = 0
            row += 1
            M[row, 0:4] = psi_0_ppp - (2-nu)*am**2*psi_0_p
            rhs[row] = 0
            row += 1

        # === Continuity at y = y1 (rows 2-5) ===
        # Region I (coeffs 0-3) = Region II (coeffs 4-7) + Yp
        # Y: Y_I(y1) = Y_II(y1)
        M[row, 0:4] = psi_y1
        M[row, 4:8] = -psi_y1
        rhs[row] = Yp  # Y_II has Yp
        row += 1
        # Y': Y_I'(y1) = Y_II'(y1)
        M[row, 0:4] = psi_y1_p
        M[row, 4:8] = -psi_y1_p
        rhs[row] = 0
        row += 1
        # Y'': Y_I''(y1) = Y_II''(y1)
        M[row, 0:4] = psi_y1_pp
        M[row, 4:8] = -psi_y1_pp
        rhs[row] = 0
        row += 1
        # Y''': Y_I'''(y1) = Y_II'''(y1)
        M[row, 0:4] = psi_y1_ppp
        M[row, 4:8] = -psi_y1_ppp
        rhs[row] = 0
        row += 1

        # === Continuity at y = y2 (rows 6-9) ===
        # Region II (coeffs 4-7) + Yp = Region III (coeffs 8-11)
        M[row, 4:8] = psi_y2
        M[row, 8:12] = -psi_y2
        rhs[row] = -Yp  # Y_II has Yp, Y_III doesn't
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

        # === Boundary conditions at y = b (Region III, coeffs 8-11) ===
        if self.bc_yb == 'C':  # Clamped
            M[row, 8:12] = psi_b
            rhs[row] = 0
            row += 1
            M[row, 8:12] = psi_b_p
            rhs[row] = 0
        elif self.bc_yb == 'S':  # Simply supported
            M[row, 8:12] = psi_b
            rhs[row] = 0
            row += 1
            M[row, 8:12] = psi_b_pp
            rhs[row] = 0
        elif self.bc_yb == 'F':  # Free (corrected signs)
            M[row, 8:12] = psi_b_pp - nu*am**2*psi_b
            rhs[row] = 0
            row += 1
            M[row, 8:12] = psi_b_ppp - (2-nu)*am**2*psi_b_p
            rhs[row] = 0

        # Solve 12x12 system
        cond_num = cond(M)
        try:
            coeffs = solve(M, rhs)
        except:
            coeffs = np.zeros(12)

        # Evaluate Ym(y) in each region
        Ym = np.zeros_like(y)
        for i, yi in enumerate(y):
            psi, _, _, _ = eval_basis(yi, am, b)

            if yi < y1:
                Ym[i] = np.dot(coeffs[0:4], psi)
            elif yi <= y2:
                Ym[i] = np.dot(coeffs[4:8], psi) + Yp
            else:
                Ym[i] = np.dot(coeffs[8:12], psi)

        return Ym, cond_num

    def _solve_circular_load(self, am, q0, x0, y0, R, y, m):
        """
        Solve for Ym(y) with circular patch load using strip superposition.

        Divides the circle into horizontal strips and sums the contributions
        from each strip, properly accounting for the varying x-extent.
        """
        n_strips = 20  # Number of horizontal strips
        Ym = np.zeros_like(y)
        cond_num = 1.0

        circ_y1 = max(0, y0 - R)
        circ_y2 = min(self.b, y0 + R)

        # Use strips to approximate the circular load
        strip_y = np.linspace(circ_y1, circ_y2, n_strips + 1)

        for i in range(n_strips):
            # Strip bounds
            y1_strip = strip_y[i]
            y2_strip = strip_y[i + 1]
            y_mid = (y1_strip + y2_strip) / 2

            # At this y-level, compute the x-extent of the circle
            dy_from_center = y_mid - y0
            if abs(dy_from_center) >= R:
                continue

            dx = np.sqrt(R**2 - dy_from_center**2)
            x1_strip = max(0, x0 - dx)
            x2_strip = min(self.a, x0 + dx)

            if x2_strip <= x1_strip:
                continue

            # Compute qm for this strip
            qm_strip = self._rect_patch_x_coef(q0, x1_strip, x2_strip, m)

            if abs(qm_strip) < 1e-30:
                continue

            # Solve ODE for this strip
            Ym_strip, cn = self._solve_patch_load_ode(am, qm_strip, y, y1_strip, y2_strip)
            cond_num = max(cond_num, cn)

            Ym += Ym_strip

        return Ym, cond_num

    def _circular_load_coef(self, p0, R, x0, m):
        """Fourier coefficient for circular patch load (legacy, not used for full solve)."""
        am = m * np.pi / self.a
        if R * am < 0.5:
            # Small radius - approximate as point load with total force pi*R^2*p0
            return 2.0 * (np.pi * R**2 * p0) / self.a * np.sin(am * x0)
        else:
            # Use Bessel function
            return 2.0 / self.a * p0 * 2*np.pi*R * jv(1, am*R) / am * np.sin(am * x0)

    def _compute_derived(self):
        """Compute moments, stresses, validation coefficients."""
        X, Y, W = self.results['X'], self.results['Y'], self.results['W']
        dx, dy = X[0,1] - X[0,0], Y[1,0] - Y[0,0]

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
            'W_coef': np.max(np.abs(W)) * self.D / (q0 * self.a**4),
            'Mx_coef': np.max(np.abs(Mx)) / (q0 * self.a**2),
            'My_coef': np.max(np.abs(My)) / (q0 * self.a**2),
        })

        # Center values
        ny, nx = W.shape
        ic, jc = ny//2, nx//2
        self.results['W_center'] = W[ic, jc]
        self.results['W_center_coef'] = abs(W[ic, jc]) * self.D / (q0 * self.a**4)
