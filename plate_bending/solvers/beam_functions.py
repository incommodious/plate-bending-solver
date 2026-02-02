"""
Beam Characteristic Functions for Rayleigh-Ritz Plate Analysis
===============================================================

Provides beam vibration eigenfunctions and their eigenvalues for various
boundary condition combinations. These are used as trial functions in the
Rayleigh-Ritz method for plate bending.

Reference: Blevins, "Formulas for Natural Frequency and Mode Shape" (1979)
"""

import numpy as np
from scipy.optimize import brentq


# Eigenvalue tables for different BC types
# These are the first few roots of the characteristic equations
EIGENVALUES = {
    'SS': None,  # Exact: beta_n = n*pi
    'CC': [4.73004074, 7.85320462, 10.99560784, 14.13716549, 17.27875966],
    'FF': [4.73004074, 7.85320462, 10.99560784, 14.13716549, 17.27875966],
    'CF': [1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839],
    'FC': [1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839],  # Same as CF
    'CS': [3.92660231, 7.06858275, 10.21017612, 13.35176878, 16.49336143],
    'SC': [3.92660231, 7.06858275, 10.21017612, 13.35176878, 16.49336143],  # Same as CS
    # SF/FS: Using quarter-wave sines for Ritz: beta = (2n-1)*pi/2
    'SF': [1.57079633, 4.71238898, 7.85398163, 10.99557429, 14.13716694],
    'FS': [1.57079633, 4.71238898, 7.85398163, 10.99557429, 14.13716694],
}


def get_eigenvalue(n, bc_type):
    """
    Get the n-th eigenvalue for the given BC type.

    Parameters
    ----------
    n : int
        Mode number (1-indexed)
    bc_type : str
        Two-character BC type: 'SS', 'CC', 'FF', 'CF', 'CS', 'SF', etc.
        First char is BC at x=0, second is BC at x=L.

    Returns
    -------
    float
        The eigenvalue beta_n * L
    """
    bc = bc_type.upper()

    if bc == 'SS':
        return n * np.pi

    if bc in EIGENVALUES:
        eigenvals = EIGENVALUES[bc]
        if n <= len(eigenvals):
            return eigenvals[n - 1]
        else:
            # Asymptotic approximation for higher modes
            if bc in ['CF', 'FC']:
                return (2*n - 1) * np.pi / 2
            elif bc in ['CS', 'SC']:
                return (4*n + 1) * np.pi / 4
            elif bc in ['SF', 'FS']:
                return (2*n - 1) * np.pi / 2  # Quarter-wave sine: (2n-1)*pi/2
            elif bc in ['CC', 'FF']:
                return (2*n + 1) * np.pi / 2
    else:
        raise ValueError(f"Unknown BC type: {bc}")


def beam_function(xi, n, bc_type, deriv=0):
    """
    Evaluate beam eigenfunction and its derivatives.

    Parameters
    ----------
    xi : array_like
        Normalized coordinate x/L in [0, 1]
    n : int
        Mode number (1-indexed)
    bc_type : str
        Two-character BC type
    deriv : int
        Derivative order (0, 1, 2, 3, or 4)

    Returns
    -------
    array_like
        Function value(s) at xi
    """
    xi = np.asarray(xi)
    bc = bc_type.upper()
    beta = get_eigenvalue(n, bc)

    if bc == 'SS':
        # Simply supported: phi_n = sin(n*pi*xi)
        if deriv == 0:
            return np.sin(beta * xi)
        elif deriv == 1:
            return beta * np.cos(beta * xi)
        elif deriv == 2:
            return -beta**2 * np.sin(beta * xi)
        elif deriv == 3:
            return -beta**3 * np.cos(beta * xi)
        elif deriv == 4:
            return beta**4 * np.sin(beta * xi)

    elif bc in ['CF', 'FC']:
        # Clamped-Free (cantilever)
        sigma = (np.sinh(beta) - np.sin(beta)) / (np.cosh(beta) + np.cos(beta))

        ch = np.cosh(beta * xi)
        sh = np.sinh(beta * xi)
        co = np.cos(beta * xi)
        si = np.sin(beta * xi)

        if deriv == 0:
            return ch - co - sigma * (sh - si)
        elif deriv == 1:
            return beta * (sh + si - sigma * (ch - co))
        elif deriv == 2:
            return beta**2 * (ch + co - sigma * (sh + si))
        elif deriv == 3:
            return beta**3 * (sh - si - sigma * (ch - co))
        elif deriv == 4:
            return beta**4 * (ch - co - sigma * (sh - si))

    elif bc in ['CS', 'SC']:
        # Clamped-Simply supported
        sigma = (np.sinh(beta) - np.sin(beta)) / (np.cosh(beta) - np.cos(beta))

        ch = np.cosh(beta * xi)
        sh = np.sinh(beta * xi)
        co = np.cos(beta * xi)
        si = np.sin(beta * xi)

        if deriv == 0:
            return sh - si - sigma * (ch - co)
        elif deriv == 1:
            return beta * (ch - co - sigma * (sh + si))
        elif deriv == 2:
            return beta**2 * (sh + si - sigma * (ch + co))
        elif deriv == 3:
            return beta**3 * (ch + co - sigma * (sh - si))
        elif deriv == 4:
            return beta**4 * (sh - si - sigma * (ch - co))

    elif bc in ['SF', 'FS']:
        # Simply supported-Free
        # Use quarter-wave sine functions as trial functions for Ritz method.
        # These satisfy w(0)=0 and allow free deflection at xi=1.
        # beta_eff = (2n-1)*pi/2 gives sin values of +/-1 at xi=1.
        beta_eff = (2*n - 1) * np.pi / 2

        if deriv == 0:
            return np.sin(beta_eff * xi)
        elif deriv == 1:
            return beta_eff * np.cos(beta_eff * xi)
        elif deriv == 2:
            return -beta_eff**2 * np.sin(beta_eff * xi)
        elif deriv == 3:
            return -beta_eff**3 * np.cos(beta_eff * xi)
        elif deriv == 4:
            return beta_eff**4 * np.sin(beta_eff * xi)

    elif bc == 'CC':
        # Clamped-Clamped
        sigma = (np.cosh(beta) - np.cos(beta)) / (np.sinh(beta) - np.sin(beta))

        ch = np.cosh(beta * xi)
        sh = np.sinh(beta * xi)
        co = np.cos(beta * xi)
        si = np.sin(beta * xi)

        if deriv == 0:
            return ch - co - sigma * (sh - si)
        elif deriv == 1:
            return beta * (sh + si - sigma * (ch - co))
        elif deriv == 2:
            return beta**2 * (ch + co - sigma * (sh + si))
        elif deriv == 3:
            return beta**3 * (sh - si - sigma * (ch - co))
        elif deriv == 4:
            return beta**4 * (ch - co - sigma * (sh - si))

    elif bc == 'FF':
        # Free-Free
        if n <= 2:
            # Rigid body modes for n=1,2
            if n == 1:
                return np.ones_like(xi) if deriv == 0 else np.zeros_like(xi)
            else:  # n == 2
                if deriv == 0:
                    return xi - 0.5
                elif deriv == 1:
                    return np.ones_like(xi)
                else:
                    return np.zeros_like(xi)

        # For n > 2, use the bending modes (same eigenvalues as CC)
        beta = EIGENVALUES['FF'][n - 3] if n - 2 <= len(EIGENVALUES['FF']) else (2*(n-2) + 1) * np.pi / 2
        sigma = (np.cosh(beta) - np.cos(beta)) / (np.sinh(beta) - np.sin(beta))

        ch = np.cosh(beta * xi)
        sh = np.sinh(beta * xi)
        co = np.cos(beta * xi)
        si = np.sin(beta * xi)

        if deriv == 0:
            return ch + co - sigma * (sh + si)
        elif deriv == 1:
            return beta * (sh - si - sigma * (ch - co))
        elif deriv == 2:
            return beta**2 * (ch - co - sigma * (sh - si))
        elif deriv == 3:
            return beta**3 * (sh + si - sigma * (ch + co))
        elif deriv == 4:
            return beta**4 * (ch + co - sigma * (sh + si))

    else:
        raise ValueError(f"Unknown BC type: {bc}")


def compute_beam_integrals(n_modes, bc_type, L=1.0, numerical=True):
    """
    Compute the integrals needed for Ritz stiffness matrix.

    I0[m,p] = integral_0^L phi_m(x) * phi_p(x) dx
    I1[m,p] = integral_0^L phi_m'(x) * phi_p'(x) dx
    I2[m,p] = integral_0^L phi_m''(x) * phi_p''(x) dx

    Parameters
    ----------
    n_modes : int
        Number of modes to compute
    bc_type : str
        Boundary condition type
    L : float
        Length of beam
    numerical : bool
        If True, use numerical integration. If False, use analytical formulas.

    Returns
    -------
    I0, I1, I2 : ndarray
        Integral matrices (n_modes x n_modes)
    """
    I0 = np.zeros((n_modes, n_modes))
    I1 = np.zeros((n_modes, n_modes))
    I2 = np.zeros((n_modes, n_modes))

    bc = bc_type.upper()

    if bc == 'SS' and not numerical:
        # Analytical: orthogonal with I0[m,m] = L/2
        for m in range(n_modes):
            I0[m, m] = L / 2
            beta_m = (m + 1) * np.pi / L
            I1[m, m] = beta_m**2 * L / 2
            I2[m, m] = beta_m**4 * L / 2
        return I0, I1, I2

    # Numerical integration using Gauss-Legendre quadrature
    from scipy.integrate import fixed_quad

    n_quad = 50  # Number of quadrature points

    for m in range(1, n_modes + 1):
        for p in range(m, n_modes + 1):  # Exploit symmetry

            # I0: integral of phi_m * phi_p
            def f0(xi):
                return beam_function(xi, m, bc, 0) * beam_function(xi, p, bc, 0)
            val0, _ = fixed_quad(f0, 0, 1, n=n_quad)
            I0[m-1, p-1] = val0 * L
            I0[p-1, m-1] = I0[m-1, p-1]  # Symmetric

            # I1: integral of phi_m' * phi_p' (scaled by 1/L^2 from chain rule)
            def f1(xi):
                return beam_function(xi, m, bc, 1) * beam_function(xi, p, bc, 1)
            val1, _ = fixed_quad(f1, 0, 1, n=n_quad)
            I1[m-1, p-1] = val1 / L  # (1/L) * (1/L) * L = 1/L
            I1[p-1, m-1] = I1[m-1, p-1]

            # I2: integral of phi_m'' * phi_p'' (scaled by 1/L^4)
            def f2(xi):
                return beam_function(xi, m, bc, 2) * beam_function(xi, p, bc, 2)
            val2, _ = fixed_quad(f2, 0, 1, n=n_quad)
            I2[m-1, p-1] = val2 / L**3  # (1/L^2) * (1/L^2) * L = 1/L^3
            I2[p-1, m-1] = I2[m-1, p-1]

    return I0, I1, I2


def compute_load_integral(n, bc_type, xi1=0, xi2=1, L=1.0):
    """
    Compute load integral for uniform load over region [xi1, xi2].

    integral_{xi1}^{xi2} phi_n(xi) dxi

    Parameters
    ----------
    n : int
        Mode number (1-indexed)
    bc_type : str
        BC type
    xi1, xi2 : float
        Integration bounds in normalized coordinates [0, 1]
    L : float
        Length

    Returns
    -------
    float
        Integral value scaled by L
    """
    from scipy.integrate import fixed_quad

    def f(xi):
        return beam_function(xi, n, bc_type, 0)

    val, _ = fixed_quad(f, xi1, xi2, n=50)
    return val * L
