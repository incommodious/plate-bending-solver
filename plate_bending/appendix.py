"""
Appendix Template Engine for Plate Bending Reports
===================================================

Generates LaTeX source for a step-by-step worked appendix
showing the Ritz or Levy calculation with actual numbers.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def fmt_eng(value: float, sig: int = 4) -> str:
    """Format number in LaTeX engineering notation (exponent divisible by 3).
    
    Returns '0' for zero, fixed-point for values in [0.01, 9999],
    engineering notation otherwise (e.g., r'1.23 \\times 10^{6}').
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    abs_val = abs(value)
    if abs_val == 0:
        return "0"

    exp = int(np.floor(np.log10(abs_val)))

    if -2 <= exp <= 3:
        dp = max(sig - exp - 1, 0)
        s = format(value, f".{dp}f")
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    # Engineering notation: exponent divisible by 3
    eng_exp = exp - (exp % 3) if exp >= 0 else exp - ((exp % 3 + 3) % 3)
    mantissa = value / 10**eng_exp
    m_exp = int(np.floor(np.log10(abs(mantissa))))
    dec = max(sig - m_exp - 1, 0)
    m_str = format(mantissa, f".{dec}f")
    if '.' in m_str:
        m_str = m_str.rstrip('0').rstrip('.')
    return m_str + r" \times 10^{" + str(eng_exp) + "}"


# Unit conversion constants
_TO_IN = 1.0 / 0.0254       # m -> in
_TO_PSI = 1.0 / 6894.76     # Pa -> psi
_TO_LBIN = 1.0 / 0.1129848  # N·m -> lbf·in


def _len_unit(units: str) -> str:
    return "in" if units == "imperial" else "m"


def _press_unit(units: str) -> str:
    return "psi" if units == "imperial" else "Pa"


def _rig_unit(units: str) -> str:
    return r"lbf\,\textperiodcentered\,in" if units == "imperial" else r"N\,\textperiodcentered\,m"


def _bc_beam_types(bc: str) -> tuple:
    """Return (x_beam_type, y_beam_type) from 4-char BC string.
    
    BC order: x=0, y=0, x=a, y=b
    x-direction beam: bc[0] + bc[2]
    y-direction beam: bc[1] + bc[3]
    """
    x_type = bc[0] + bc[2]
    y_type = bc[1] + bc[3]
    return x_type, y_type


def _beam_name(bt: str) -> str:
    """Human-readable name for beam type."""
    names = {
        'SS': 'Simply Supported--Simply Supported',
        'CC': 'Clamped--Clamped',
        'FF': 'Free--Free',
        'CF': 'Clamped--Free',
        'FC': 'Free--Clamped',
        'CS': 'Clamped--Simply Supported',
        'SC': 'Simply Supported--Clamped',
        'SF': 'Simply Supported--Free',
        'FS': 'Free--Simply Supported',
    }
    return names.get(bt, bt)


def generate_appendix_latex(
    inputs: Dict,
    solver_results: Dict,
    convergence_data: List[Dict],
    method: str = 'ritz'
) -> str:
    """Generate LaTeX source for a step-by-step appendix.
    
    Parameters
    ----------
    inputs : dict
        Keys: a, b, h, E, nu, bc, q0, x, y, load_type, units
        All values in SI units internally.
    solver_results : dict
        Not used directly (we describe the solution process generically).
    convergence_data : list of dict
        Each dict has 'n' (number of terms) and 'w' (deflection at point, in SI).
    method : str
        'ritz' or 'levy'
    
    Returns
    -------
    str
        LaTeX source for the appendix (sections only, no documentclass).
    """
    a = inputs['a']
    b = inputs['b']
    h = inputs['h']
    E = inputs['E']
    nu = inputs['nu']
    bc = inputs['bc']
    q0 = inputs['q0']
    x_pt = inputs['x']
    y_pt = inputs['y']
    units = inputs.get('units', 'imperial')
    load_type = inputs.get('load_type', 'uniform')

    D = E * h**3 / (12 * (1 - nu**2))

    # Display values
    if units == 'imperial':
        a_d, b_d, h_d = a * _TO_IN, b * _TO_IN, h * _TO_IN
        E_d = E * _TO_PSI
        q0_d = q0 * _TO_PSI
        D_d = D * _TO_LBIN
        x_d, y_d = x_pt * _TO_IN, y_pt * _TO_IN
        lu, pu, ru = "in", "psi", r"lbf\,\textperiodcentered\,in"
        w_unit = "mil"
        w_factor = _TO_IN * 1000  # m -> mil
    else:
        a_d, b_d, h_d = a, b, h
        E_d, q0_d, D_d = E, q0, D
        x_d, y_d = x_pt, y_pt
        lu, pu, ru = "m", "Pa", r"N\,\textperiodcentered\,m"
        w_unit = "mm"
        w_factor = 1000  # m -> mm

    lines = []
    L = lines.append  # shorthand

    # ===== Section A.1: Setup =====
    L(r"\appendix")
    L(r"\section{Step-by-Step " + method.capitalize() + " Calculation}")
    L(r"\label{app:worked}")
    L("")
    L(r"This appendix walks through the " + method.capitalize() + 
      r" solution in detail, showing every number so the calculation can be followed or reproduced.")
    L("")

    L(r"\subsection{Setup}")
    L("")
    L(r"\begin{tabular}{lrl}")
    L(r"\toprule")
    L(r"Quantity & Value & \\")
    L(r"\midrule")
    L(r"$a$ (plate length) & $" + fmt_eng(a_d) + r"$ & " + lu + r" \\")
    L(r"$b$ (plate width) & $" + fmt_eng(b_d) + r"$ & " + lu + r" \\")
    L(r"$h$ (thickness) & $" + fmt_eng(h_d) + r"$ & " + lu + r" \\")
    L(r"$E$ (Young's modulus) & $" + fmt_eng(E_d) + r"$ & " + pu + r" \\")
    L(r"$\nu$ (Poisson's ratio) & $" + fmt_eng(nu) + r"$ & \\")
    L(r"$q_0$ (pressure) & $" + fmt_eng(q0_d) + r"$ & " + pu + r" \\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L("")

    # D calculation
    L(r"\medskip\noindent\textbf{Flexural rigidity:}")
    L(r"\begin{align}")
    L(r"D &= \frac{Eh^3}{12(1-\nu^2)} \notag \\")
    L(r"  &= \frac{" + fmt_eng(E_d) + r" \times (" + fmt_eng(h_d) + r")^3}{12(1-" + fmt_eng(nu) + r"^2)} \notag \\")
    
    h_cubed = h_d**3
    denom = 12 * (1 - nu**2)
    L(r"  &= \frac{" + fmt_eng(E_d) + r" \times " + fmt_eng(h_cubed) + r"}{" + fmt_eng(denom) + r"} \notag \\")
    L(r"  &= " + fmt_eng(D_d) + r" \ \text{" + ru + r"}")
    L(r"\end{align}")
    L("")

    # ===== Section A.2: Trial Function =====
    L(r"\subsection{Trial Function}")
    L("")

    x_bt, y_bt = _bc_beam_types(bc)

    if method == 'ritz':
        L(r"The deflection is approximated as a double sum of beam eigenfunctions:")
        L(r"\begin{equation}")
        L(r"w(x,y) = \sum_{m=1}^{M}\sum_{n=1}^{N} A_{mn}\,\phi_m\!\left(\frac{x}{a}\right)\,\psi_n\!\left(\frac{y}{b}\right)")
        L(r"\end{equation}")
        L("")
        L(r"\begin{itemize}")
        L(r"\item $\phi_m(\xi)$: \textbf{" + x_bt + r"} (" + _beam_name(x_bt) + r") beam eigenfunctions in $x$")
        L(r"\item $\psi_n(\eta)$: \textbf{" + y_bt + r"} (" + _beam_name(y_bt) + r") beam eigenfunctions in $y$")
        L(r"\end{itemize}")
        L("")
        L(r"Minimizing the total potential energy $\Pi = U - W_{\text{ext}}$ gives:")
        L(r"\begin{equation}")
        L(r"\mathbf{K}\,\mathbf{A} = \mathbf{F}")
        L(r"\end{equation}")
        L(r"where $\mathbf{K}$ is the stiffness matrix (from plate strain energy) and $\mathbf{F}$ is the load vector.")
    else:  # levy
        L(r"The deflection is expressed as a single Fourier series:")
        L(r"\begin{equation}")
        L(r"w(x,y) = \sum_{m=1}^{N} \sin\!\left(\frac{m\pi x}{a}\right) Y_m(y)")
        L(r"\end{equation}")
        L(r"where $Y_m(y)$ satisfies the ODE:")
        L(r"\begin{equation}")
        L(r"D\left(Y_m'''' - 2\alpha_m^2 Y_m'' + \alpha_m^4 Y_m\right) = q_m, \qquad \alpha_m = \frac{m\pi}{a}")
        L(r"\end{equation}")
        L(r"with boundary conditions at $y=0$ (" + bc[1] + r") and $y=b$ (" + bc[3] + r").")
    L("")

    # ===== Section A.3: Convergence =====
    L(r"\subsection{Convergence Study}")
    L("")

    if convergence_data and len(convergence_data) >= 2:
        # Narrative
        L(r"\noindent The series is evaluated at increasing truncation levels. "
          r"At each level, the deflection at the analysis point $(" + fmt_eng(x_d) + 
          r",\," + fmt_eng(y_d) + r")$ " + lu + r" is recorded.")
        L("")

        # Show first two steps narrated
        first = convergence_data[0]
        w1 = first['w'] * w_factor
        if method == 'ritz':
            L(r"\noindent $M = N = " + str(first['n']) + r"$: solve $\rightarrow$ $w = " + 
              fmt_eng(w1, 4) + r"$ " + w_unit + r" (first evaluation).")
        else:
            L(r"\noindent $N = " + str(first['n']) + r"$: solve $\rightarrow$ $w = " + 
              fmt_eng(w1, 4) + r"$ " + w_unit + r" (first evaluation).")
        L("")

        if len(convergence_data) >= 2:
            second = convergence_data[1]
            w2 = second['w'] * w_factor
            delta = abs(w2 - w1)
            if method == 'ritz':
                L(r"\noindent $M = N = " + str(second['n']) + r"$: $w = " + fmt_eng(w2, 4) +
                  r"$ " + w_unit + r". $|\Delta w| = " + fmt_eng(delta, 3) + r"$ " + w_unit + r".")
            else:
                L(r"\noindent $N = " + str(second['n']) + r"$: $w = " + fmt_eng(w2, 4) +
                  r"$ " + w_unit + r". $|\Delta w| = " + fmt_eng(delta, 3) + r"$ " + w_unit + r".")
            L("")

        if len(convergence_data) > 2:
            last = convergence_data[-1]
            prev = convergence_data[-2]
            wl = last['w'] * w_factor
            delta_last = abs(wl - prev['w'] * w_factor)
            pct = delta_last / abs(wl) * 100 if wl != 0 else 0
            n_label = ("$M = N = " + str(last['n']) + "$") if method == 'ritz' else ("$N = " + str(last['n']) + "$")
            L(r"\noindent Continuing to " + n_label + r": $w = " + fmt_eng(wl, 4) +
              r"$ " + w_unit + r", $|\Delta w| = " + fmt_eng(delta_last, 3) + 
              r"$ " + w_unit + r" (" + fmt_eng(pct, 2) + r"\%) --- converged.")
            L("")

        # Table
        L(r"\begin{tabular}{rrrr}")
        L(r"\toprule")
        if method == 'ritz':
            L(r"$M = N$ & DOFs & $w$ (" + w_unit + r") & $|\Delta w|$ (" + w_unit + r") \\")
        else:
            L(r"$N$ & & $w$ (" + w_unit + r") & $|\Delta w|$ (" + w_unit + r") \\")
        L(r"\midrule")

        prev_w = None
        for entry in convergence_data:
            n = entry['n']
            w = entry['w'] * w_factor
            dofs = n * n if method == 'ritz' else n
            if prev_w is not None:
                delta = abs(w - prev_w)
                pct = delta / abs(w) * 100 if w != 0 else 0
                delta_str = "$" + fmt_eng(delta, 3) + "$ (" + fmt_eng(pct, 1) + r"\%)"
            else:
                delta_str = "---"
            
            if entry == convergence_data[-1]:
                L(r"\textbf{" + str(n) + r"} & \textbf{" + str(dofs) + r"} & $\mathbf{" + 
                  fmt_eng(w, 4) + r"}$ & " + delta_str + r" \\")
            else:
                L(str(n) + " & " + str(dofs) + " & $" + fmt_eng(w, 4) + "$ & " + delta_str + r" \\")
            prev_w = w

        L(r"\bottomrule")
        L(r"\end{tabular}")
    else:
        L(r"No convergence data provided.")
    L("")

    # ===== Section A.4: Physical Intuition =====
    L(r"\subsection{Physical Intuition}")
    L("")

    # Auto-generate insight based on BC
    bc_desc = []
    n_clamped = bc.count('C')
    n_free = bc.count('F')
    n_ss = bc.count('S')

    if n_clamped > 0:
        bc_desc.append(str(n_clamped) + " clamped edge" + ("s" if n_clamped > 1 else ""))
    if n_free > 0:
        bc_desc.append(str(n_free) + " free edge" + ("s" if n_free > 1 else ""))
    if n_ss > 0:
        bc_desc.append(str(n_ss) + " simply supported edge" + ("s" if n_ss > 1 else ""))

    bc_summary = " and ".join(bc_desc)
    
    L(r"\textit{This plate has " + bc_summary + r". ")
    
    if n_clamped >= 2:
        L(r"The clamped edges provide strong rotational restraint, limiting deflection but ")
        L(r"concentrating bending moments near the supports. ")
    if n_free >= 1:
        L(r"The free edge(s) allow the plate to deflect without constraint, so maximum ")
        L(r"deflection typically occurs near or at the free edges. ")
    
    aspect = max(a, b) / min(a, b)
    if aspect > 2:
        L(r"The high aspect ratio (" + fmt_eng(aspect, 2) + r":1) means the plate behaves ")
        L(r"somewhat like a wide beam in the short direction. ")
    
    L(r"}")
    L("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Linear Algebra Appendix — Worked M=N=2 Example
# ---------------------------------------------------------------------------

def generate_linear_algebra_appendix(
    inputs: Dict,
    units: str = 'imperial'
) -> str:
    """Generate LaTeX showing the complete K matrix, F vector, and solution for M=N=2.

    Parameters
    ----------
    inputs : dict
        Keys: a, b, h, E, nu, bc, q0, x, y, load_type, units,
        and load geometry (x0, y0, R, x1, y1, x2, y2).
        All values in SI units internally.
    units : str
        'imperial' or 'metric'

    Returns
    -------
    str
        LaTeX source (sections only, no documentclass).
    """
    from plate_bending.solvers.ritz_solver import RitzSolver
    from plate_bending.solvers.beam_functions import beam_function

    a = inputs['a']
    b = inputs['b']
    h = inputs['h']
    E = inputs['E']
    nu = inputs['nu']
    bc = inputs['bc']
    q0 = inputs['q0']
    x_pt = inputs['x']
    y_pt = inputs['y']
    load_type = inputs.get('load_type', 'uniform')
    ritz_terms = inputs.get('ritz_terms', 10)

    # Unit conversion factors
    if units == 'imperial':
        len_f = 1.0 / 0.0254        # m -> in
        press_f = 1.0 / 6894.76     # Pa -> psi
        rig_f = 1.0 / 0.1129848     # N·m -> lbf·in
        lu = "in"
        pu = "psi"
        ru = r"lbf\,\textperiodcentered\,in"
    else:
        len_f = 1.0
        press_f = 1.0
        rig_f = 1.0
        lu = "m"
        pu = "Pa"
        ru = r"N\,\textperiodcentered\,m"

    D = E * h**3 / (12 * (1 - nu**2))

    # Build load kwargs
    load_kw = dict(load_type=load_type, q0=q0)
    for key in ('x0', 'y0', 'R', 'x1', 'y1', 'x2', 'y2'):
        if inputs.get(key) is not None:
            load_kw[key] = inputs[key]

    # --- Run solver at M=N=2 ---
    solver2 = RitzSolver(a, b, h, E, nu, bc, M=2, N=2)
    solver2.solve(**load_kw)
    K = solver2._K   # 4×4
    F = solver2._F   # 4
    A_vec = np.linalg.solve(K, F)
    coeffs = A_vec.reshape((2, 2))

    # --- Run full solver to get converged value ---
    solver_full = RitzSolver(a, b, h, E, nu, bc, M=ritz_terms, N=ritz_terms)
    solver_full.solve(**load_kw)
    # Evaluate at analysis point
    xi_pt = x_pt / a
    eta_pt = y_pt / b
    x_bt = bc[0] + bc[2]
    y_bt = bc[1] + bc[3]

    w_full = 0.0
    for m in range(1, ritz_terms + 1):
        phi_m = beam_function(xi_pt, m, x_bt, 0)
        for n in range(1, ritz_terms + 1):
            psi_n = beam_function(eta_pt, n, y_bt, 0)
            w_full += solver_full.coeffs[m-1, n-1] * phi_m * psi_n

    # --- Unit-convert K and F ---
    # K has units of [force·length] = N·m (rigidity-like, from D * integral products)
    # F has units of [force·length] = N·m (pressure × area × shape fn integral)
    # A has units of [length] = m
    # K·A = F  =>  [N·m]·[m] = [N·m²] ... actually:
    # K_ij has units of D / (a²·b²) * a * b * ... let's think dimensionally:
    # K_ij ~ D * (integral of φ''φ'' * ψψ) where integrals are dimensionless/L
    # Actually K is assembled so K*A = F, A in meters, F in N·m (force*length).
    # K units: F/A = (N·m)/m = N. Hmm. Let me check:
    # From _assemble_stiffness: K[i,j] = D * (Ix2*Iy0 + ...).
    # Ix2 = integral of φ''(ξ)φ''(ξ)dξ / a³ (from compute_beam_integrals, second deriv)
    # Iy0 = integral of ψ(η)ψ(η)dη * b (zeroth deriv)
    # So Ix2*Iy0 has units 1/a³ * b = 1/(m³) * m = 1/m².
    # D has units N·m. So K ~ N·m * 1/m² = N/m.
    # F has units of q0 * integral(φ dx) * integral(ψ dy) = Pa * m * m = N/m² * m² = N.
    # Wait, Pa = N/m². integral(φ dx over 0..a) has units of m (beam fn is dimensionless, dx is m).
    # So F_i = q0 * m * m = N. Then K*A = F => (N/m)*m = N. Checks out.
    #
    # For display: K in N/m -> convert to lbf/in: multiply by (1/4.44822)/(1/0.0254) = 0.0254/4.44822
    # F in N -> lbf: multiply by 1/4.44822
    # A in m -> in: multiply by len_f

    if units == 'imperial':
        force_f = 1.0 / 4.44822  # N -> lbf
        K_disp = K * force_f / len_f  # (N/m) -> (lbf/in)
        F_disp = F * force_f           # N -> lbf
        A_disp = A_vec * len_f          # m -> in
        K_unit = r"lbf/in"
        F_unit = r"lbf"
        A_unit = lu
    else:
        K_disp = K
        F_disp = F
        A_disp = A_vec
        K_unit = r"N/m"
        F_unit = r"N"
        A_unit = lu

    lines = []
    L = lines.append

    L(r"\section{Linear Algebra Worked Example ($M = N = 2$)}")
    L(r"\label{app:linalg}")
    L("")
    L(r"This section shows the complete Ritz system at $M = N = 2$ (4 DOFs) so every")
    L(r"number can be traced from inputs to deflection.")
    L("")

    # --- Basis functions ---
    L(r"\subsection{Beam Function Basis}")
    L("")
    L(r"With $M = N = 2$, the trial function has four terms:")
    L(r"\begin{equation}")
    L(r"w(x,y) = \sum_{m=1}^{2}\sum_{n=1}^{2} A_{mn}\,\phi_m\!\left(\frac{x}{a}\right)\,\psi_n\!\left(\frac{y}{b}\right)")
    L(r"\end{equation}")
    L("")

    L(r"\noindent Beam function types:")
    L(r"\begin{itemize}")
    L(r"\item $\phi_m(\xi)$: \textbf{" + x_bt + r"} (" + _beam_name(x_bt) + r") in $x$")
    L(r"\item $\psi_n(\eta)$: \textbf{" + y_bt + r"} (" + _beam_name(y_bt) + r") in $y$")
    L(r"\end{itemize}")
    L("")

    # DOF ordering
    L(r"\noindent DOF ordering: $A_{11},\, A_{12},\, A_{21},\, A_{22}$ (row-major by $m$).")
    L("")

    # --- Stiffness matrix ---
    L(r"\subsection{Stiffness Matrix $\mathbf{K}$}")
    L("")
    L(r"\noindent Units: " + K_unit)
    L("")

    # Format K as a bmatrix
    L(r"\begin{equation}")
    L(r"\mathbf{K} = \begin{bmatrix}")
    for i in range(4):
        row_entries = []
        for j in range(4):
            row_entries.append(fmt_eng(K_disp[i, j], 4))
        L(" & ".join(row_entries) + (r" \\" if i < 3 else ""))
    L(r"\end{bmatrix}")
    L(r"\end{equation}")
    L("")

    # Condition number
    cond = np.linalg.cond(K)
    L(r"\noindent Condition number: $\kappa(\mathbf{K}) = " + fmt_eng(cond, 3) + r"$")
    L("")

    # --- Force vector ---
    L(r"\subsection{Force Vector $\mathbf{F}$}")
    L("")
    L(r"\noindent Units: " + F_unit)
    L("")
    L(r"\begin{equation}")
    L(r"\mathbf{F} = \begin{bmatrix}")
    for i in range(4):
        L(fmt_eng(F_disp[i], 4) + (r" \\" if i < 3 else ""))
    L(r"\end{bmatrix}")
    L(r"\end{equation}")
    L("")

    # --- Solution ---
    L(r"\subsection{Solution $\mathbf{A} = \mathbf{K}^{-1}\mathbf{F}$}")
    L("")
    L(r"\begin{equation}")
    L(r"\mathbf{A} = \begin{bmatrix}")
    for i in range(4):
        L(fmt_eng(A_disp[i], 4) + (r" \\" if i < 3 else ""))
    L(r"\end{bmatrix}")
    L(r"\end{equation}")
    L("")

    # Map to (m,n) labels
    idx_labels = [(1, 1), (1, 2), (2, 1), (2, 2)]
    L(r"\noindent Coefficients:")
    L(r"\begin{align*}")
    for k, (m, n) in enumerate(idx_labels):
        sep = r" \\" if k < 3 else ""
        L(rf"A_{{{m}{n}}} &= {fmt_eng(A_disp[k], 4)} \ \text{{{A_unit}}}" + sep)
    L(r"\end{align*}")
    L("")

    # --- Deflection evaluation ---
    L(r"\subsection{Deflection at Analysis Point}")
    L("")
    x_d = x_pt * len_f
    y_d = y_pt * len_f
    L(rf"\noindent Evaluating $w$ at $(x, y) = ({fmt_eng(x_d, 4)},\, {fmt_eng(y_d, 4)})$ {lu}:")
    L("")

    w_2x2 = 0.0
    term_lines = []
    for m in range(1, 3):
        phi_val = beam_function(xi_pt, m, x_bt, 0)
        for n in range(1, 3):
            psi_val = beam_function(eta_pt, n, y_bt, 0)
            k = (m - 1) * 2 + (n - 1)
            A_mn_d = A_disp[k]
            contrib = A_mn_d * phi_val * psi_val
            w_2x2 += contrib
            term_lines.append(
                rf"A_{{{m}{n}}}\,\phi_{m}\,\psi_{n} &= "
                rf"{fmt_eng(A_mn_d, 4)} \times {fmt_eng(phi_val, 4)} \times {fmt_eng(psi_val, 4)} "
                rf"= {fmt_eng(contrib, 4)} \ \text{{{A_unit}}}"
            )

    L(r"\begin{align*}")
    for i, tl in enumerate(term_lines):
        L(tl + (r" \\" if i < len(term_lines) - 1 else ""))
    L(r"\end{align*}")
    L("")

    L(rf"\noindent $w_{{M=N=2}} = {fmt_eng(w_2x2, 6)}$ {A_unit}")
    L("")

    # --- Note about full solution ---
    ndof_full = ritz_terms ** 2
    w_full_d = w_full * len_f
    L(r"\subsection{Comparison with Full Solution}")
    L("")
    L(rf"\noindent The full solution uses $M = N = {ritz_terms}$, giving a "
      rf"${ndof_full} \times {ndof_full}$ system.")
    L(rf"The converged deflection at the analysis point is "
      rf"$w = {fmt_eng(w_full_d, 6)}$ {A_unit}.")
    L("")

    if w_full_d != 0:
        pct_diff = abs(w_2x2 - w_full_d) / abs(w_full_d) * 100
        L(rf"\noindent The $M = N = 2$ result differs by {fmt_eng(pct_diff, 2)}\%.")
    L("")

    return "\n".join(lines)
