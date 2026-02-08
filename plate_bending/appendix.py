"""
Appendix Template Engine for Plate Bending Reports
===================================================

Generates LaTeX source for a step-by-step worked appendix
showing the Ritz or Levy calculation with actual numbers.
"""

import numpy as np
from typing import Dict, List, Optional


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
