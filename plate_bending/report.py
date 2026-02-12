"""
Point Report Generator for Plate Bending Solvers
================================================

Generates a LaTeX calculation sheet for a specific (x, y) point.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.fit_solver import FITSolver
from plate_bending.solvers.ritz_solver import RitzSolver
from plate_bending.validation.benchmarks import Benchmarks


@dataclass
class ReportInputs:
    a: float
    b: float
    h: float
    E: float
    nu: float
    bc: str
    load_type: str
    q0: float
    x: float
    y: float
    method: str
    units: str = "metric"  # "metric" or "imperial"
    # Load geometry (all in SI internally)
    x0: Optional[float] = None   # load center x (circular, point)
    y0: Optional[float] = None   # load center y (circular, point)
    R: Optional[float] = None    # radius (circular)
    x1: Optional[float] = None   # patch bounds
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    F_allowable: Optional[float] = None  # allowable stress in Pa (SI)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

UNIT_SYSTEMS = {
    "metric": {
        "length": ("m", 1.0),
        "pressure": ("Pa", 1.0),
        "stress": ("MPa", 1e-6),
        "moment": (r"N$\cdot$m/m", 1.0),
        "rigidity": (r"N$\cdot$m", 1.0),
        "force": ("N", 1.0),
    },
    "imperial": {
        "length": ("in", 1.0 / 0.0254),
        "pressure": ("psi", 1.0 / 6894.76),
        "stress": ("psi", 1.0 / 6894.76),
        "moment": (r"lbf$\cdot$in/in", 1.0 / 0.1129848),
        "rigidity": (r"lbf$\cdot$in", 1.0 / 0.1129848),
        "force": ("lbf", 1.0 / 4.44822),
    },
}


def _convert_unit(value: float, unit_type: str, system: str) -> Tuple[float, str]:
    """Convert a value to the specified unit system."""
    unit_label, factor = UNIT_SYSTEMS[system][unit_type]
    return value * factor, unit_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plate_rigidity(E: float, h: float, nu: float) -> float:
    return E * h**3 / (12.0 * (1.0 - nu**2))


def _validate_bc(bc: str) -> str:
    if bc is None:
        raise ValueError("Boundary condition string is required.")
    bc = bc.strip().upper()
    if len(bc) != 4:
        raise ValueError(f"Boundary condition must be 4 characters, got: {bc!r}")
    for c in bc:
        if c not in ("S", "C", "F"):
            raise ValueError(f"Invalid boundary condition character: {c}")
    return bc


def _grid_axes(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X[0, :], Y[:, 0]


def _bilinear_sample(x: float, y: float, X: np.ndarray, Y: np.ndarray, F: np.ndarray) -> float:
    xg, yg = _grid_axes(X, Y)
    if x < xg[0] or x > xg[-1] or y < yg[0] or y > yg[-1]:
        raise ValueError(f"Point ({x}, {y}) out of bounds.")

    i = int(np.clip(np.searchsorted(xg, x) - 1, 0, len(xg) - 2))
    j = int(np.clip(np.searchsorted(yg, y) - 1, 0, len(yg) - 2))

    x1, x2 = xg[i], xg[i + 1]
    y1, y2 = yg[j], yg[j + 1]

    f11, f21 = F[j, i], F[j, i + 1]
    f12, f22 = F[j + 1, i], F[j + 1, i + 1]

    if x2 == x1 or y2 == y1:
        return float(f11)

    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)
    return float((1-tx)*(1-ty)*f11 + tx*(1-ty)*f21 + (1-tx)*ty*f12 + tx*ty*f22)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _solve_case(inputs: ReportInputs, n_terms: int = 100, ritz_terms: int = 10) -> Dict:
    bc = _validate_bc(inputs.bc)
    method = inputs.method.lower()

    load_kw = dict(
        load_type=inputs.load_type, q0=inputs.q0,
        x0=inputs.x0, y0=inputs.y0, R=inputs.R,
        x1=inputs.x1, y1=inputs.y1, x2=inputs.x2, y2=inputs.y2,
    )

    if method == "levy":
        if bc[0] != "S" or bc[2] != "S":
            raise ValueError("Levy solver requires simply supported x-edges.")
        solver = StableLevySolver(inputs.a, inputs.b, inputs.h, inputs.E, inputs.nu,
                                  bc_y0=bc[1], bc_yb=bc[3], n_terms=n_terms)
        results = solver.solve(**load_kw)
    elif method == "fit":
        solver = FITSolver(inputs.a, inputs.b, inputs.h, inputs.E, inputs.nu, bc, n_terms)
        results = solver.solve(**load_kw, auto_converge=False)
    elif method == "ritz":
        solver = RitzSolver(inputs.a, inputs.b, inputs.h, inputs.E, inputs.nu, bc,
                            M=ritz_terms, N=ritz_terms)
        results = solver.solve(**load_kw)
    else:
        raise ValueError(f"Unknown method: {inputs.method}")

    return results


def _point_results(results: Dict, x: float, y: float) -> Dict[str, float]:
    X, Y = results["X"], results["Y"]
    return {
        "W": _bilinear_sample(x, y, X, Y, results["W"]),
        "Mx": _bilinear_sample(x, y, X, Y, results["Mx"]),
        "My": _bilinear_sample(x, y, X, Y, results["My"]),
        "sigma_x": _bilinear_sample(x, y, X, Y, results["sigma_x"]),
        "sigma_y": _bilinear_sample(x, y, X, Y, results["sigma_y"]),
    }


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def _build_convergence_table(inputs: ReportInputs, base_n_terms: int,
                             ritz_terms: int) -> List[Tuple[str, float, float]]:
    method = inputs.method.lower()
    entries: List[Tuple[str, float, float]] = []

    if method in ("levy", "fit"):
        levels = [5, 10, 20, 40, 60, base_n_terms]
        levels = sorted(set(n for n in levels if n <= base_n_terms))
        prev = None
        for n in levels:
            res = _solve_case(inputs, n_terms=n, ritz_terms=ritz_terms)
            val = _point_results(res, inputs.x, inputs.y)["W"]
            delta = abs(val - prev) if prev is not None else np.nan
            entries.append((f"$N={n}$", val, delta))
            prev = val
    elif method == "ritz":
        levels = sorted(set(n for n in [2, 4, 6, 8, ritz_terms] if n <= ritz_terms))
        prev = None
        for n in levels:
            res = _solve_case(inputs, n_terms=base_n_terms, ritz_terms=n)
            val = _point_results(res, inputs.x, inputs.y)["W"]
            delta = abs(val - prev) if prev is not None else np.nan
            entries.append((f"$M=N={n}$", val, delta))
            prev = val

    return entries


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def _fmt(value: float, sig: int = 4) -> str:
    """Format number for LaTeX with *sig* significant figures.

    - 0             → "0"
    - 0.01 … 9999   → fixed-point
    - outside that   → engineering notation (exponent ÷ 3, mantissa ∈ [1,1000))
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    abs_val = abs(value)
    if abs_val == 0:
        return "0"

    exp = int(np.floor(np.log10(abs_val)))

    if -2 <= exp <= 3:
        dp = max(sig - exp - 1, 0)
        s = f"{value:.{dp}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    eng_exp = exp - (exp % 3) if exp >= 0 else exp - ((exp % 3 + 3) % 3)
    mantissa = value / 10**eng_exp
    m_exp = int(np.floor(np.log10(abs(mantissa))))
    dec = max(sig - m_exp - 1, 0)
    m_str = f"{mantissa:.{dec}f}"
    if '.' in m_str:
        m_str = m_str.rstrip('0').rstrip('.')
    return rf"{m_str} \times 10^{{{eng_exp}}}"


def _fmt_plain(value: float, sig: int = 4) -> str:
    """Format number for plain text contexts."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    abs_val = abs(value)
    if abs_val == 0:
        return "0"
    exp = int(np.floor(np.log10(abs_val)))
    if -2 <= exp <= 3:
        dp = max(sig - exp - 1, 0)
        s = f"{value:.{dp}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s
    return f"{value:.{sig-1}e}"


# ---------------------------------------------------------------------------
# LaTeX sections
# ---------------------------------------------------------------------------

def _assumptions_section() -> List[str]:
    """Return LaTeX lines for Assumptions & Limitations section."""
    return [
        r"\subsection{Assumptions \& Limitations}",
        r"\begin{itemize}",
        r"\item \textbf{Kirchhoff thin plate theory} --- plate thickness is small "
        r"compared to in-plane dimensions ($h/a < 0.1$); transverse shear "
        r"deformation is neglected.",
        r"\item \textbf{Small deflections} --- maximum deflection is small "
        r"compared to the plate thickness ($w \ll h$); geometric nonlinearity "
        r"is neglected.",
        r"\item \textbf{Linear elastic, isotropic, homogeneous material} --- "
        r"the constitutive relation is $\sigma = E\varepsilon$ with constant $E$ "
        r"and $\nu$ throughout.",
        r"\item \textbf{No membrane forces} --- in-plane loads and mid-surface "
        r"stretching are not considered.",
        r"\item \textbf{No thermal effects} --- temperature is uniform and "
        r"constant.",
        r"\item \textbf{Sign convention:} Positive deflection $w$ is downward "
        r"(in the direction of applied load). Positive bending moments $M_x$, "
        r"$M_y$ produce tension on the \emph{bottom} face ($z = +h/2$).",
        r"\end{itemize}",
        "",
    ]


def _load_expansion_section(inputs: ReportInputs, q0_disp: float, press_unit: str,
                            len_unit: str, units: str, n_terms: int) -> List[str]:
    """Return LaTeX lines showing Fourier load expansion with numbers."""
    lt = inputs.load_type.lower()
    lines: List[str] = [
        r"\section{Fourier Load Expansion}",
        r"The load $q(x,y)$ is expanded in a Fourier sine series in $x$:",
        r"\begin{equation}",
        r"q(x,y) = \sum_{m=1}^{\infty} q_m(y)\,\sin\!\left(\frac{m\pi x}{a}\right)",
        r"\end{equation}",
        "",
    ]

    a = inputs.a
    a_disp, _ = _convert_unit(a, "length", units)

    if lt == "uniform":
        lines += [
            r"\noindent For uniform pressure $q_0$, the Fourier coefficients are:",
            r"\begin{equation}",
            r"q_m = \frac{2}{a}\int_0^a q_0 \sin\!\left(\frac{m\pi x}{a}\right)dx "
            r"= \begin{cases} \dfrac{4q_0}{m\pi} & m\ \text{odd} \\[6pt] 0 & m\ \text{even} \end{cases}",
            r"\end{equation}",
            "",
            r"\noindent Substituting $q_0 = " + _fmt(q0_disp, 4) + r"$ " + press_unit + ":",
            "",
        ]
        for m in (1, 3, 5):
            qm_si = 4.0 * inputs.q0 / (m * np.pi)
            qm_disp, _ = _convert_unit(qm_si, "pressure", units)
            lines.append(
                rf"\noindent $q_{{{m}}} = \dfrac{{4 \times {_fmt(q0_disp, 4)}}}"
                rf"{{{m}\pi}} = {_fmt(qm_disp, 4)}$ {press_unit}"
            )
            lines.append("")

    elif lt == "rect_patch":
        x1_d, _ = _convert_unit(inputs.x1, "length", units)
        x2_d, _ = _convert_unit(inputs.x2, "length", units)
        lines += [
            r"\noindent For a rectangular patch load between $x_1$ and $x_2$:",
            r"\begin{equation}",
            r"q_m = \frac{2q_0}{m\pi}\left[\cos\frac{m\pi x_1}{a} - \cos\frac{m\pi x_2}{a}\right]",
            r"\end{equation}",
            "",
            rf"\noindent With $q_0 = {_fmt(q0_disp, 4)}$ {press_unit}, "
            rf"$x_1 = {_fmt(x1_d, 4)}$ {len_unit}, $x_2 = {_fmt(x2_d, 4)}$ {len_unit}:",
            "",
        ]
        for m in (1, 3):
            cos1 = np.cos(m * np.pi * inputs.x1 / a)
            cos2 = np.cos(m * np.pi * inputs.x2 / a)
            qm_si = 2.0 * inputs.q0 / (m * np.pi) * (cos1 - cos2)
            qm_disp, _ = _convert_unit(qm_si, "pressure", units)
            lines.append(
                rf"\noindent $q_{{{m}}} = \dfrac{{2 \times {_fmt(q0_disp, 4)}}}"
                rf"{{{m}\pi}}\left[\cos\dfrac{{{m}\pi \times {_fmt(x1_d, 4)}}}"
                rf"{{{_fmt(a_disp, 4)}}} - \cos\dfrac{{{m}\pi \times {_fmt(x2_d, 4)}}}"
                rf"{{{_fmt(a_disp, 4)}}}\right] = {_fmt(qm_disp, 4)}$ {press_unit}"
            )
            lines.append("")

    elif lt == "circular":
        lines += [
            r"\noindent The circular patch is decomposed into thin rectangular strips "
            r"parallel to the $x$-axis. For each strip $j$ of width $\Delta y_j$ "
            r"at ordinate $y_j$, the chord half-width $\ell_j$ is determined from "
            r"the circle geometry, and the corresponding Fourier coefficient is "
            r"computed as for a rectangular patch of width $2\ell_j$.",
            "",
            r"\noindent The resulting strip contributions are superposed. "
            r"Because strip widths vary with $y$, the coefficients $q_m(y)$ are "
            r"computed numerically for each mode $m$.",
            "",
        ]

    elif lt == "point":
        x0_d, _ = _convert_unit(inputs.x0, "length", units)
        force_d, force_unit = _convert_unit(inputs.q0, "force", units)
        lines += [
            r"\noindent For a concentrated force $P$ at $x = x_0$:",
            r"\begin{equation}",
            r"q_m = \frac{2P}{a}\sin\frac{m\pi x_0}{a}",
            r"\end{equation}",
            "",
            rf"\noindent With $P = {_fmt(force_d, 4)}$ {force_unit}, "
            rf"$x_0 = {_fmt(x0_d, 4)}$ {len_unit}:",
            "",
        ]
        for m in (1, 3):
            qm_si = 2.0 * inputs.q0 / a * np.sin(m * np.pi * inputs.x0 / a)
            qm_disp, qm_unit = _convert_unit(qm_si, "force", units)
            # qm has units of force/length
            qm_per_len = qm_disp / (_convert_unit(a, "length", units)[0])
            len_u = _convert_unit(1.0, "length", units)[1]
            lines.append(
                rf"\noindent $q_{{{m}}} = \dfrac{{2 \times {_fmt(force_d, 4)}}}"
                rf"{{{_fmt(a_disp, 4)}}}\sin\dfrac{{{m}\pi \times {_fmt(x0_d, 4)}}}"
                rf"{{{_fmt(a_disp, 4)}}} = {_fmt(qm_disp / a_disp, 4)}$ {force_unit}/{len_u}"
            )
            lines.append("")

    return lines


def _levy_worked_example(inputs: ReportInputs, D: float, q0_disp: float,
                         D_disp: float, a_disp: float, b_disp: float,
                         len_unit: str, press_unit: str, rig_unit: str,
                         units: str, point: Dict[str, float],
                         n_terms: int) -> List[str]:
    """Return LaTeX lines showing the m=1 Levy term worked out."""
    a = inputs.a
    b = inputs.b
    lt = inputs.load_type.lower()

    alpha1 = np.pi / a
    alpha1_disp = np.pi / a_disp  # 1/len_unit

    # Compute q1
    if lt == "uniform":
        q1 = 4.0 * inputs.q0 / np.pi
    elif lt == "point":
        x0 = inputs.x0 if inputs.x0 is not None else a / 2.0
        q1 = 2.0 * inputs.q0 / a * np.sin(np.pi * x0 / a)
    elif lt == "rect_patch":
        x1 = inputs.x1 if inputs.x1 is not None else 0.4 * a
        x2 = inputs.x2 if inputs.x2 is not None else 0.6 * a
        q1 = 2.0 * inputs.q0 / np.pi * (np.cos(np.pi * x1 / a) - np.cos(np.pi * x2 / a))
    else:
        # circular or other — skip worked example
        return []

    q1_disp, _ = _convert_unit(q1, "pressure", units)
    Yp = q1 / (D * alpha1**4)
    Yp_disp, _ = _convert_unit(Yp, "length", units)

    # Get Y1(y) from the solver by running a 1-term solve
    solver = StableLevySolver(a, b, inputs.h, inputs.E, inputs.nu,
                              bc_y0=inputs.bc[1], bc_yb=inputs.bc[3], n_terms=1)
    load_kw = dict(load_type=inputs.load_type, q0=inputs.q0,
                   x0=inputs.x0, y0=inputs.y0, R=inputs.R,
                   x1=inputs.x1, y1=inputs.y1, x2=inputs.x2, y2=inputs.y2)
    res1 = solver.solve(**load_kw)
    w1_at_point = _bilinear_sample(inputs.x, inputs.y, res1["X"], res1["Y"], res1["W"])
    w1_disp, _ = _convert_unit(w1_at_point, "length", units)

    # Y1 at analysis point: w1 = sin(π x/a) · Y1(y), so Y1 = w1/sin(πx/a)
    sin_val = np.sin(np.pi * inputs.x / a)
    if abs(sin_val) > 1e-12:
        Y1_at_y = w1_at_point / sin_val
    else:
        Y1_at_y = 0.0
    Y1_disp, _ = _convert_unit(Y1_at_y, "length", units)

    x_disp, _ = _convert_unit(inputs.x, "length", units)
    y_disp, _ = _convert_unit(inputs.y, "length", units)

    lines = [
        r"\subsection{Worked Example: $m = 1$ Term}",
        "",
        rf"\noindent Wave number: $\alpha_1 = \pi / a = \pi / {_fmt(a_disp, 4)} "
        rf"= {_fmt(alpha1_disp, 4)}$ {len_unit}$^{{-1}}$",
        "",
        r"\noindent The ODE for $Y_1(y)$ becomes:",
        r"\begin{equation}",
        rf"D\!\left(Y_1'''' - 2\alpha_1^2 Y_1'' + \alpha_1^4 Y_1\right) = q_1, "
        rf"\qquad D = {_fmt(D_disp, 4)}\ \text{{{rig_unit}}},\quad "
        rf"q_1 = {_fmt(q1_disp, 4)}\ \text{{{press_unit}}}",
        r"\end{equation}",
        "",
        r"\noindent \textbf{Particular solution} (constant load on RHS):",
        r"\begin{equation}",
        rf"Y_p = \frac{{q_1}}{{D\,\alpha_1^4}} = "
        rf"\frac{{{_fmt(q1_disp, 4)}}}{{{_fmt(D_disp, 4)} \times {_fmt(alpha1_disp, 4)}^4}} "
        rf"= {_fmt(Yp_disp, 4)}\ \text{{{len_unit}}}",
        r"\end{equation}",
        "",
        r"\noindent \textbf{Homogeneous solution:}",
        r"\begin{equation}",
        r"Y_h = (C_1 + C_2\,\alpha_1 y)\cosh(\alpha_1 y) "
        r"+ (C_3 + C_4\,\alpha_1 y)\sinh(\alpha_1 y)",
        r"\end{equation}",
        "",
        rf"\noindent The constants $C_1$--$C_4$ are determined by the boundary "
        rf"conditions at $y = 0$ ({inputs.bc[1]}) and $y = b$ ({inputs.bc[3]}).",
        "",
        rf"\noindent Evaluating at the analysis point "
        rf"$(x, y) = ({_fmt(x_disp, 4)},\, {_fmt(y_disp, 4)})$ {len_unit}:",
        r"\begin{align*}",
        rf"Y_1({_fmt(y_disp, 4)}) &= {_fmt(Y1_disp, 4)}\ \text{{{len_unit}}} \\",
        rf"w_1 &= \sin\!\left(\frac{{\pi \times {_fmt(x_disp, 4)}}}"
        rf"{{{_fmt(a_disp, 4)}}}\right) \times {_fmt(Y1_disp, 4)} "
        rf"= {_fmt(w1_disp, 6)}\ \text{{{len_unit}}}",
        r"\end{align*}",
        "",
        rf"\noindent The full solution sums $N = {n_terms}$ such terms.",
        "",
    ]

    return lines


def _moment_derivation_section(inputs: ReportInputs, point: Dict[str, float],
                               h_disp: float, len_unit: str, mom_unit: str,
                               stress_unit: str, units: str) -> List[str]:
    """Return LaTeX lines deriving moments and stresses with numbers."""
    method = inputs.method.lower()

    Mx_disp, _ = _convert_unit(point["Mx"], "moment", units)
    My_disp, _ = _convert_unit(point["My"], "moment", units)
    sigma_x_disp, _ = _convert_unit(point["sigma_x"], "stress", units)
    sigma_y_disp, _ = _convert_unit(point["sigma_y"], "stress", units)

    a_disp, _ = _convert_unit(inputs.a, "length", units)
    x_disp, _ = _convert_unit(inputs.x, "length", units)
    y_disp, _ = _convert_unit(inputs.y, "length", units)

    lines = [
        r"\subsection{Moment and Stress Derivation}",
        r"\noindent Bending moments are obtained from the deflection surface:",
        r"\begin{align}",
        r"M_x &= -D\!\left(\frac{\partial^2 w}{\partial x^2} "
        r"+ \nu\,\frac{\partial^2 w}{\partial y^2}\right) \\",
        r"M_y &= -D\!\left(\frac{\partial^2 w}{\partial y^2} "
        r"+ \nu\,\frac{\partial^2 w}{\partial x^2}\right)",
        r"\end{align}",
        "",
    ]

    if method == "levy":
        lines += [
            r"\noindent For the L\'evy solution $w = \sum \sin(m\pi x/a)\,Y_m(y)$, "
            r"the second derivatives are:",
            r"\begin{align*}",
            r"\frac{\partial^2 w}{\partial x^2} &= "
            r"-\sum_{m=1}^{N}\left(\frac{m\pi}{a}\right)^{\!2}"
            r"\sin\!\left(\frac{m\pi x}{a}\right)Y_m(y) \\",
            r"\frac{\partial^2 w}{\partial y^2} &= "
            r"\sum_{m=1}^{N}\sin\!\left(\frac{m\pi x}{a}\right)Y_m''(y)",
            r"\end{align*}",
            "",
        ]

    lines += [
        rf"\noindent Evaluating at $(x, y) = ({_fmt(x_disp, 4)},\, {_fmt(y_disp, 4)})$ "
        rf"{len_unit}:",
        "",
        rf"\noindent $M_x = {_fmt(Mx_disp, 6)}$ {mom_unit}",
        "",
        rf"\noindent $M_y = {_fmt(My_disp, 6)}$ {mom_unit}",
        "",
        r"\bigskip",
        r"\noindent \textbf{Bending stresses} at the plate surfaces ($z = \pm h/2$):",
        r"\begin{equation}",
        r"\sigma_x = \frac{6\,M_x}{h^2}, \qquad \sigma_y = \frac{6\,M_y}{h^2}",
        r"\end{equation}",
        "",
        r"\noindent Substituting values step by step:",
        r"\begin{align}",
        rf"\sigma_x &= \frac{{6 \times {_fmt(Mx_disp, 4)}}}{{{_fmt(h_disp, 4)}^2}} "
        rf"= \frac{{{_fmt(6.0 * Mx_disp, 4)}}}{{{_fmt(h_disp**2, 4)}}} "
        rf"= {_fmt(sigma_x_disp, 6)} \ \text{{{stress_unit}}} \\",
        rf"\sigma_y &= \frac{{6 \times {_fmt(My_disp, 4)}}}{{{_fmt(h_disp, 4)}^2}} "
        rf"= \frac{{{_fmt(6.0 * My_disp, 4)}}}{{{_fmt(h_disp**2, 4)}}} "
        rf"= {_fmt(sigma_y_disp, 6)} \ \text{{{stress_unit}}}",
        r"\end{align}",
        "",
    ]

    return lines


def _loading_description(inputs: ReportInputs, q0_disp: float, press_unit: str,
                         len_unit: str, units: str) -> List[str]:
    """Return LaTeX lines describing the applied loading."""
    lt = inputs.load_type.lower()
    lines: List[str] = []

    if lt == "uniform":
        lines.append(rf"Uniformly distributed pressure $q_0 = {_fmt(q0_disp, 4)}$ {press_unit} "
                      r"applied over the entire plate surface.")
    elif lt == "circular":
        x0_d, _ = _convert_unit(inputs.x0, "length", units)
        y0_d, _ = _convert_unit(inputs.y0, "length", units)
        R_d, _ = _convert_unit(inputs.R, "length", units)
        lines.append(rf"Uniform pressure $q_0 = {_fmt(q0_disp, 4)}$ {press_unit} applied over a "
                      rf"circular area of radius $R = {_fmt(R_d, 4)}$ {len_unit} "
                      rf"centered at $(x_0, y_0) = ({_fmt(x0_d, 4)},\, {_fmt(y0_d, 4)})$ {len_unit}.")
        lines.append("")
        lines.append(r"The circular patch is approximated by a superposition of thin rectangular "
                      r"strips, each solved independently via the Levy ODE with piecewise "
                      r"particular solutions.")
    elif lt == "rect_patch":
        x1_d, _ = _convert_unit(inputs.x1, "length", units)
        y1_d, _ = _convert_unit(inputs.y1, "length", units)
        x2_d, _ = _convert_unit(inputs.x2, "length", units)
        y2_d, _ = _convert_unit(inputs.y2, "length", units)
        lines.append(rf"Uniform pressure $q_0 = {_fmt(q0_disp, 4)}$ {press_unit} applied over the "
                      rf"rectangular patch $[{_fmt(x1_d,4)},\,{_fmt(x2_d,4)}] \times "
                      rf"[{_fmt(y1_d,4)},\,{_fmt(y2_d,4)}]$ {len_unit}.")
    elif lt == "point":
        x0_d, _ = _convert_unit(inputs.x0, "length", units)
        y0_d, _ = _convert_unit(inputs.y0, "length", units)
        force_d, force_unit = _convert_unit(inputs.q0, "force", units)
        lines.append(rf"Concentrated force $P = {_fmt(force_d, 4)}$ {force_unit} applied at "
                      rf"$(x_0, y_0) = ({_fmt(x0_d, 4)},\, {_fmt(y0_d, 4)})$ {len_unit}.")
    else:
        lines.append(rf"Load type: \textbf{{{inputs.load_type}}}, $q_0 = {_fmt(q0_disp, 4)}$ {press_unit}")

    return lines


def _method_equations(inputs: ReportInputs, D_disp: float, n_terms: int, ritz_terms: int,
                      a_disp: float, b_disp: float, q0_disp: float,
                      len_unit: str, press_unit: str, rig_unit: str) -> str:
    method = inputs.method.lower()
    lt = inputs.load_type.lower()

    D_str = _fmt(D_disp, 4)
    q0_str = _fmt(q0_disp, 4)
    a_str = _fmt(a_disp, 4)
    b_str = _fmt(b_disp, 4)

    # Fourier coefficient description depends on load type
    if lt == "uniform":
        qm_line = r"q_m = \frac{4 q_0}{m\pi}\ \text{(odd $m$ only)}"
    elif lt == "circular":
        qm_line = (r"q_m\ \text{from strip superposition of the circular patch "
                    r"(see Loading section)}")
    elif lt == "rect_patch":
        qm_line = (r"q_m = \frac{2q_0}{m\pi}\left[\cos\frac{m\pi x_1}{a} - "
                    r"\cos\frac{m\pi x_2}{a}\right]")
    elif lt == "point":
        qm_line = r"q_m = \frac{2P}{a}\sin\frac{m\pi x_0}{a}"
    else:
        qm_line = r"q_m\ \text{(load-dependent)}"

    if method == "levy":
        return (
            r"\begin{align*}"
            r"w(x,y) &= \sum_{m=1}^{N} \sin\!\left(\frac{m\pi x}{a}\right) Y_m(y), \quad "
            r"\alpha_m = \frac{m\pi}{a} \\"
            r"D \left(Y_m'''' - 2\alpha_m^2 Y_m'' + \alpha_m^4 Y_m\right) &= q_m \\"
            rf"& {qm_line} \\"
            rf"a &= {a_str}\ \text{{{len_unit}}},\quad D = {D_str}\ \text{{{rig_unit}}},\quad "
            rf"q_0 = {q0_str}\ \text{{{press_unit}}},\quad N = {n_terms}"
            r"\end{align*}"
        )
    if method == "fit":
        return (
            r"\begin{align*}"
            r"w(x,y) &= \sum_{m=1}^{N}\sum_{n=1}^{N} W_{mn}\,"
            r"\sin\!\left(\frac{m\pi x}{a}\right)\sin\!\left(\frac{n\pi y}{b}\right) \\"
            r"W_{mn} &= \frac{q_{mn}}{D\!\left[\left(\frac{m\pi}{a}\right)^2+"
            r"\left(\frac{n\pi}{b}\right)^2\right]^2} \\"
            rf"a &= {a_str}\ \text{{{len_unit}}},\quad b = {b_str}\ \text{{{len_unit}}},\quad "
            rf"D = {D_str}\ \text{{{rig_unit}}},\quad N = {n_terms}"
            r"\end{align*}"
        )
    if method == "ritz":
        return (
            r"\begin{align*}"
            r"w(x,y) &= \sum_{m=1}^{M}\sum_{n=1}^{N} A_{mn}\,\phi_m(x)\,\psi_n(y) \\"
            r"\mathbf{K}\mathbf{A} &= \mathbf{F} \\"
            rf"a &= {a_str}\ \text{{{len_unit}}},\quad b = {b_str}\ \text{{{len_unit}}},\quad "
            rf"D = {D_str}\ \text{{{rig_unit}}},\quad \nu = {inputs.nu:.4g},\quad M=N={ritz_terms}"
            r"\end{align*}"
        )
    return ""


def _convergence_narrative(entries: List[Tuple[str, float, float]],
                           len_conv: float, len_unit: str,
                           method: str) -> List[str]:
    """Return LaTeX lines narrating the first few convergence steps."""
    lines: List[str] = []
    if len(entries) < 2:
        return lines

    lines.append(r"\noindent\textbf{Procedure:} The series is evaluated at increasing truncation "
                  r"levels.  At each level the deflection at the analysis point is recorded and "
                  r"compared to the previous level.  Convergence is achieved when $|\Delta w|$ "
                  r"becomes negligible.")
    lines.append("")

    # Show the first two steps worked out
    for idx in range(min(2, len(entries))):
        label, val_si, delta_si = entries[idx]
        val_d = val_si * len_conv
        if idx == 0:
            lines.append(rf"\noindent {label}: solve $\rightarrow$ "
                          rf"$w = {_fmt(val_d, 6)}$ {len_unit} (first evaluation, no comparison).")
        else:
            prev_d = entries[idx - 1][1] * len_conv
            delta_d = delta_si * len_conv if not np.isnan(delta_si) else 0.0
            lines.append(rf"\noindent {label}: solve $\rightarrow$ "
                          rf"$w = {_fmt(val_d, 6)}$ {len_unit}.\quad "
                          rf"$|\Delta w| = |{_fmt(val_d, 6)} - {_fmt(prev_d, 6)}| = "
                          rf"{_fmt(delta_d, 3)}$ {len_unit}.")
        lines.append("")

    if len(entries) > 2:
        last_val_d = entries[-1][1] * len_conv
        last_delta_d = entries[-1][2] * len_conv if not np.isnan(entries[-1][2]) else 0.0
        lines.append(rf"\noindent Continuing to {entries[-1][0]}: "
                      rf"$w = {_fmt(last_val_d, 6)}$ {len_unit}, "
                      rf"$|\Delta w| = {_fmt(last_delta_d, 3)}$ {len_unit} "
                      r"--- series has converged.")
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Peak stress & margin of safety sections
# ---------------------------------------------------------------------------

def _peak_stress_table(results: Dict, h_disp: float, stress_unit: str,
                       len_unit: str, units: str) -> List[str]:
    """Return LaTeX lines for the Peak Stress Summary section."""
    X, Y = results["X"], results["Y"]
    sigma_x = results["sigma_x"]
    sigma_y = results["sigma_y"]
    W = results["W"]

    def _find_peak(arr: np.ndarray) -> Tuple[float, float, float]:
        abs_arr = np.abs(arr)
        idx = np.unravel_index(np.argmax(abs_arr), abs_arr.shape)
        val = float(arr[idx])
        px = float(X[idx])
        py = float(Y[idx])
        return val, px, py

    def _region(px: float, py: float, a: float, b: float) -> str:
        labels = []
        if py < 0.15 * b:
            labels.append("$y{=}0$ edge")
        elif py > 0.85 * b:
            labels.append("$y{=}b$ edge")
        if px < 0.15 * a:
            labels.append("$x{=}0$ edge")
        elif px > 0.85 * a:
            labels.append("$x{=}a$ edge")
        if not labels:
            labels.append("interior")
        return ", ".join(labels)

    a_si = float(X[0, -1])
    b_si = float(Y[-1, 0])

    sx_val, sx_x, sx_y = _find_peak(sigma_x)
    sy_val, sy_x, sy_y = _find_peak(sigma_y)
    w_val, w_x, w_y = _find_peak(W)

    _, stress_conv = UNIT_SYSTEMS[units]["stress"]
    _, len_conv = UNIT_SYSTEMS[units]["length"]

    lines = [
        r"\subsection{Peak Stress Summary}",
        r"\begin{tabular}{llrl}",
        r"\toprule",
        rf"Quantity & Value ({stress_unit} / {len_unit}) & Location $({len_unit})$ & Region \\",
        r"\midrule",
        rf"Max $|\sigma_x|$ & ${_fmt(sx_val * stress_conv, 4)}$ {stress_unit} "
        rf"& $({_fmt(sx_x * len_conv, 4)},\, {_fmt(sx_y * len_conv, 4)})$ "
        rf"& {_region(sx_x, sx_y, a_si, b_si)} \\",
        rf"Max $|\sigma_y|$ & ${_fmt(sy_val * stress_conv, 4)}$ {stress_unit} "
        rf"& $({_fmt(sy_x * len_conv, 4)},\, {_fmt(sy_y * len_conv, 4)})$ "
        rf"& {_region(sy_x, sy_y, a_si, b_si)} \\",
        rf"Max $|w|$ & ${_fmt(w_val * len_conv, 6)}$ {len_unit} "
        rf"& $({_fmt(w_x * len_conv, 4)},\, {_fmt(w_y * len_conv, 4)})$ "
        rf"& {_region(w_x, w_y, a_si, b_si)} \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return lines


def _margin_of_safety_section(inputs: ReportInputs, point: Dict[str, float],
                               results: Dict, h_disp: float,
                               stress_unit: str, units: str) -> List[str]:
    """Return LaTeX lines for the Margin of Safety section."""
    if inputs.F_allowable is None:
        return []

    F_allow = inputs.F_allowable
    _, stress_conv = UNIT_SYSTEMS[units]["stress"]
    F_allow_disp = F_allow * stress_conv

    # Stresses at analysis point
    sx_pt = abs(point["sigma_x"])
    sy_pt = abs(point["sigma_y"])

    # Peak stresses from full field
    sx_peak = float(np.max(np.abs(results["sigma_x"])))
    sy_peak = float(np.max(np.abs(results["sigma_y"])))

    def _ms(allow: float, actual: float) -> float:
        if actual == 0:
            return float('inf')
        if allow == 0:
            return -1.0
        return allow / actual - 1.0

    ms_sx_pt = _ms(F_allow, sx_pt)
    ms_sy_pt = _ms(F_allow, sy_pt)
    ms_sx_pk = _ms(F_allow, sx_peak)
    ms_sy_pk = _ms(F_allow, sy_peak)

    all_ms = [ms_sx_pt, ms_sy_pt, ms_sx_pk, ms_sy_pk]
    governing = min(all_ms)

    if governing >= 0:
        verdict = rf"\textcolor{{green!50!black}}{{PASSES}}"
        sign = "+"
    else:
        verdict = rf"\textcolor{{red}}{{FAILS}}"
        sign = ""

    def _ms_fmt(ms: float) -> str:
        if ms == float('inf'):
            return r"$\infty$"
        return f"${'+' if ms >= 0 else ''}{_fmt(ms, 3)}$"

    def _s_fmt(val: float) -> str:
        return f"${_fmt(val * stress_conv, 4)}$"

    f_str = f"${_fmt(F_allow_disp, 4)}$"

    lines = [
        r"\subsection{Margin of Safety}",
        r"The margin of safety is defined as:",
        r"\begin{equation}",
        r"\mathrm{MS} = \frac{F_{\mathrm{allow}}}{f_{\mathrm{actual}}} - 1",
        r"\end{equation}",
        "",
        rf"\noindent The plate {verdict} with governing "
        rf"$\mathrm{{MS}} = {sign}{_fmt(governing, 3)}$.",
        "",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        rf"Check & $\sigma$ ({stress_unit}) & $F_{{\mathrm{{allow}}}}$ ({stress_unit}) & MS \\",
        r"\midrule",
        rf"$\sigma_x$ at point & {_s_fmt(sx_pt)} & {f_str} & {_ms_fmt(ms_sx_pt)} \\",
        rf"$\sigma_y$ at point & {_s_fmt(sy_pt)} & {f_str} & {_ms_fmt(ms_sy_pt)} \\",
        rf"$\sigma_x$ peak & {_s_fmt(sx_peak)} & {f_str} & {_ms_fmt(ms_sx_pk)} \\",
        rf"$\sigma_y$ peak & {_s_fmt(sy_peak)} & {f_str} & {_ms_fmt(ms_sy_pk)} \\",
        r"\midrule",
        rf"\textbf{{Governing}} & & & \textbf{{{_ms_fmt(governing)}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_point_report(inputs: ReportInputs,
                          n_terms: int = 100,
                          ritz_terms: int = 10,
                          include_convergence: bool = True) -> Tuple[str, Dict]:
    bc = _validate_bc(inputs.bc)
    units = inputs.units if inputs.units in UNIT_SYSTEMS else "metric"

    # Rebuild inputs with validated BC (preserves load geometry fields)
    inputs = ReportInputs(
        a=inputs.a, b=inputs.b, h=inputs.h, E=inputs.E, nu=inputs.nu, bc=bc,
        load_type=inputs.load_type, q0=inputs.q0, x=inputs.x, y=inputs.y,
        method=inputs.method, units=units,
        x0=inputs.x0, y0=inputs.y0, R=inputs.R,
        x1=inputs.x1, y1=inputs.y1, x2=inputs.x2, y2=inputs.y2,
        F_allowable=inputs.F_allowable,
    )

    results = _solve_case(inputs, n_terms=n_terms, ritz_terms=ritz_terms)
    point = _point_results(results, inputs.x, inputs.y)

    D = _plate_rigidity(inputs.E, inputs.h, inputs.nu)
    q0 = inputs.q0 if inputs.q0 != 0 else 1.0

    W_coef = point["W"] * D / (q0 * inputs.a**4)
    Mx_coef = point["Mx"] / (q0 * inputs.a**2)
    My_coef = point["My"] / (q0 * inputs.a**2)

    convergence = _build_convergence_table(inputs, n_terms, ritz_terms) if include_convergence else []

    bench = Benchmarks.get(inputs.bc)
    bench_source = bench["source"] if bench else "N/A"

    # --- Unit conversions for display ---
    a_disp, len_unit = _convert_unit(inputs.a, "length", units)
    b_disp, _ = _convert_unit(inputs.b, "length", units)
    h_disp, _ = _convert_unit(inputs.h, "length", units)
    x_disp, _ = _convert_unit(inputs.x, "length", units)
    y_disp, _ = _convert_unit(inputs.y, "length", units)
    E_disp, press_unit = _convert_unit(inputs.E, "pressure", units)
    q0_disp, _ = _convert_unit(inputs.q0, "pressure", units)
    D_disp, rig_unit = _convert_unit(D, "rigidity", units)

    W_disp, _ = _convert_unit(point["W"], "length", units)
    Mx_disp, mom_unit = _convert_unit(point["Mx"], "moment", units)
    My_disp, _ = _convert_unit(point["My"], "moment", units)
    sigma_x_disp, stress_unit = _convert_unit(point["sigma_x"], "stress", units)
    sigma_y_disp, _ = _convert_unit(point["sigma_y"], "stress", units)

    bc_names = {'S': 'Simply Supported', 'C': 'Clamped', 'F': 'Free'}
    bc_desc = (f"$x=0$: {bc_names.get(bc[0],'?')}, $y=0$: {bc_names.get(bc[1],'?')}, "
               f"$x=a$: {bc_names.get(bc[2],'?')}, $y=b$: {bc_names.get(bc[3],'?')}")
    unit_note = "Imperial (in, psi, lbf)" if units == "imperial" else "Metric (m, Pa, N)"

    # ===== Build LaTeX =====
    latex_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{siunitx}",
        r"\usepackage{xcolor}",
        r"\geometry{margin=1in}",
        r"\title{Rectangular Plate Bending Analysis\\[0.5em]\large Point Calculation Report}",
        r"\author{Plate Bending Solver}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        "",
        rf"\noindent\textbf{{Unit System:}} {unit_note}",
        "",
        # --- Section 1: Input Parameters ---
        r"\section{Input Parameters}",
        r"\subsection{Plate Geometry \& Material}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Parameter & Symbol & Value \\",
        r"\midrule",
        rf"Plate length (x-direction) & $a$ & ${_fmt(a_disp, 4)}$ {len_unit} \\",
        rf"Plate width (y-direction) & $b$ & ${_fmt(b_disp, 4)}$ {len_unit} \\",
        rf"Thickness & $h$ & ${_fmt(h_disp, 4)}$ {len_unit} \\",
        rf"Young's modulus & $E$ & ${_fmt(E_disp, 4)}$ {press_unit} \\",
        rf"Poisson's ratio & $\nu$ & ${inputs.nu:.4g}$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\subsection{Flexural Rigidity}",
        r"\begin{equation}",
        rf"D = \frac{{Eh^3}}{{12(1-\nu^2)}} = {_fmt(D_disp, 4)} \ \text{{{rig_unit}}}",
        r"\end{equation}",
        "",
        r"\subsection{Boundary Conditions}",
        rf"Configuration: \textbf{{{inputs.bc}}} --- {bc_desc}",
        "",
        r"\subsection{Loading}",
    ]
    latex_lines += _loading_description(inputs, q0_disp, press_unit, len_unit, units)
    latex_lines += [""]
    latex_lines += _load_expansion_section(inputs, q0_disp, press_unit, len_unit, units, n_terms)
    latex_lines += [
        r"\subsection{Analysis Point}",
        rf"$(x, y) = ({_fmt(x_disp, 4)},\, {_fmt(y_disp, 4)})$ {len_unit}",
        "",
    ]
    latex_lines += _assumptions_section()
    latex_lines += [
        # --- Section 2: Governing Equation ---
        r"\section{Governing Equation}",
        r"The deflection $w(x,y)$ of a thin plate under transverse loading "
        r"satisfies the biharmonic equation:",
        r"\begin{equation}",
        r"D\nabla^4 w = D\!\left(\frac{\partial^4 w}{\partial x^4} "
        r"+ 2\frac{\partial^4 w}{\partial x^2 \partial y^2} "
        r"+ \frac{\partial^4 w}{\partial y^4}\right) = q(x,y)",
        r"\end{equation}",
        "",
        # --- Section 3: Solution Method ---
        r"\section{Solution Method}",
        rf"Method: \textbf{{{inputs.method.capitalize()}}}",
        "",
        _method_equations(inputs, D_disp, n_terms, ritz_terms,
                         a_disp, b_disp, q0_disp, len_unit, press_unit, rig_unit),
        "",
    ]
    if inputs.method.lower() == "levy":
        latex_lines += _levy_worked_example(inputs, D, q0_disp, D_disp, a_disp, b_disp,
                                            len_unit, press_unit, rig_unit, units, point, n_terms)
    latex_lines += [
        # --- Section 4: Convergence Study ---
        r"\section{Convergence Study}",
    ]

    if convergence:
        _, len_conv = UNIT_SYSTEMS[units]["length"]

        # Narrative showing worked steps
        latex_lines += _convergence_narrative(convergence, len_conv, len_unit,
                                              inputs.method.lower())

        # Summary table
        latex_lines += [
            r"\begin{tabular}{lrr}",
            r"\toprule",
            rf"Series Level & $w(x,y)$ [{len_unit}] & $|\Delta w|$ [{len_unit}] \\",
            r"\midrule",
        ]
        for label, val, delta in convergence:
            val_d = val * len_conv
            delta_d = delta * len_conv if not np.isnan(delta) else np.nan
            latex_lines.append(rf"{label} & ${_fmt(val_d, 6)}$ & ${_fmt(delta_d, 3)}$ \\")
        latex_lines += [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    else:
        latex_lines.append(r"No convergence table requested.")

    # --- Section 5: Results ---
    latex_lines += [
        "",
        r"\section{Results at Analysis Point}",
        rf"Results computed at $(x, y) = ({_fmt(x_disp, 4)},\, {_fmt(y_disp, 4)})$ {len_unit}:",
        "",
        r"\subsection{Deflection \& Moments}",
        r"\begin{tabular}{llrl}",
        r"\toprule",
        r"Quantity & Symbol & Value & Units \\",
        r"\midrule",
        rf"Deflection & $w$ & ${_fmt(W_disp, 6)}$ & {len_unit} \\",
        rf"Bending moment (x) & $M_x$ & ${_fmt(Mx_disp, 6)}$ & {mom_unit} \\",
        rf"Bending moment (y) & $M_y$ & ${_fmt(My_disp, 6)}$ & {mom_unit} \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    latex_lines += _moment_derivation_section(inputs, point, h_disp, len_unit,
                                              mom_unit, stress_unit, units)
    latex_lines += [
    ]

    # Peak Stress Summary (always)
    latex_lines += _peak_stress_table(results, h_disp, stress_unit, len_unit, units)

    # Margin of Safety (only if F_allowable set)
    latex_lines += _margin_of_safety_section(inputs, point, results, h_disp, stress_unit, units)

    latex_lines += [
        # --- Section 6: Non-Dimensional Coefficients ---
        r"\section{Non-Dimensional Coefficients}",
        r"For comparison with tabulated reference values "
        r"(e.g., Timoshenko \& Woinowsky-Krieger):",
        r"\begin{equation}",
        r"\bar{w} = \frac{w D}{q_0 a^4}, \qquad "
        r"\bar{M}_x = \frac{M_x}{q_0 a^2}, \qquad "
        r"\bar{M}_y = \frac{M_y}{q_0 a^2}",
        r"\end{equation}",
        r"\textit{(Non-dimensional coefficients are unit-system independent.)}",
        "",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Coefficient & Expression & Value \\",
        r"\midrule",
        rf"$\bar{{w}}$ & $w D / (q_0 a^4)$ & ${_fmt(W_coef, 6)}$ \\",
        rf"$\bar{{M}}_x$ & $M_x / (q_0 a^2)$ & ${_fmt(Mx_coef, 6)}$ \\",
        rf"$\bar{{M}}_y$ & $M_y / (q_0 a^2)$ & ${_fmt(My_coef, 6)}$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
        # --- Section 7: Reference ---
        r"\section{Reference Validation}",
        rf"Benchmark source for BC = {inputs.bc}: {bench_source}.",
        "",
        r"\section*{References}",
        r"\begin{enumerate}",
        r"\item Timoshenko, S. and Woinowsky-Krieger, S., "
        r"\emph{Theory of Plates and Shells}, 2nd ed., McGraw-Hill, 1959.",
        r"\item Szilard, R., \emph{Theories and Applications of Plate Analysis}, "
        r"John Wiley \& Sons, 2004.",
        r"\item Xu, R. et al., ``Analytical Bending Solutions of Orthotropic "
        r"Rectangular Thin Plates with Two Adjacent Edges Free,'' "
        r"\emph{Archives of Applied Mechanics}, 2020.",
        r"\end{enumerate}",
        r"\end{document}",
    ]

    report = "\n".join(latex_lines)
    data = {
        "point": point,
        "W_coef": W_coef,
        "Mx_coef": Mx_coef,
        "My_coef": My_coef,
        "D": D,
        "results": results,
    }
    return report, data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a point report for plate bending solvers.")
    p.add_argument("--bc", required=True,
                   help="Boundary conditions (e.g., SSSS, SCSF)")
    p.add_argument("--load", dest="load_type", required=True,
                   help="Load type: uniform, circular, rect_patch, point")
    p.add_argument("--q", dest="q0", type=float, required=True,
                   help="Load magnitude (pressure or force for point)")
    p.add_argument("--a", type=float, required=True, help="Plate length x")
    p.add_argument("--b", type=float, required=True, help="Plate width y")
    p.add_argument("--h", type=float, required=True, help="Plate thickness")
    p.add_argument("--E", dest="E", type=float, required=True,
                   help="Young's modulus")
    p.add_argument("--nu", type=float, required=True, help="Poisson's ratio")
    p.add_argument("--x", type=float, default=None,
                   help="Analysis point x (default: center)")
    p.add_argument("--y", type=float, default=None,
                   help="Analysis point y (default: center)")
    p.add_argument("--method", required=True, choices=["levy", "fit", "ritz"],
                   help="Solver method")
    p.add_argument("--units", choices=["metric", "imperial"], default="metric",
                   help="Unit system (default: metric)")
    p.add_argument("--output", required=True, help="Output .tex file path")
    p.add_argument("--compile", action="store_true",
                   help="Compile to PDF with tectonic")
    # Load geometry
    p.add_argument("--x0", type=float, default=None,
                   help="Load center x (circular, point)")
    p.add_argument("--y0", type=float, default=None,
                   help="Load center y (circular, point)")
    p.add_argument("--R", type=float, default=None,
                   help="Circular patch radius")
    p.add_argument("--allowable", type=float, default=None,
                   help="Allowable stress (psi for imperial, Pa for metric)")
    p.add_argument("--x1", type=float, default=None, help="Patch x-start")
    p.add_argument("--y1", type=float, default=None, help="Patch y-start")
    p.add_argument("--x2", type=float, default=None, help="Patch x-end")
    p.add_argument("--y2", type=float, default=None, help="Patch y-end")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    x = args.x if args.x is not None else args.a / 2.0
    y = args.y if args.y is not None else args.b / 2.0

    # For load geometry, default center to plate center if not given
    x0 = args.x0 if args.x0 is not None else args.a / 2.0
    y0 = args.y0 if args.y0 is not None else args.b / 2.0

    # Imperial → SI conversion
    if args.units == "imperial":
        to_m = 0.0254
        to_pa = 6894.76
        a = args.a * to_m
        b = args.b * to_m
        h = args.h * to_m
        E = args.E * to_pa
        q0 = args.q0 * to_pa
        x = x * to_m
        y = y * to_m
        x0 = x0 * to_m
        y0 = y0 * to_m
        R = args.R * to_m if args.R is not None else None
        x1 = args.x1 * to_m if args.x1 is not None else None
        y1 = args.y1 * to_m if args.y1 is not None else None
        x2 = args.x2 * to_m if args.x2 is not None else None
        y2 = args.y2 * to_m if args.y2 is not None else None
        F_allowable = args.allowable * to_pa if args.allowable is not None else None
    else:
        a, b, h, E, q0 = args.a, args.b, args.h, args.E, args.q0
        R = args.R
        x1, y1, x2, y2 = args.x1, args.y1, args.x2, args.y2
        F_allowable = args.allowable

    inputs = ReportInputs(
        a=a, b=b, h=h, E=E, nu=args.nu,
        bc=args.bc, load_type=args.load_type, q0=q0,
        x=x, y=y, method=args.method, units=args.units,
        x0=x0, y0=y0, R=R,
        x1=x1, y1=y1, x2=x2, y2=y2,
        F_allowable=F_allowable,
    )

    report, _ = generate_point_report(inputs)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {args.output}")

    if args.compile:
        import subprocess
        import os
        pdf_path = os.path.splitext(args.output)[0] + ".pdf"
        result = subprocess.run(["tectonic", args.output],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Compiled {pdf_path}")
        else:
            print(f"tectonic failed:\n{result.stderr}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
