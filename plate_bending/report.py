"""
Point Report Generator for Plate Bending Solvers
================================================

Generates a LaTeX calculation sheet for a specific (x, y) point.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


# Unit conversion factors (from SI base)
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
        "length": ("in", 1.0 / 0.0254),  # m -> in
        "pressure": ("psi", 1.0 / 6894.76),  # Pa -> psi
        "stress": ("psi", 1.0 / 6894.76),  # Pa -> psi (same as pressure)
        "moment": (r"lbf$\cdot$in/in", 1.0 / 0.1129848),  # N·m/m -> lbf·in/in
        "rigidity": (r"lbf$\cdot$in", 1.0 / 0.1129848),  # N·m -> lbf·in
        "force": ("lbf", 1.0 / 4.44822),  # N -> lbf
    },
}


def _convert_unit(value: float, unit_type: str, system: str) -> Tuple[float, str]:
    """Convert a value to the specified unit system. Returns (converted_value, unit_label)."""
    unit_label, factor = UNIT_SYSTEMS[system][unit_type]
    return value * factor, unit_label


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
    x = X[0, :]
    y = Y[:, 0]
    return x, y


def _bilinear_sample(x: float, y: float, X: np.ndarray, Y: np.ndarray, F: np.ndarray) -> float:
    xg, yg = _grid_axes(X, Y)
    if x < xg[0] or x > xg[-1] or y < yg[0] or y > yg[-1]:
        raise ValueError(f"Point ({x}, {y}) out of bounds.")

    i = np.searchsorted(xg, x) - 1
    j = np.searchsorted(yg, y) - 1
    i = int(np.clip(i, 0, len(xg) - 2))
    j = int(np.clip(j, 0, len(yg) - 2))

    x1, x2 = xg[i], xg[i + 1]
    y1, y2 = yg[j], yg[j + 1]

    f11 = F[j, i]
    f21 = F[j, i + 1]
    f12 = F[j + 1, i]
    f22 = F[j + 1, i + 1]

    if x2 == x1 or y2 == y1:
        return float(f11)

    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)

    return float((1 - tx) * (1 - ty) * f11 + tx * (1 - ty) * f21 + (1 - tx) * ty * f12 + tx * ty * f22)


def _solve_case(inputs: ReportInputs, n_terms: int = 100, ritz_terms: int = 10) -> Dict:
    bc = _validate_bc(inputs.bc)
    method = inputs.method.lower()

    if method == "levy":
        if bc[0] != "S" or bc[2] != "S":
            raise ValueError("Levy solver requires simply supported x-edges (bc[0] and bc[2]).")
        solver = StableLevySolver(inputs.a, inputs.b, inputs.h, inputs.E, inputs.nu,
                                  bc_y0=bc[1], bc_yb=bc[3], n_terms=n_terms)
        results = solver.solve(inputs.load_type, inputs.q0)
    elif method == "fit":
        solver = FITSolver(inputs.a, inputs.b, inputs.h, inputs.E, inputs.nu, bc, n_terms)
        results = solver.solve(inputs.load_type, inputs.q0, auto_converge=False)
    elif method == "ritz":
        solver = RitzSolver(inputs.a, inputs.b, inputs.h, inputs.E, inputs.nu, bc,
                            M=ritz_terms, N=ritz_terms)
        results = solver.solve(inputs.load_type, inputs.q0)
    else:
        raise ValueError(f"Unknown method: {inputs.method}")

    return results


def _point_results(results: Dict, x: float, y: float) -> Dict[str, float]:
    X = results["X"]
    Y = results["Y"]
    data = {
        "W": _bilinear_sample(x, y, X, Y, results["W"]),
        "Mx": _bilinear_sample(x, y, X, Y, results["Mx"]),
        "My": _bilinear_sample(x, y, X, Y, results["My"]),
        "sigma_x": _bilinear_sample(x, y, X, Y, results["sigma_x"]),
        "sigma_y": _bilinear_sample(x, y, X, Y, results["sigma_y"]),
    }
    return data


def _build_convergence_table(inputs: ReportInputs, base_n_terms: int, ritz_terms: int) -> List[Tuple[str, float, float]]:
    method = inputs.method.lower()
    entries: List[Tuple[str, float, float]] = []

    if method == "levy":
        levels = [5, 10, 20, 40, 60, base_n_terms]
        levels = [n for n in levels if n <= base_n_terms]
        prev = None
        for n in levels:
            res = _solve_case(inputs, n_terms=n, ritz_terms=ritz_terms)
            val = _point_results(res, inputs.x, inputs.y)["W"]
            delta = abs(val - prev) if prev is not None else np.nan
            entries.append((f"$N={n}$", val, delta))
            prev = val
    elif method == "fit":
        levels = [5, 10, 20, 30, base_n_terms]
        levels = [n for n in levels if n <= base_n_terms]
        prev = None
        for n in levels:
            res = _solve_case(inputs, n_terms=n, ritz_terms=ritz_terms)
            val = _point_results(res, inputs.x, inputs.y)["W"]
            delta = abs(val - prev) if prev is not None else np.nan
            entries.append((f"$N={n}$", val, delta))
            prev = val
    elif method == "ritz":
        levels = [2, 4, 6, 8, ritz_terms]
        levels = [n for n in levels if n <= ritz_terms]
        prev = None
        for n in levels:
            res = _solve_case(inputs, n_terms=base_n_terms, ritz_terms=n)
            val = _point_results(res, inputs.x, inputs.y)["W"]
            delta = abs(val - prev) if prev is not None else np.nan
            entries.append((f"$M=N={n}$", val, delta))
            prev = val
            prev = val
    else:
        return entries

    return entries


def _fmt(value: float, sig: int = 4) -> str:
    """Format number for LaTeX with *sig* significant figures.

    Strategy:
      - 0             → "0"
      - 0.01 … 9999   → fixed-point with *sig* significant figures
      - outside that   → engineering notation (exponent divisible by 3,
                         mantissa in [1, 1000))
    Trailing zeros after the decimal point are stripped for readability.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    abs_val = abs(value)
    if abs_val == 0:
        return "0"

    # Determine order of magnitude
    exp = int(np.floor(np.log10(abs_val)))

    if -2 <= exp <= 3:
        # Fixed-point: compute decimal places to give *sig* significant figures
        decimal_places = max(sig - exp - 1, 0)
        s = f"{value:.{decimal_places}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    # Engineering notation: exponent divisible by 3, mantissa in [1, 1000)
    eng_exp = exp - (exp % 3) if exp >= 0 else exp - ((exp % 3 + 3) % 3)
    # Correct: for negative exponents we want the next-lower multiple of 3
    # e.g. exp = -4 → eng_exp = -6 so mantissa ~100
    mantissa = value / 10**eng_exp
    # Decimal places for the mantissa so total sig figs are preserved
    mantissa_exp = int(np.floor(np.log10(abs(mantissa))))
    dec = max(sig - mantissa_exp - 1, 0)
    m_str = f"{mantissa:.{dec}f}"
    if '.' in m_str:
        m_str = m_str.rstrip('0').rstrip('.')
    return rf"{m_str} \times 10^{{{eng_exp}}}"


def _fmt_plain(value: float, sig: int = 4) -> str:
    """Format number for plain text contexts with *sig* significant figures."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    abs_val = abs(value)
    if abs_val == 0:
        return "0"
    exp = int(np.floor(np.log10(abs_val)))
    if -2 <= exp <= 3:
        decimal_places = max(sig - exp - 1, 0)
        s = f"{value:.{decimal_places}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s
    return f"{value:.{sig-1}e}"


def _method_equations(inputs: ReportInputs, D_disp: float, n_terms: int, ritz_terms: int,
                      a_disp: float, b_disp: float, q0_disp: float,
                      len_unit: str, press_unit: str, rig_unit: str) -> str:
    method = inputs.method.lower()
    
    D_str = _fmt(D_disp, 4)
    q0_str = _fmt(q0_disp, 4)
    a_str = _fmt(a_disp, 4)
    b_str = _fmt(b_disp, 4)

    if method == "levy":
        return (
            r"\begin{align*}"
            r"w(x,y) &= \sum_{m=1}^{N} \sin\left(\frac{m\pi x}{a}\right) Y_m(y), \quad "
            r"\alpha_m = \frac{m\pi}{a} \\"
            r"D \left(Y_m'''' - 2\alpha_m^2 Y_m'' + \alpha_m^4 Y_m\right) &= q_m, \quad "
            r"q_m = \frac{4 q_0}{m\pi}\ \text{(odd $m$ only)} \\"
            rf"a = {a_str}\ \text{{{len_unit}}},\ D = {D_str}\ \text{{{rig_unit}}},\ "
            rf"q_0 = {q0_str}\ \text{{{press_unit}}},\ N = {n_terms}"
            r"\end{align*}"
        )
    if method == "fit":
        return (
            r"\begin{align*}"
            r"w(x,y) &= \sum_{m=1}^{N}\sum_{n=1}^{N} W_{mn} "
            r"\sin\left(\frac{m\pi x}{a}\right)\sin\left(\frac{n\pi y}{b}\right) \\"
            r"W_{mn} &= \frac{q_{mn}}{D\left(\left(\frac{m\pi}{a}\right)^2+\left(\frac{n\pi}{b}\right)^2\right)^2},\quad "
            r"q_{mn} = \frac{16 q_0}{m n \pi^2}\ \text{(odd $m,n$)} \\"
            rf"a = {a_str}\ \text{{{len_unit}}},\ b = {b_str}\ \text{{{len_unit}}},\ "
            rf"D = {D_str}\ \text{{{rig_unit}}},\ q_0 = {q0_str}\ \text{{{press_unit}}},\ N = {n_terms}"
            r"\end{align*}"
        )
    if method == "ritz":
        return (
            r"\begin{align*}"
            r"w(x,y) &= \sum_{m=1}^{M}\sum_{n=1}^{N} A_{mn}\,\phi_m(x)\,\psi_n(y) \\"
            r"\mathbf{K}\mathbf{A} &= \mathbf{F},\quad "
            r"K_{mnpq} = D\left(I_{x2}I_{y0}+I_{x0}I_{y2}+\nu(I_{x20}I_{y20}+I_{x20}I_{y20})"
            r"+2(1-\nu)I_{x1}I_{y1}\right) \\"
            rf"a = {a_str}\ \text{{{len_unit}}},\ b = {b_str}\ \text{{{len_unit}}},\ "
            rf"D = {D_str}\ \text{{{rig_unit}}},\ \nu = {inputs.nu:.4g},\ M=N={ritz_terms}"
            r"\end{align*}"
        )
    return ""


def generate_point_report(inputs: ReportInputs,
                          n_terms: int = 100,
                          ritz_terms: int = 10,
                          include_convergence: bool = True) -> Tuple[str, Dict]:
    bc = _validate_bc(inputs.bc)
    units = inputs.units if inputs.units in UNIT_SYSTEMS else "metric"
    inputs = ReportInputs(
        a=inputs.a, b=inputs.b, h=inputs.h, E=inputs.E, nu=inputs.nu, bc=bc,
        load_type=inputs.load_type, q0=inputs.q0, x=inputs.x, y=inputs.y,
        method=inputs.method, units=units,
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

    # Unit conversions for display
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
    bc_desc = f"$x=0$: {bc_names.get(bc[0],'?')}, $y=0$: {bc_names.get(bc[1],'?')}, $x=a$: {bc_names.get(bc[2],'?')}, $y=b$: {bc_names.get(bc[3],'?')}"
    
    unit_note = "Imperial (in, psi, lbf)" if units == "imperial" else "Metric (m, Pa, N)"

    latex_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{siunitx}",
        r"\geometry{margin=1in}",
        r"\title{Rectangular Plate Bending Analysis\\[0.5em]\large Point Calculation Report}",
        r"\author{Plate Bending Solver}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        "",
        rf"\noindent\textbf{{Unit System:}} {unit_note}",
        "",
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
        rf"Configuration: \textbf{{{inputs.bc}}} — {bc_desc}",
        "",
        r"\subsection{Loading}",
        rf"Type: \textbf{{{inputs.load_type}}}, $q_0 = {_fmt(q0_disp, 4)}$ {press_unit}",
        "",
        r"\subsection{Analysis Point}",
        rf"$(x, y) = ({_fmt(x_disp, 4)}, {_fmt(y_disp, 4)})$ {len_unit}",
        "",
        r"\section{Governing Equation}",
        r"The deflection $w(x,y)$ of a thin plate under transverse loading satisfies the biharmonic equation:",
        r"\begin{equation}",
        r"D\nabla^4 w(x,y) = D\left(\frac{\partial^4 w}{\partial x^4} + 2\frac{\partial^4 w}{\partial x^2 \partial y^2} + \frac{\partial^4 w}{\partial y^4}\right) = q(x,y)",
        r"\end{equation}",
        "",
        r"\section{Solution Method}",
        rf"Method: \textbf{{{inputs.method.capitalize()}}}",
        "",
        _method_equations(inputs, D_disp, n_terms, ritz_terms,
                         a_disp, b_disp, q0_disp, len_unit, press_unit, rig_unit),
        "",
        r"\section{Convergence Study}",
    ]

    if convergence:
        _, len_conv = UNIT_SYSTEMS[units]["length"]
        latex_lines += [
            rf"\noindent Deflection values in {len_unit}.",
            "",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            rf"Series Level & $w(x,y)$ [{len_unit}] & $| \Delta w |$ [{len_unit}] \\",
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

    latex_lines += [
        r"\section{Results at Analysis Point}",
        rf"Results computed at $(x, y) = ({_fmt(x_disp, 4)}, {_fmt(y_disp, 4)})$ {len_unit}:",
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
        r"\subsection{Bending Stresses}",
        r"The maximum bending stresses occur at the plate surfaces ($z = \pm h/2$) and are computed from:",
        r"\begin{equation}",
        r"\sigma_x = \frac{6 M_x}{h^2}, \qquad \sigma_y = \frac{6 M_y}{h^2}",
        r"\end{equation}",
        "",
        r"\noindent Substituting values:",
        r"\begin{align}",
        rf"\sigma_x &= \frac{{6 \times M_x}}{{h^2}} = {_fmt(sigma_x_disp, 6)} \ \text{{{stress_unit}}} \\",
        rf"\sigma_y &= \frac{{6 \times M_y}}{{h^2}} = {_fmt(sigma_y_disp, 6)} \ \text{{{stress_unit}}}",
        r"\end{align}",
        "",
        r"\section{Non-Dimensional Coefficients}",
        r"For comparison with tabulated reference values (e.g., Timoshenko \& Woinowsky-Krieger):",
        r"\begin{equation}",
        r"\bar{w} = \frac{w D}{q_0 a^4}, \qquad \bar{M}_x = \frac{M_x}{q_0 a^2}, \qquad \bar{M}_y = \frac{M_y}{q_0 a^2}",
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
        r"\section{Reference Validation}",
        rf"Benchmark source for BC = {inputs.bc}: {bench_source}.",
        "",
        r"\section*{References}",
        r"\begin{enumerate}",
        r"\item Timoshenko, S. and Woinowsky-Krieger, S., \emph{Theory of Plates and Shells}, 2nd ed., McGraw-Hill, 1959.",
        r"\item Szilard, R., \emph{Theories and Applications of Plate Analysis}, John Wiley \& Sons, 2004.",
        r"\item Xu, R. et al., ``Analytical Bending Solutions of Orthotropic Rectangular Thin Plates with Two Adjacent Edges Free,'' \emph{Archives of Applied Mechanics}, 2020.",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a point report for plate bending solvers.")
    parser.add_argument("--bc", required=True, help="Boundary conditions (e.g., SSSS, SCSF)")
    parser.add_argument("--load", dest="load_type", required=True, help="Load type (uniform, point, circular, rect_patch)")
    parser.add_argument("--q", dest="q0", type=float, required=True, help="Load magnitude (Pa for metric, psi for imperial)")
    parser.add_argument("--a", type=float, required=True, help="Plate length in x (m for metric, in for imperial)")
    parser.add_argument("--b", type=float, required=True, help="Plate length in y (m for metric, in for imperial)")
    parser.add_argument("--h", type=float, required=True, help="Plate thickness (m for metric, in for imperial)")
    parser.add_argument("--E", dest="E", type=float, required=True, help="Young's modulus (Pa for metric, psi for imperial)")
    parser.add_argument("--nu", type=float, required=True, help="Poisson's ratio")
    parser.add_argument("--x", type=float, default=None, help="Point x-coordinate (default: center)")
    parser.add_argument("--y", type=float, default=None, help="Point y-coordinate (default: center)")
    parser.add_argument("--method", required=True, choices=["levy", "fit", "ritz"], help="Solver method")
    parser.add_argument("--units", choices=["metric", "imperial"], default="metric", help="Unit system for output (default: metric)")
    parser.add_argument("--output", required=True, help="Output LaTeX file path")
    parser.add_argument("--compile", action="store_true", help="Compile to PDF with tectonic after generating .tex")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    x = args.x if args.x is not None else args.a / 2.0
    y = args.y if args.y is not None else args.b / 2.0
    
    # If imperial, convert input to SI for internal calculations
    if args.units == "imperial":
        # Input is in imperial, convert to SI
        a = args.a * 0.0254  # in -> m
        b = args.b * 0.0254
        h = args.h * 0.0254
        E = args.E * 6894.76  # psi -> Pa
        q0 = args.q0 * 6894.76
        x = x * 0.0254
        y = y * 0.0254
    else:
        a, b, h, E, q0 = args.a, args.b, args.h, args.E, args.q0
    
    inputs = ReportInputs(
        a=a,
        b=b,
        h=h,
        E=E,
        nu=args.nu,
        bc=args.bc,
        load_type=args.load_type,
        q0=q0,
        x=x,
        y=y,
        method=args.method,
        units=args.units,
    )
    report, _ = generate_point_report(inputs)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {args.output}")

    if args.compile:
        import subprocess
        import os
        pdf_path = os.path.splitext(args.output)[0] + ".pdf"
        result = subprocess.run(["tectonic", args.output], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Compiled {pdf_path}")
        else:
            print(f"tectonic failed:\n{result.stderr}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
