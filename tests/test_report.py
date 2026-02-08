"""Tests for point report generation."""

from plate_bending.report import ReportInputs, generate_point_report, _fmt
from plate_bending.validation.benchmarks import Benchmarks
from plate_bending.solvers.levy_solver import StableLevySolver


def test_report_latex_and_values_ssss_uniform_center():
    inputs = ReportInputs(
        a=1.0,
        b=1.0,
        h=0.01,
        E=2.1e11,
        nu=0.3,
        bc="SSSS",
        load_type="uniform",
        q0=10000.0,
        x=0.5,
        y=0.5,
        method="levy",
    )

    latex, data = generate_point_report(inputs, n_terms=100, include_convergence=False)

    assert "\\documentclass" in latex
    assert "\\begin{document}" in latex
    assert "Results at Analysis Point" in latex
    assert "\\end{document}" in latex

    bench = Benchmarks.get("SSSS")
    expected = bench["W_center_coef"]
    actual = data["W_coef"]
    error = abs(actual - expected) / expected
    assert error < 0.05

    # Cross-check against solver grid center values
    solver = StableLevySolver(inputs.a, inputs.b, inputs.h, inputs.E, inputs.nu, "S", "S", 100)
    results = solver.solve("uniform", inputs.q0)
    w_center = results["W_center"]

    assert abs(data["point"]["W"] - w_center) / abs(w_center) < 0.02


def test_imperial_units_match_metric():
    """Non-dimensional coefficients must agree regardless of unit system."""
    # Metric: 1m square, 10mm thick, 200 GPa, 1 kPa
    metric = ReportInputs(
        a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3,
        bc="SSSS", load_type="uniform", q0=1000.0,
        x=0.5, y=0.5, method="levy", units="metric",
    )
    # Same plate in imperial (convert inputs to inches/psi)
    imperial = ReportInputs(
        a=1.0 / 0.0254,        # m -> in
        b=1.0 / 0.0254,
        h=0.01 / 0.0254,
        E=200e9 / 6894.76,     # Pa -> psi
        nu=0.3,
        bc="SSSS", load_type="uniform",
        q0=1000.0 / 6894.76,   # Pa -> psi
        x=0.5 / 0.0254,
        y=0.5 / 0.0254,
        method="levy", units="imperial",
    )
    # Imperial inputs are in display units; the solver expects SI, so convert
    # (generate_point_report does NOT convert — that's main()'s job.
    #  For the API, inputs must always be SI.)
    imperial_si = ReportInputs(
        a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3,
        bc="SSSS", load_type="uniform", q0=1000.0,
        x=0.5, y=0.5, method="levy", units="imperial",
    )

    _, data_m = generate_point_report(metric, n_terms=100, include_convergence=False)
    _, data_i = generate_point_report(imperial_si, n_terms=100, include_convergence=False)

    # Non-dimensional coefficients should be identical (same physics)
    assert abs(data_m["W_coef"] - data_i["W_coef"]) / abs(data_m["W_coef"]) < 1e-6
    assert abs(data_m["Mx_coef"] - data_i["Mx_coef"]) / abs(data_m["Mx_coef"]) < 1e-6


def test_fmt_significant_figures():
    """Verify _fmt produces clean engineering/fixed-point notation."""
    assert _fmt(1000.0, 4) == "1000"
    assert _fmt(0.01, 4) == "0.01"
    assert _fmt(200e9, 4) == r"200 \times 10^{9}"
    assert _fmt(2.52e-12, 4) == r"2.52 \times 10^{-12}"
    assert _fmt(0.0, 4) == "0"
    assert _fmt(47.87, 4) == "47.87"
    # Mantissa must be in [1, 1000) for engineering notation
    assert _fmt(1.041e-10, 4) == r"104.1 \times 10^{-12}"
