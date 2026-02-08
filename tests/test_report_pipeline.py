"""
Tests for the plate bending report generation pipeline.

Covers: report.py, figures.py, geometry_diagram.py, appendix.py
"""
import math
import os
import tempfile

import numpy as np
import pytest

from plate_bending.report import (
    ReportInputs,
    generate_point_report,
    _fmt,
    _bilinear_sample,
)
from plate_bending.figures import (
    generate_deflection_contour,
    generate_stress_contours,
    generate_deflection_profile,
)
from plate_bending.geometry_diagram import generate_geometry_diagram
from plate_bending.appendix import generate_appendix_latex, fmt_eng
from plate_bending.solvers.ritz_solver import RitzSolver
from plate_bending.solvers.levy_solver import StableLevySolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_BCS = ["SSSS", "SCSF", "SCSC", "SSSF", "SSSC", "SFSF",
           "CCCC", "FCFC", "CFCF", "CCCF"]

LEVY_BCS = [bc for bc in ALL_BCS if bc[0] == "S" and bc[2] == "S"]
NON_LEVY_BCS = [bc for bc in ALL_BCS if not (bc[0] == "S" and bc[2] == "S")]

LOAD_TYPES = ["uniform", "circular", "rect_patch", "point"]

STD = dict(a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3, q0=10000)


def _make_inputs(bc="SSSS", load_type="uniform", method="levy", units="metric",
                 x=0.5, y=0.5, **kw):
    params = dict(STD, bc=bc, load_type=load_type, method=method, units=units,
                  x=x, y=y)
    if load_type == "circular":
        params.setdefault("x0", 0.5)
        params.setdefault("y0", 0.5)
        params.setdefault("R", 0.1)
    elif load_type == "point":
        params.setdefault("x0", 0.5)
        params.setdefault("y0", 0.5)
    elif load_type == "rect_patch":
        params.setdefault("x1", 0.3)
        params.setdefault("y1", 0.3)
        params.setdefault("x2", 0.7)
        params.setdefault("y2", 0.7)
    params.update(kw)
    return ReportInputs(**params)


def _solve_ritz(bc="SSSS", load_type="uniform", M=8, N=8, **kw):
    s = RitzSolver(STD["a"], STD["b"], STD["h"], STD["E"], STD["nu"], bc=bc, M=M, N=N)
    load_kw = dict(load_type=load_type, q0=STD["q0"])
    if load_type == "circular":
        load_kw.update(x0=0.5, y0=0.5, R=0.1)
    elif load_type == "point":
        load_kw.update(x0=0.5, y0=0.5)
    elif load_type == "rect_patch":
        load_kw.update(x1=0.3, y1=0.3, x2=0.7, y2=0.7)
    load_kw.update(kw)
    return s.solve(**load_kw)


def _solve_levy(bc="SSSS", load_type="uniform", n_terms=20, **kw):
    s = StableLevySolver(STD["a"], STD["b"], STD["h"], STD["E"], STD["nu"],
                         bc_y0=bc[1], bc_yb=bc[3], n_terms=n_terms)
    load_kw = dict(load_type=load_type, q0=STD["q0"])
    if load_type == "circular":
        load_kw.update(x0=0.5, y0=0.5, R=0.1)
    elif load_type == "point":
        load_kw.update(x0=0.5, y0=0.5)
    elif load_type == "rect_patch":
        load_kw.update(x1=0.3, y1=0.3, x2=0.7, y2=0.7)
    load_kw.update(kw)
    return s.solve(**load_kw)


# ===================================================================
# Report module tests (1-10)
# ===================================================================

class TestReportBCs:
    """1. All 10 BCs generate valid LaTeX."""

    @pytest.mark.parametrize("bc", ALL_BCS)
    def test_bc_generates_valid_latex(self, bc):
        if bc in LEVY_BCS:
            method = "levy"
        else:
            method = "ritz"
        inp = _make_inputs(bc=bc, method=method)
        latex, data = generate_point_report(inp, n_terms=5, ritz_terms=4,
                                            include_convergence=False)
        assert r"\begin{document}" in latex
        assert r"\end{document}" in latex


class TestReportLoadTypes:
    """2. All 4 load types produce reports without error."""

    @pytest.mark.parametrize("load_type", LOAD_TYPES)
    def test_load_type(self, load_type):
        method = "levy" if load_type == "point" else "ritz"
        inp = _make_inputs(load_type=load_type, method=method)
        latex, data = generate_point_report(inp, n_terms=5, ritz_terms=4,
                                            include_convergence=False)
        assert r"\begin{document}" in latex


class TestImperialMetricCoefficients:
    """3. Imperial vs metric: non-dimensional coefficients must be identical."""

    def test_nondim_coefficients_match(self):
        inp_m = _make_inputs(units="metric", method="ritz")
        inp_i = _make_inputs(units="imperial", method="ritz")
        _, data_m = generate_point_report(inp_m, ritz_terms=6, include_convergence=False)
        _, data_i = generate_point_report(inp_i, ritz_terms=6, include_convergence=False)
        assert data_m["W_coef"] == pytest.approx(data_i["W_coef"], rel=1e-10)
        assert data_m["Mx_coef"] == pytest.approx(data_i["Mx_coef"], rel=1e-10)
        assert data_m["My_coef"] == pytest.approx(data_i["My_coef"], rel=1e-10)


class TestFmtEdgeCases:
    """4. _fmt edge cases."""

    def test_zero(self):
        assert _fmt(0) == "0"

    def test_nan(self):
        assert _fmt(float("nan")) == "-"

    def test_none(self):
        assert _fmt(None) == "-"

    def test_very_large(self):
        r = _fmt(1e15)
        assert r"10^" in r

    def test_very_small_positive(self):
        r = _fmt(1e-15)
        assert r"10^" in r

    def test_negative(self):
        r = _fmt(-42.0)
        assert "-" in r

    def test_tiny(self):
        r = _fmt(1e-300)
        assert r"10^" in r


class TestBilinearSampleGrid:
    """5. _bilinear_sample at grid points matches array values exactly."""

    def test_grid_points(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 2, 4)
        X, Y = np.meshgrid(x, y)
        F = np.sin(X) * np.cos(Y)
        for j in range(len(y)):
            for i in range(len(x)):
                val = _bilinear_sample(x[i], y[j], X, Y, F)
                assert val == pytest.approx(F[j, i], abs=1e-14)


class TestBilinearSampleMidpoints:
    """6. _bilinear_sample at midpoints is average of neighbors (within 1%)."""

    def test_midpoints(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        X, Y = np.meshgrid(x, y)
        F = X**2 + Y**2  # bilinear on each cell is approximate
        # For a bilinear function, midpoint should equal average of 4 corners
        for j in range(len(y) - 1):
            for i in range(len(x) - 1):
                mx = (x[i] + x[i + 1]) / 2
                my = (y[j] + y[j + 1]) / 2
                val = _bilinear_sample(mx, my, X, Y, F)
                avg = (F[j, i] + F[j, i + 1] + F[j + 1, i] + F[j + 1, i + 1]) / 4
                assert val == pytest.approx(avg, rel=0.01)


class TestBilinearSampleOutOfBounds:
    """7. _bilinear_sample out of bounds raises ValueError."""

    def test_out_of_bounds(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        X, Y = np.meshgrid(x, y)
        F = np.ones_like(X)
        with pytest.raises(ValueError):
            _bilinear_sample(-0.1, 0.5, X, Y, F)
        with pytest.raises(ValueError):
            _bilinear_sample(0.5, 1.1, X, Y, F)


class TestRitzNonLevyBCs:
    """8. Report with ritz method for non-Levy BCs."""

    @pytest.mark.parametrize("bc", NON_LEVY_BCS)
    def test_ritz_non_levy(self, bc):
        inp = _make_inputs(bc=bc, method="ritz")
        latex, data = generate_point_report(inp, ritz_terms=4,
                                            include_convergence=False)
        assert r"\begin{document}" in latex
        assert data["W_coef"] != 0


class TestConvergenceTable:
    """9. Report with convergence=True produces convergence table."""

    def test_convergence_in_latex(self):
        inp = _make_inputs(method="ritz")
        latex, _ = generate_point_report(inp, ritz_terms=6,
                                         include_convergence=True)
        assert r"\Delta w" in latex or "Convergence" in latex


class TestPointLoad:
    """10. ReportInputs with point load (q0 is force, not pressure)."""

    def test_point_load_report(self):
        inp = _make_inputs(load_type="point", method="levy", q0=500)
        latex, data = generate_point_report(inp, ritz_terms=4,
                                            include_convergence=False)
        assert "force" in latex.lower() or "concentrated" in latex.lower() or "P =" in latex


# ===================================================================
# Figures tests (11-15)
# ===================================================================

@pytest.fixture
def results_standard():
    return _solve_ritz()


@pytest.fixture
def results_extreme():
    """10:1 aspect ratio plate."""
    s = RitzSolver(10.0, 1.0, 0.01, 200e9, 0.3, bc="SSSS", M=6, N=6)
    return s.solve(load_type="uniform", q0=10000)


class TestFiguresProducePNG:
    """11. All 3 figure functions produce PNG files >1KB."""

    def test_deflection_contour(self, results_standard, tmp_path):
        p = str(tmp_path / "defl.png")
        generate_deflection_contour(results_standard, p)
        assert os.path.exists(p) and os.path.getsize(p) > 1024

    def test_stress_contours(self, results_standard, tmp_path):
        p = str(tmp_path / "stress.png")
        generate_stress_contours(results_standard, p)
        assert os.path.exists(p) and os.path.getsize(p) > 1024

    def test_deflection_profile(self, results_standard, tmp_path):
        p = str(tmp_path / "profile.png")
        generate_deflection_profile(results_standard, p)
        assert os.path.exists(p) and os.path.getsize(p) > 1024


class TestFiguresUnits:
    """12. Both imperial and metric units produce files."""

    @pytest.mark.parametrize("units", ["imperial", "metric"])
    def test_units(self, results_standard, tmp_path, units):
        p = str(tmp_path / f"defl_{units}.png")
        generate_deflection_contour(results_standard, p, units=units)
        assert os.path.exists(p) and os.path.getsize(p) > 1024


class TestDeflectionContourLoadCenter:
    """13. generate_deflection_contour with and without load_center."""

    def test_without_load_center(self, results_standard, tmp_path):
        p = str(tmp_path / "no_lc.png")
        generate_deflection_contour(results_standard, p)
        assert os.path.exists(p)

    def test_with_load_center(self, results_standard, tmp_path):
        p = str(tmp_path / "with_lc.png")
        generate_deflection_contour(results_standard, p, load_center=(500, 500),
                                    load_radius=50)
        assert os.path.exists(p)


class TestDeflectionProfileCustomY:
    """14. generate_deflection_profile with custom y_section."""

    def test_custom_y(self, results_standard, tmp_path):
        p = str(tmp_path / "prof_custom.png")
        generate_deflection_profile(results_standard, p, y_section=0.25)
        assert os.path.exists(p) and os.path.getsize(p) > 1024


class TestFiguresExtremeAspect:
    """15. Figures with extreme aspect ratio plate (10:1)."""

    def test_extreme_aspect(self, results_extreme, tmp_path):
        p1 = str(tmp_path / "ext_defl.png")
        p2 = str(tmp_path / "ext_stress.png")
        p3 = str(tmp_path / "ext_prof.png")
        generate_deflection_contour(results_extreme, p1)
        generate_stress_contours(results_extreme, p2)
        generate_deflection_profile(results_extreme, p3)
        for p in [p1, p2, p3]:
            assert os.path.exists(p) and os.path.getsize(p) > 1024


# ===================================================================
# Geometry diagram tests (16-20)
# ===================================================================

class TestDiagramAllBCs:
    """16. All 10 BC strings produce diagrams without error."""

    @pytest.mark.parametrize("bc", ALL_BCS)
    def test_bc_diagram(self, bc, tmp_path):
        p = str(tmp_path / f"diag_{bc}.png")
        generate_geometry_diagram(1.0, 1.0, bc, p)
        assert os.path.exists(p) and os.path.getsize(p) > 1024


class TestDiagramAllLoadTypes:
    """17. All 4 load types visualize without error."""

    @pytest.mark.parametrize("load_type", LOAD_TYPES)
    def test_load_type_diagram(self, load_type, tmp_path):
        p = str(tmp_path / f"diag_{load_type}.png")
        kw = {}
        if load_type == "uniform":
            pass
        elif load_type == "circular":
            kw = dict(load_center=(0.5, 0.5), load_radius=0.1)
        elif load_type == "rect_patch":
            kw = dict(load_bounds=(0.3, 0.3, 0.7, 0.7))
        elif load_type == "point":
            kw = dict(load_center=(0.5, 0.5))
        generate_geometry_diagram(1.0, 1.0, "SSSS", p, load_type=load_type, **kw)
        assert os.path.exists(p)


class TestDiagramNoLoad:
    """18. Diagram with no load (load_type=None)."""

    def test_no_load(self, tmp_path):
        p = str(tmp_path / "no_load.png")
        generate_geometry_diagram(1.0, 1.0, "SSSS", p, load_type=None)
        assert os.path.exists(p)


class TestDiagramExtremeDimensions:
    """19. Diagram with extreme dimensions (100:1 ratio)."""

    def test_extreme_ratio(self, tmp_path):
        p = str(tmp_path / "extreme.png")
        generate_geometry_diagram(100.0, 1.0, "SSSS", p)
        assert os.path.exists(p)


class TestDiagramUnits:
    """20. Diagram with imperial and metric units."""

    @pytest.mark.parametrize("units", ["imperial", "metric"])
    def test_units(self, units, tmp_path):
        p = str(tmp_path / f"diag_{units}.png")
        generate_geometry_diagram(1.0, 1.0, "SSSS", p, units=units, h=0.01)
        assert os.path.exists(p)


# ===================================================================
# Appendix tests (21-25)
# ===================================================================

def _std_appendix_inputs(units="metric"):
    return dict(a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3, bc="SSSS",
                q0=10000, x=0.5, y=0.5, load_type="uniform", units=units)


def _std_convergence_data():
    return [
        {"n": 2, "w": -1e-4},
        {"n": 4, "w": -1.1e-4},
        {"n": 6, "w": -1.12e-4},
        {"n": 8, "w": -1.121e-4},
    ]


class TestFmtEngMatchesFmt:
    """21. fmt_eng matches _fmt for common values."""

    @pytest.mark.parametrize("val", [0, 1, 1000, 1e6, 1e-3, 0.00406])
    def test_match(self, val):
        assert fmt_eng(val) == _fmt(val)


class TestAppendixValidLatex:
    """22. generate_appendix_latex produces valid LaTeX fragments."""

    def test_valid_latex(self):
        latex = generate_appendix_latex(
            _std_appendix_inputs(), {}, _std_convergence_data(), method="ritz"
        )
        assert r"\appendix" in latex
        assert r"\section" in latex
        assert r"D" in latex


class TestAppendixMethods:
    """23. Appendix with ritz and levy methods."""

    @pytest.mark.parametrize("method", ["ritz", "levy"])
    def test_method(self, method):
        latex = generate_appendix_latex(
            _std_appendix_inputs(), {}, _std_convergence_data(), method=method
        )
        assert method.capitalize() in latex or method in latex


class TestAppendixEmptyConvergence:
    """24. Appendix with empty convergence_data still works."""

    def test_empty(self):
        latex = generate_appendix_latex(
            _std_appendix_inputs(), {}, [], method="ritz"
        )
        assert r"\appendix" in latex
        assert "No convergence data" in latex


class TestAppendixUnits:
    """25. Appendix with imperial and metric units."""

    @pytest.mark.parametrize("units", ["imperial", "metric"])
    def test_units(self, units):
        latex = generate_appendix_latex(
            _std_appendix_inputs(units=units), {}, _std_convergence_data(), method="ritz"
        )
        if units == "imperial":
            assert "in" in latex or "psi" in latex
        else:
            assert "Pa" in latex or r"N" in latex
