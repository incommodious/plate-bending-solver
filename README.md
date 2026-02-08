<div align="center">

# 🔩 Plate Bending Solver

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-blueviolet.svg)](https://scipy.org/)

**A Python tool for analyzing rectangular plate bending with various boundary conditions and load types.**

*Three solution methods (Levy, FIT, Ritz), PDF report generation, and a graphical user interface.*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Report Generator](#-report-generator) • [Methods](#method-comparison) • [References](#references)

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧮 Solution Methods
| Method | Description |
|--------|-------------|
| **Levy** | Analytical series solution, highest accuracy |
| **FIT** | Finite Integral Transform, equivalent to Levy |
| **Ritz** | Rayleigh-Ritz with beam eigenfunctions, handles all BCs |

</td>
<td width="50%">

### 📊 Output Results
- **Deflection** (W) with contour plots
- **Bending moments** (Mx, My)
- **Bending stresses** (σx, σy)
- **PDF reports** with step-by-step calculations
- **Publication-quality figures**

</td>
</tr>
</table>

### 🔧 Boundary Conditions

| Code | Edges (x=0, y=0, x=a, y=b) | Solver Support |
|------|----------------------------|----------------|
| `SSSS` | All simply supported | Levy, FIT, Ritz |
| `SCSC` | SS x-edges, clamped y-edges | Levy, FIT, Ritz |
| `SCSS` | SS x-edges, C at y=0, S at y=b | Levy, FIT, Ritz |
| `SCSF` | SS x-edges, C at y=0, F at y=b | Levy, FIT, Ritz |
| `SSSF` | SS x-edges, S at y=0, F at y=b | Levy, FIT, Ritz |
| `SFSF` | SS x-edges, F on y-edges | Levy, FIT, Ritz |
| `CCCC` | All clamped | Ritz |
| `FCFC` | Free x-edges, clamped y-edges | Ritz |
| `CCCF` | C on 3 edges, F at y=b | Ritz |
| `FCCC` | F at x=0, C on 3 edges | Ritz |

**S** = Simply Supported, **C** = Clamped, **F** = Free

> **Note:** Levy and FIT require simply supported x-edges (S_S_ pattern). For non-Levy BCs, the Ritz solver is used automatically.

### 📦 Load Types
- ⬛ **Uniform** distributed pressure
- 🟧 **Rectangular** patch load
- 🔵 **Circular** patch load
- 📍 **Point** (concentrated) load

---

## 🚀 Installation

### Requirements
- Python 3.10+
- NumPy, SciPy, Matplotlib
- [tectonic](https://tectonic-typesetting.github.io/) (optional, for PDF compilation)

### Quick Start
```bash
git clone https://github.com/incommodious/plate-bending-solver.git
cd plate-bending-solver

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### GUI Application
```bash
python plate-bending-gui.py
```

### Programmatic Usage

```python
from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.ritz_solver import RitzSolver

# Plate parameters
a, b, h = 1.0, 1.0, 0.01  # m
E, nu = 2.1e11, 0.3
q0 = 10000  # Pa

# Levy solver (Levy-type BCs only)
levy = StableLevySolver(a, b, h, E, nu, bc_y0='C', bc_yb='F', n_terms=50)
result = levy.solve('uniform', q0)
print(f"Max deflection: {result['W_max']*1000:.4f} mm")

# Ritz solver (any BC)
ritz = RitzSolver(a, b, h, E, nu, bc='FCFC', M=15, N=15)
result = ritz.solve('circular', q0, x0=0.5, y0=0.5, R=0.1)
print(f"Max deflection: {result['W_max']*1000:.4f} mm")
```

---

## 📄 Report Generator

Generate publication-quality PDF reports with contour plots and step-by-step calculations.

### CLI Usage
```bash
# Basic report (metric)
python -m plate_bending.report \
  --bc SSSS --load uniform --q 1000 \
  --a 1.0 --b 1.0 --h 0.01 --E 200e9 --nu 0.3 \
  --method levy --output report.tex --compile

# Imperial units with circular load
python -m plate_bending.report \
  --bc FCFC --load circular --q 176.7 \
  --a 6.0 --b 1.5 --h 0.07 --E 29e6 --nu 0.3 \
  --x0 0.75 --y0 0.75 --R 0.35 \
  --method ritz --units imperial --output report.tex --compile
```

### Report Contents
- Input parameters with unit conversions
- Flexural rigidity calculation (step-by-step)
- Governing equation and solution method description
- Convergence study with narrative
- Deflection & stress results at analysis point
- Non-dimensional coefficients for benchmark comparison
- **Contour plots**: deflection field, σx and σy stress fields
- **Geometry diagram**: auto-generated with BC symbols
- **Appendix**: worked step-by-step Ritz/Levy calculation with actual numbers

### Report Modules

| Module | Purpose |
|--------|---------|
| `plate_bending/report.py` | LaTeX report generation, CLI entry point |
| `plate_bending/figures.py` | Matplotlib contour & profile plots |
| `plate_bending/geometry_diagram.py` | Auto BC symbol geometry diagrams |
| `plate_bending/appendix.py` | Step-by-step calculation appendix |

---

## 📁 Project Structure

```
plate-bending-solver/
├── plate-bending-gui.py          # GUI application
├── README.md
├── AUDIT_REPORT.md               # Physics/math audit results
├── requirements.txt
├── plate_bending/                # Core package
│   ├── solvers/
│   │   ├── levy_solver.py        # Levy series solution
│   │   ├── fit_solver.py         # Finite Integral Transform
│   │   ├── ritz_solver.py        # Rayleigh-Ritz method
│   │   └── beam_functions.py     # Beam eigenfunctions (SS, CC, CF, FC, CS, SC, FF)
│   ├── validation/
│   │   └── benchmarks.py         # Timoshenko/Szilard reference values
│   ├── report.py                 # LaTeX report generator
│   ├── figures.py                # Plot generation
│   ├── geometry_diagram.py       # Plate diagram generator
│   └── appendix.py               # Calculation appendix engine
├── tests/
│   ├── test_comprehensive_validation.py  # 5 core validation tests
│   ├── test_fcfc_cccf.py                 # 7 non-Levy BC tests
│   ├── test_report.py                    # 4 report module tests
│   └── deep_audit.py                     # Extended physics audit
├── report_output/                # Generated reports & figures
└── docs/                        # Reference papers
```

---

## 📊 Method Comparison

| Feature | Levy | FIT | Ritz |
|:--------|:----:|:---:|:----:|
| Accuracy | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ (~1-5%) |
| Speed | 🚀 Fast | 🚀 Fast | 🏃 Medium |
| Levy-type BCs | ✅ | ✅ | ✅ |
| Non-Levy BCs (CCCC, FCFC, etc.) | ❌ | ❌ | ✅ |
| Patch/Circular loads | ✅ | ✅ | ✅ |
| Point loads | ✅ | ✅ | ✅ |

> **Levy-type BCs** require simply supported x-edges (S at x=0 and x=a). All other configurations need the Ritz solver.

---

## ✅ Validation

Validated against classical references and cross-method checks:

| BC | Reference | Error |
|----|-----------|-------|
| SSSS | Timoshenko Table 8 | 0.06% |
| SSSF | Timoshenko Table 48 | 0.06% |
| CCCC | Timoshenko Table 35 | 0.66% |
| SCSC | Timoshenko Table 36 | ~0.1% |
| FCFC | Ritz convergence study | Converged (0.47%) |
| CCCF | Ritz convergence study | Converged (0.13%) |

```bash
# Run all tests
python tests/test_comprehensive_validation.py   # 5/5 pass
python tests/test_fcfc_cccf.py                   # 7/7 pass
python -c "from tests.test_report import *; test_report_latex_and_values_ssss_uniform_center(); test_imperial_units_match_metric(); test_fmt_significant_figures(); test_circular_load_report(); print('4/4 report tests pass')"
```

---

## 📚 References

1. Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*, 2nd ed. McGraw-Hill.
2. Szilard, R. (2004). *Theories and Applications of Plate Analysis*. John Wiley & Sons.
3. Ventsel, E. & Krauthammer, T. (2001). *Thin Plates and Shells*. Marcel Dekker.
4. Blevins, R. D. (1979). *Formulas for Natural Frequency and Mode Shape*. Van Nostrand Reinhold.
5. Xu, Q., et al. (2020). "[Analytical Bending Solutions of Orthotropic Rectangular Thin Plates...](https://doi.org/10.1155/2020/8848879)" *Advances in Civil Engineering*.

---

## 📄 License

MIT License — free for academic and commercial use.

---

<div align="center">

**Made with ❤️ for structural engineers and researchers**

</div>
