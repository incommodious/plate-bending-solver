<div align="center">

# 🔩 Plate Bending Solver

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-blueviolet.svg)](https://scipy.org/)

**A Python-based tool for analyzing rectangular plate bending with various boundary conditions and load types.**

*Features three solution methods (Levy, FIT, Ritz) with a graphical user interface for interactive analysis.*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Methods](#method-comparison) • [References](#references)

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
| **FIT** | Finite Integral Transform, extended for Levy-type plates |
| **Ritz** | Rayleigh-Ritz with beam eigenfunctions, flexible |

</td>
<td width="50%">

### 📊 Output Results
- **Deflection** (W)
- **Bending moments** (Mx, My)
- **Von Mises stress**
- **3D surface visualization**
- **Cross-method comparison**

</td>
</tr>
</table>

### 🔧 Boundary Conditions

| Code | Description |
|------|-------------|
| `SSSS` | Simply supported on all edges |
| `SCSC` | Simply supported on x-edges, clamped on y-edges |
| `SCSS` | Simply supported x-edges, clamped at y=0, simply supported at y=b |
| `SCSF` | Simply supported x-edges, clamped at y=0, free at y=b |
| `SSSF` | Simply supported x-edges and y=0, free at y=b |
| `SFSF` | Simply supported x-edges, free on y-edges |

### 📦 Load Types
- ⬛ **Uniform** distributed load
- 🟧 **Rectangular** patch load  
- 🔵 **Circular** patch load
- 📍 **Point** load

---

## 🚀 Installation

### Requirements
- Python 3.10+
- NumPy, SciPy, Matplotlib
- Tkinter (usually included with Python)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/incommodious/plate-bending-solver.git
cd plate-bending-solver

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy scipy matplotlib
```

---

## 💻 Usage

### GUI Application
```bash
python plate-bending-gui.py
```

The GUI allows you to:
1. 🎯 Select boundary condition preset (SSSS, SCSC, etc.)
2. 📐 Enter plate properties (dimensions, thickness, material)
3. ⚖️ Choose load type and parameters
4. ▶️ Run analysis and view results
5. 🔄 Compare Levy, FIT, and Ritz methods

### Programmatic Usage
<details>
<summary>Click to expand code example</summary>

```python
from plate_bending.solvers.levy_solver import StableLevySolver
from plate_bending.solvers.fit_solver import FITSolver
from plate_bending.solvers.ritz_solver import RitzSolver

# Plate parameters
a, b, h = 1.0, 1.0, 0.01  # dimensions and thickness (m)
E, nu = 2.1e11, 0.3       # Young's modulus (Pa), Poisson's ratio
q0 = 10000                 # load intensity (Pa)

# Levy solver for SCSF plate
levy = StableLevySolver(a, b, h, E, nu, bc_y0='C', bc_yb='F', n_terms=50)
result = levy.solve('uniform', q0)
print(f"Levy - Max deflection: {result['W_max']*1000:.4f} mm")

# FIT solver (uses navier for SSSS, levy_ode for other Levy-type BCs)
fit = FITSolver(a, b, h, E, nu, bc='SCSF', n_terms=50)
result = fit.solve('uniform', q0, auto_converge=True)
print(f"FIT - Max deflection: {result['W_max']*1000:.4f} mm (method: {fit.method})")

# Ritz solver
ritz = RitzSolver(a, b, h, E, nu, bc='SCSF', M=15, N=15)
result = ritz.solve('rect_patch', q0, x1=0.3, y1=0.3, x2=0.7, y2=0.7)
print(f"Ritz - Max deflection: {result['W_max']*1000:.4f} mm")
```

</details>

---

## 📁 Project Structure

```
plate-bending-solver/
├── 📄 plate-bending-gui.py      # Main GUI application
├── 📄 README.md                 # This file
├── 📄 PROGRESS.md               # Development history
├── 📦 plate_bending/            # Core package
│   ├── solvers/                 # Solution methods
│   │   ├── levy_solver.py       # Levy series solution
│   │   ├── fit_solver.py        # Finite Integral Transform
│   │   ├── ritz_solver.py       # Rayleigh-Ritz method
│   │   └── beam_functions.py    # Beam eigenfunction library
│   └── validation/              # Benchmarks and tests
├── 🧪 tests/                    # Test scripts
└── 📚 docs/                     # Documentation and references
```

---

## 📊 Method Comparison

| Method | Best For | Accuracy | Speed |
|:------:|:---------|:--------:|:-----:|
| **Levy** | All Levy-type BCs | ⭐⭐⭐ Highest | 🚀 Fast |
| **FIT** | All Levy-type BCs | ⭐⭐⭐ Matches Levy | 🚀 Fast |
| **Ritz** | All BCs, flexible | ⭐⭐ Good (~1-5%) | 🏃 Medium |

<details>
<summary><b>FIT Method Details</b></summary>

- **SSSS plates** (`navier` method): Uses true Navier double sine series (bidirectional FIT)
- **Other Levy-type plates** (`levy_ode` method): Uses sine series in x and solves ODE in y

For non-SSSS Levy-type plates (SCSC, SCSS, SCSF, SSSF, SFSF), FIT uses the `levy_ode` method which is mathematically equivalent to the Levy solver:
- Fourier sine series in x-direction
- 4th-order ODE solution in y-direction with stable exponential basis
- Same boundary condition handling (clamped, simply supported, free)

</details>

<details>
<summary><b>Accuracy Notes</b></summary>

- **Uniform loads**: Levy and FIT match exactly (same closed-form solution)
- **Patch/Circular loads**: FIT matches Levy within ~0.5%
- **Levy vs Ritz**: typically agree within 1-5%
- For SSSF/SFSF plates, Levy and FIT are more accurate than Ritz

</details>

---

## ✅ Validation

The solvers are validated against:
- 📖 Timoshenko & Woinowsky-Krieger analytical solutions
- 🔄 Cross-method verification (Levy vs Ritz)
- 🔬 Physical behavior checks (load ordering, edge effects)

```bash
# Run validation tests
python tests/test_comprehensive_validation.py
```

---

## 📚 References

1. Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*. McGraw-Hill.
2. Blevins, R. D. (1979). *Formulas for Natural Frequency and Mode Shape*. Van Nostrand Reinhold.
3. Xu, Q., et al. (2020). "[Analytical Bending Solutions of Orthotropic Rectangular Thin Plates...](https://doi.org/10.1155/2020/8848879)" *Advances in Civil Engineering*.

---

## 📄 License

MIT License - Free for academic and commercial use.

---

<div align="center">

**Made with ❤️ for structural engineers and researchers**

</div>
