# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this project is

A Python tool for **Kirchhoff (thin) plate bending** of rectangular plates. It computes
deflection `W`, bending moments `Mx/My/Mxy`, and bending stresses for various edge
conditions and load types using three independent methods, and produces LaTeX/PDF
engineering reports with step-by-step, hand-checkable calculations.

The owner is a mechanical engineer, not a professional programmer. **Mathematical and
physical correctness is the top priority** — every change to solver math must be backed by
a benchmark or cross-solver comparison. Explain engineering/math reasoning plainly in
commit messages and summaries.

## Repository layout

```
plate-bending-gui.py              # Tkinter GUI (Levy/FIT/Ritz side-by-side comparison)
plate_bending/
  __init__.py                     # __version__ = "1.0.0"
  solvers/
    levy_solver.py                # StableLevySolver  – Levy single series, stable exp. basis
    fit_solver.py                 # FITSolver         – Navier (SSSS) or Levy-ODE (other Levy BCs)
    ritz_solver.py                # RitzSolver        – Rayleigh-Ritz with beam eigenfunctions
    beam_functions.py             # beam_function(), get_eigenvalue(), integrals for Ritz
  validation/
    benchmarks.py                 # Benchmarks.DATA – Timoshenko/Szilard coefficients (nu=0.3)
    test_solvers.py               # legacy validation script (not collected by pytest run on tests/)
  report.py                       # ReportInputs dataclass, generate_point_report(), CLI (main)
  figures.py                      # matplotlib contour/profile plots (PNG, DPI 200)
  geometry_diagram.py             # plate sketch with BC symbols and load marker
  appendix.py                     # LaTeX appendix: worked Ritz/Levy calc, K matrix, F vector
tests/                            # pytest suite (249 tests) + standalone audit scripts
docs/                             # reference papers (PDF), implementation plan, progress notes
report_output/                    # sample generated report (.tex/.pdf) and figures
README.md, PROGRESS.md            # user docs and dated changelog
AUDIT_REPORT.md, CODE_REVIEW_SUMMARY.md, Gemini_FIT_Opinions.md   # historical review notes
```

## Setup and commands

```bash
pip install -r requirements.txt pytest     # numpy, scipy, matplotlib (+ pytest for tests)

# Full test suite (~30 s). Use a non-interactive matplotlib backend in headless envs.
MPLBACKEND=Agg python -m pytest tests/ -q

# A single file / test
MPLBACKEND=Agg python -m pytest tests/test_solver_matrix.py -q
MPLBACKEND=Agg python -m pytest "tests/test_solver_matrix.py::test_ritz_bc_load" -q

# GUI (needs a display + tkinter)
python plate-bending-gui.py

# Report CLI (writes .tex; --compile needs `tectonic` on PATH)
python -m plate_bending.report --bc SSSS --load uniform --q 1000 \
  --a 1.0 --b 1.0 --h 0.01 --E 200e9 --nu 0.3 --method levy --output report.tex
```

Notes:
- There is no linter/formatter config, no `pyproject.toml`/`setup.py`, and no CI. Run
  pytest from the **repo root** so `plate_bending` is importable.
- `tests/test_patch_load.py` runs module-level code that overwrites
  `test_patch_load_results.png` in the repo root. Revert it (`git checkout
  test_patch_load_results.png`) rather than committing the regenerated image.
- `tests/deep_audit.py`, `tests/run_validation.py`, `tests/test_gui_imports.py` are
  script-style checks; the real assertions live in the `test_*` pytest files.
- Warnings about `invalid value encountered in divide` come from the deliberate
  `test_zero_thickness` edge case and are expected.

## Conventions

### Boundary-condition strings
Four characters, edge order **x=0, y=0, x=a, y=b**; `S` = simply supported,
`C` = clamped, `F` = free. Example: `SCSF` = SS at x=0 and x=a, clamped at y=0, free at y=b.

- **Levy-type BCs** (`S?S?` — both x-edges simply supported): `SSSS, SCSC, SCSS, SCSF,
  SSSF, SFSF`. Supported by Levy, FIT and Ritz.
- **Non-Levy BCs** (`CCCC, FCFC, CCCF, FCCC`): **Ritz only**.
- `StableLevySolver` takes `bc_y0`, `bc_yb` separately (x-edges are implicitly S);
  `FITSolver` and `RitzSolver` take the full 4-char `bc` string.
- The GUI only exposes the y-edge choices (Levy-type BCs). Non-Levy BCs are available via
  the API and report CLI.

### Load types
`load_type` is one of `'uniform'`, `'rect_patch'` (`x1, y1, x2, y2`), `'circular'`
(`x0, y0, R`), `'point'` (`x0, y0`; `q0` is a **force**, not a pressure).
**`RitzSolver` does not support `'point'`** (raises `ValueError`) despite what README's
method table says — tests use `RITZ_LOAD_TYPES = ['uniform', 'rect_patch', 'circular']`.

### Solver API contract
All solvers: `Solver(a, b, h, E, nu, ...)` then `.solve(load_type, q0, **geometry)`, which
returns (and stores in `self.results`) a dict with at least:
`X, Y` (61×61 meshgrid, indexed `[iy, ix]`), `W, Mx, My, Mxy, sigma_x, sigma_y, tau_xy,
von_mises, W_max, W_max_loc, Mx_max, My_max, vm_max, W_coef, Mx_coef, My_coef,
W_center, W_center_coef, q0, load_type, method`.
- Non-dimensional coefficients: `W_coef = W_max·D/(q0·a⁴)`, `M_coef = M_max/(q0·a²)` —
  these are what get compared to `Benchmarks.DATA`.
- Moments/stresses are derived from `W` by finite differences (`np.gradient`) in each
  solver's `_compute_derived()`; keep the three implementations consistent.
- `D = E·h³ / (12·(1−ν²))`; bending stress `σ = 6M/h²`.
- `FITSolver.solve(..., auto_converge=True)` by default; the report uses
  `auto_converge=False` for reproducible term counts.
- `RitzSolver` keeps `_K`, `_F`, `coeffs` after solving — `appendix.py` reads these.
- If you add a result key or change a signature, update **all three solvers**, the GUI,
  `report.py`, and the tests that consume it.

### Units
Everything is **SI internally** (m, Pa, N). Imperial (in, psi, lbf) is handled only at the
I/O boundary: `report.main()` converts CLI inputs to SI, and `UNIT_SYSTEMS` in
`report.py` / helpers in `figures.py` and `appendix.py` convert for display. Don't
introduce imperial values into solver code.

### Numerical stability details (don't "simplify" these away)
- Levy/FIT use the **stable exponential basis** `e^{-αm·y}`, `y·e^{-αm·y}`,
  `e^{-αm(b−y)}`, `(b−y)·e^{-αm(b−y)}` instead of `sinh/cosh`, to avoid overflow at high
  `m`.
- Free-edge conditions (sign was a past bug — keep the minus signs):
  moment `Y'' − ν·αm²·Y = 0`, shear `Y''' − (2−ν)·αm²·Y' = 0`.
- Reversed beam functions (`FC`, `SC`, `FS`) are coordinate reflections of `CF`, `CS`,
  `SF` (`ξ → 1−ξ`, with odd derivatives changing sign). This was another past bug.
- Patch loads in Levy/FIT are solved piecewise in y (3 regions) with continuity
  conditions.

### Report generation
`generate_point_report(inputs: ReportInputs)` returns `(latex_str, results)`. Sections
are built by small `_section` helpers in `report.py` returning lists of LaTeX lines; the
appendix comes from `appendix.py`, figures from `figures.py` / `geometry_diagram.py`.
Number formatting goes through `_fmt` / `_fmt_plain` (report) and `fmt_eng` (appendix) —
use them instead of raw f-strings so output stays free of `e+05`-style notation.

## Validation expectations

Reference results (a=b, ν=0.3) the suite relies on:

| BC   | Reference            | Expected agreement |
|------|----------------------|--------------------|
| SSSS | Timoshenko Table 8   | ~0.06% (W_center_coef 0.00406) |
| SSSF | Timoshenko Table 48  | ~0.06% |
| SCSC | Timoshenko Table 36  | ~0.1%  |
| CCCC | Timoshenko Table 35  | ~0.7% (Ritz) |
| FCFC, CCCF | Ritz convergence | <0.5% change with more terms |

Levy and FIT should agree to ~0.00% for uniform loads on all Levy-type BCs (~0.3% for
patch/circular). Ritz typically lands within ~1–5% of Levy. The SCSF benchmark (0.01124)
is from this project's own Levy solver; Szilard's 0.01377 was judged inconsistent — see
`AUDIT_REPORT.md` and `docs/PROGRESS_NOTES.txt` before touching it.

When changing solver math:
1. Run the full pytest suite.
2. Check against `Benchmarks.DATA` and cross-solver agreement (Levy vs FIT vs Ritz).
3. Sanity-check physics: deflection ordering (more constraint → less deflection,
   e.g. `CCCC < SCSC < SSSS < SCSF < SSSF < SFSF`), symmetry, linearity in `q0`,
   max deflection on the free edge for plates with a free edge.

## Workflow

- Commit messages follow a light Conventional Commits style: `feat: …`, `fix: …`,
  `test: …`, `docs: …`.
- Add/extend pytest tests in `tests/` for any new BC, load type, or report feature.
  `tests/test_solver_matrix.py` is the parametrised BC × load matrix — extend its lists
  (`LEVY_BCS`, `NON_LEVY_BCS`, `LOAD_TYPES`, `RITZ_LOAD_TYPES`) when adding support.
- Add a dated entry to `PROGRESS.md` for substantive features or math fixes, and keep
  README's BC/load tables in sync.
- Don't commit generated scratch output (regenerated PNGs, ad-hoc `.tex/.pdf`) unless the
  change is intentionally updating `report_output/` samples.
