# Plate Bending Solver - Development Progress

## 2026-02-01: Extended FIT Solver for Free Edge Boundary Conditions

### Problem Statement
The Finite Integral Transform (FIT) solver was showing incorrect results for SCSF (clamped at y=0, free at y=b) boundary conditions. The maximum deflection was incorrectly displayed at the center of the plate instead of near the free edge, where it should physically occur.

### Root Cause Analysis
The original FIT implementation used a standard Navier double sine series:
```
W(x,y) = sum_m sum_n W_mn * sin(m*pi*x/a) * sin(n*pi*y/b)
```

This formulation inherently forces W=0 at all four edges because:
- sin(m*pi*0/a) = 0 at x=0
- sin(m*pi*a/a) = 0 at x=a
- sin(n*pi*0/b) = 0 at y=0
- sin(n*pi*b/b) = 0 at y=b

For SSSS (all simply supported), this is correct. However, for plates with free edges (like SCSF), the free edge should have non-zero deflection, and typically has the maximum deflection.

### Solution Implemented
Based on the methodology described in Xu et al. (2020) "Analytical Bending Solutions of Orthotropic Rectangular Thin Plates with Two Adjacent Edges Free", the FIT solver was extended to properly handle all Levy-type boundary conditions.

#### Key Changes to `fit_solver.py`:

1. **Boundary Condition Parsing**: Added logic to detect boundary condition type:
   - `is_ssss`: True only for SSSS plates
   - `is_levy_type`: True for plates with simply supported x-edges (S___S_ pattern)
   - `has_free_edge` / `has_clamped_edge`: Flags for special edge conditions

2. **Solution Method Selection**:
   - `navier`: Standard double sine series for SSSS (exact, fastest)
   - `levy_ode`: Uses stable exponential basis for Levy-type plates with C/F edges

3. **Levy-ODE Method**: For non-SSSS Levy-type plates, the solver:
   - Uses standard sine series in x-direction (sin(m*pi*x/a))
   - Solves the Levy ODE in y-direction using stable exponential basis functions:
     - psi_1(y) = e^(-am*(b-y))
     - psi_2(y) = (b-y)*e^(-am*(b-y))
     - psi_3(y) = e^(-am*y)
     - psi_4(y) = y*e^(-am*y)
   - Applies proper boundary conditions for C (clamped) and F (free) edges

4. **Free Edge Conditions**: Correctly implements:
   - Moment: Y''(y) - nu*am^2*Y(y) = 0
   - Shear: Y'''(y) - (2-nu)*am^2*Y'(y) = 0

### Validation Results

**Uniform Load (all BCs):**
| BC | Levy W_max (mm) | FIT W_max (mm) | FIT Method | Difference |
|----|-----------------|----------------|------------|------------|
| SSSS | 2.1124 | 2.1124 | navier | 0.00% |
| SCSC | 0.9969 | 0.9969 | levy_ode | 0.00% |
| SCSS | 1.4856 | 1.4856 | levy_ode | 0.00% |
| SCSF | 5.8427 | 5.8427 | levy_ode | 0.00% |
| SSSF | 6.6833 | 6.6833 | levy_ode | 0.00% |
| SFSF | 7.8059 | 7.8059 | levy_ode | 0.00% |

**SCSF with Different Load Types:**
| Load Type | Levy (mm) | FIT (mm) | Ritz (mm) | FIT Diff | Ritz Diff |
|-----------|-----------|----------|-----------|----------|-----------|
| Uniform | 5.8427 | 5.8427 | 5.8035 | 0.00% | 0.67% |
| Rect Patch | 1.1062 | 1.1093 | 1.1013 | 0.28% | 0.44% |
| Circular | 0.2219 | 0.2227 | 0.2203 | 0.36% | 0.73% |

Test parameters: a=b=1.0m, h=0.01m, E=2.1e11 Pa, nu=0.3, q0=10000 Pa

### Why FIT and Levy Match for Uniform Loads

For non-SSSS Levy-type plates, both FIT (levy_ode method) and Levy use mathematically
identical approaches:
1. **Same x-direction treatment**: Fourier sine series `sin(mπx/a)`
2. **Same y-direction ODE**: 4th order equation with identical boundary conditions
3. **Same basis functions**: Stable exponential basis for numerical stability

This is expected and correct - for Levy-type plates, the Levy method IS the appropriate
"integral transform" approach. The true bidirectional FIT (Navier double sine series)
only works for SSSS plates.

Small differences (~0.3%) in patch/circular loads arise from implementation details in
handling y-direction load variations (piecewise ODE solving).

### Maximum Deflection Location
The fix correctly places maximum deflection near free edges:
- **SCSF**: Max at (0.5, 1.0) - center of free edge at y=b
- **SSSF**: Max at (0.5, 1.0) - center of free edge at y=b
- **SFSF**: Max at (0.5, 0.0) or (0.5, 1.0) - both y-edges are free

### GUI Updates
- Changed "Skip FIT for non-SSSS" checkbox to "Skip FIT solver" (now defaulted to unchecked)
- Updated FIT validity check to recognize all Levy-type BCs as valid
- Status display now shows method used (FIT vs FIT-Extended)

### Files Modified
- `plate_bending/solvers/fit_solver.py` - Complete rewrite with extended FIT support
- `plate-bending-gui.py` - Updated FIT validity logic and UI
- `README.md` - Updated method comparison and accuracy notes

### References
- Xu, Q., Yang, Z., Ullah, S., Zhang, J., & Gao, Y. (2020). "Analytical Bending Solutions of Orthotropic Rectangular Thin Plates with Two Adjacent Edges Free and the Others Clamped or Simply Supported Using Finite Integral Transform Method". *Advances in Civil Engineering*, Vol. 2020, Article ID 8848879.

---

## Previous Development History

### 2026-01-31: Free Edge Sign Corrections
- Fixed sign errors in free edge boundary conditions for Levy solver
- Corrected moment condition: Y''(y) - nu*am^2*Y(y) = 0 (was incorrectly using +)
- Corrected shear condition: Y'''(y) - (2-nu)*am^2*Y'(y) = 0 (was incorrectly using +)

### Initial Implementation
- Levy solver with stable exponential basis functions
- FIT solver with Navier double sine series (SSSS only)
- Ritz solver with beam eigenfunctions
- GUI with method comparison and visualization
