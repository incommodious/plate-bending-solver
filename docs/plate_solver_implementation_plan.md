# Plate Bending Solver Implementation Plan
## SCSF and SSSF with Patch Loads - Lévy + Ritz Comparison

---

## Executive Summary

**Key Insight**: SCSF and SSSF are **Lévy-type boundary conditions** (two opposite edges simply supported), meaning exact single-series solutions exist and are computationally superior to double-series Ritz. However, Ritz provides valuable independent validation.

**Recommended Strategy**:
1. **Primary Solver**: Fix and extend Lévy method for patch loads (faster, more accurate)
2. **Validation Solver**: Implement Ritz method for comparison (general, independent check)
3. **GUI Integration**: Side-by-side comparison with benchmarks

---

## Phase 1: Fix Existing Lévy Solver (2-3 hours)

### 1.1 Sign Error Corrections

**Location**: `StableLevySolver._solve_stable_ode()`, lines ~172-177 and ~197-203

**Current (WRONG)**:
```python
elif self.bc_y0 == 'F':  # Free edge
    # Y''(0) + ν·α²·Y(0) = 0  ← WRONG SIGN
    M[0] = [psi1_0_pp + nu*am**2*psi1_0, ...]
    rhs[0] = -nu * am**2 * Yp
    # Y'''(0) + (2-ν)·α²·Y'(0) = 0  ← WRONG SIGN
    M[1] = [psi1_0_ppp + (2-nu)*am**2*psi1_0_p, ...]
```

**Corrected**:
```python
elif self.bc_y0 == 'F':  # Free: M_y = 0, V_y = 0
    # Moment: Y''(0) - ν·α²·Y(0) = 0
    M[0] = [psi1_0_pp - nu*am**2*psi1_0, 
            psi2_0_pp - nu*am**2*psi2_0,
            psi3_0_pp - nu*am**2*psi3_0, 
            psi4_0_pp - nu*am**2*psi4_0]
    rhs[0] = nu * am**2 * Yp  # Sign flip on RHS too
    
    # Shear: Y'''(0) - (2-ν)·α²·Y'(0) = 0
    M[1] = [psi1_0_ppp - (2-nu)*am**2*psi1_0_p, 
            psi2_0_ppp - (2-nu)*am**2*psi2_0_p,
            psi3_0_ppp - (2-nu)*am**2*psi3_0_p, 
            psi4_0_ppp - (2-nu)*am**2*psi4_0_p]
    rhs[1] = 0
```

**Same fix needed for `bc_yb == 'F'` section.**

### 1.2 Validation After Fix

Run validation suite with uniform load:
- SCSF: Compare to Timoshenko/Szilard W_coef ≈ 0.01377
- SSSF: Compare to Timoshenko W_coef ≈ 0.01286
- SSSS: Should still match 0.00406 (regression check)

---

## Phase 2: Add Patch Load Support to Lévy (4-6 hours)

### 2.1 The Mathematical Challenge

Current code computes `Yp = qm / (D * am**4)` - a **constant** particular solution valid only for uniform loads. For a patch load centered at (x₀, y₀), the load in the Lévy framework becomes:

```
q(x,y) = q₀ · H(x,y)  where H is the patch indicator function

After x-transform:
qm(y) = (2/a) ∫₀ᵃ q(x,y) sin(mπx/a) dx
```

For a **rectangular patch** [x₁,x₂] × [y₁,y₂]:
```
qm(y) = (2q₀/a) · [a/(mπ)] · [cos(mπx₁/a) - cos(mπx₂/a)] · H(y₁ ≤ y ≤ y₂)
      = Qm · rect(y; y₁, y₂)
```

This is a **piecewise constant** in y, requiring a **piecewise particular solution**.

### 2.2 Piecewise Particular Solution Method

**Divide the y-domain into three regions**:
- Region I:   0 ≤ y < y₁  (no load)
- Region II:  y₁ ≤ y ≤ y₂  (loaded)
- Region III: y₂ < y ≤ b  (no load)

**In each region**, the ODE is:
```
Y''''(y) - 2α²Y''(y) + α⁴Y(y) = qm(y)/D
```

**Particular solutions**:
- Regions I, III: Yp = 0
- Region II: Yp = Qm / (D·α⁴)

**General solution in each region** (using stable basis):
```
Y_I(y)   = A₁·ψ₁(y) + B₁·ψ₂(y) + C₁·ψ₃(y) + D₁·ψ₄(y)
Y_II(y)  = A₂·ψ₁(y) + B₂·ψ₂(y) + C₂·ψ₃(y) + D₂·ψ₄(y) + Yp
Y_III(y) = A₃·ψ₁(y) + B₃·ψ₂(y) + C₃·ψ₃(y) + D₃·ψ₄(y)
```

**12 unknowns** require **12 equations**:
- 4 BCs at y = 0 and y = b (from plate edges)
- 4 continuity conditions at y = y₁ (Y, Y', Y'', Y''' continuous)
- 4 continuity conditions at y = y₂

### 2.3 Implementation Structure

```python
def _solve_patch_load_ode(self, am, Qm, y, y1, y2):
    """
    Solve Lévy ODE with rectangular patch load in [y1, y2].
    Returns Ym(y) evaluated at all y points.
    """
    b = self.b
    nu = self.nu
    D = self.D
    
    # Particular solution magnitude in loaded region
    Yp = Qm / (D * am**4)
    
    # Build 12x12 system
    M = np.zeros((12, 12))
    rhs = np.zeros(12)
    
    # === Boundary conditions at y = 0 (rows 0-1) ===
    # These use Region I coefficients (indices 0-3)
    self._fill_bc_y0(M, rhs, am, 0, slice(0,4), Yp_val=0)
    
    # === Continuity at y = y1 (rows 2-5) ===
    # Region I (coeffs 0-3) = Region II (coeffs 4-7) + Yp
    self._fill_continuity(M, rhs, am, y1, 2, slice(0,4), slice(4,8), Yp)
    
    # === Continuity at y = y2 (rows 6-9) ===
    # Region II (coeffs 4-7) + Yp = Region III (coeffs 8-12)
    self._fill_continuity(M, rhs, am, y2, 6, slice(4,8), slice(8,12), -Yp)
    
    # === Boundary conditions at y = b (rows 10-11) ===
    # These use Region III coefficients (indices 8-11)
    self._fill_bc_yb(M, rhs, am, 10, slice(8,12), Yp_val=0)
    
    # Solve 12x12 system
    coeffs = np.linalg.solve(M, rhs)
    
    # Evaluate Ym(y) in each region
    Ym = np.zeros_like(y)
    for i, yi in enumerate(y):
        if yi < y1:
            Ym[i] = self._eval_Y(yi, am, coeffs[0:4], 0)
        elif yi <= y2:
            Ym[i] = self._eval_Y(yi, am, coeffs[4:8], Yp)
        else:
            Ym[i] = self._eval_Y(yi, am, coeffs[8:12], 0)
    
    return Ym
```

### 2.4 Circular Patch Approximation

For circular patches, use **equivalent rectangular approximation**:
```python
def _circular_to_rect_patch(self, x0, y0, R):
    """Convert circular patch to equivalent rectangular patch."""
    # Option 1: Equivalent area square
    s = np.sqrt(np.pi) * R  # side length
    x1, x2 = x0 - s/2, x0 + s/2
    y1, y2 = y0 - s/2, y0 + s/2
    
    # Clip to plate boundaries
    x1, x2 = max(0, x1), min(self.a, x2)
    y1, y2 = max(0, y1), min(self.b, y2)
    
    return x1, y1, x2, y2
```

**Better option**: Numerical integration of qm(y) for circular patch, then use piecewise polynomial particular solution. This is more accurate but more complex.

---

## Phase 3: Implement Ritz Solver (6-8 hours)

### 3.1 Class Structure

```python
class RitzSolver:
    """
    Rayleigh-Ritz solver for rectangular plate bending.
    
    Approximation: w(x,y) = Σₘ Σₙ Aₘₙ · Xₘ(x) · Yₙ(y)
    
    Supports:
    - SSSS, SCSF, SSSF, SCSC boundary conditions
    - Uniform, rectangular patch, and circular patch loads
    """
    
    def __init__(self, a, b, h, E, nu, bc='SCSF', M=10, N=10):
        self.a, self.b, self.h = a, b, h
        self.E, self.nu = E, nu
        self.D = E * h**3 / (12 * (1 - nu**2))
        self.bc = bc.upper()
        self.M, self.N = M, N  # Number of terms in each direction
        self.ndof = M * N
        
        # Parse boundary conditions
        self.bc_x0 = bc[0]  # S, C, or F
        self.bc_y0 = bc[1]
        self.bc_xa = bc[2]
        self.bc_yb = bc[3]
        
        # Precompute eigenvalues
        self._compute_eigenvalues()
        
        # Precompute beam function integrals
        self._compute_integrals()
```

### 3.2 Beam Function Definitions

```python
# Eigenvalues for different BC types
EIGENVALUES = {
    'SS': lambda n: n * np.pi,  # Simply supported both ends
    'CF': [1.8751, 4.6941, 7.8548, 10.996, 14.137],  # Clamped-Free
    'CS': [3.9266, 7.0686, 10.210, 13.352, 16.493],  # Clamped-Simply supported
    'SF': [3.9266, 7.0686, 10.210, 13.352, 16.493],  # Simply supported-Free (same roots as CS)
    'CC': [4.7300, 7.8532, 10.996, 14.137, 17.279],  # Clamped-Clamped
    'FF': [4.7300, 7.8532, 10.996, 14.137, 17.279],  # Free-Free (same roots as CC)
}

def _beam_function(self, xi, n, bc_type):
    """
    Evaluate n-th beam function at normalized coordinate xi ∈ [0,1].
    
    Parameters:
        xi: normalized coordinate x/L
        n: mode number (1-indexed)
        bc_type: 'SS', 'CF', 'CS', 'SF', 'CC', 'FF'
    """
    if bc_type == 'SS':
        return np.sin(n * np.pi * xi)
    
    # Get eigenvalue
    beta = self._get_eigenvalue(n, bc_type)
    
    if bc_type == 'CF':  # Clamped-Free (cantilever)
        sigma = (np.sinh(beta) - np.sin(beta)) / (np.cosh(beta) + np.cos(beta))
        return (np.cosh(beta*xi) - np.cos(beta*xi) - 
                sigma * (np.sinh(beta*xi) - np.sin(beta*xi)))
    
    elif bc_type == 'CS':  # Clamped-Simply supported
        sigma = (np.sinh(beta) - np.sin(beta)) / (np.cosh(beta) - np.cos(beta))
        return (np.sinh(beta*xi) - np.sin(beta*xi) - 
                sigma * (np.cosh(beta*xi) - np.cos(beta*xi)))
    
    elif bc_type == 'SF':  # Simply supported-Free
        ratio = np.sin(beta) / np.sinh(beta)
        return np.sin(beta*xi) - ratio * np.sinh(beta*xi)
    
    elif bc_type == 'CC':  # Clamped-Clamped
        sigma = (np.cosh(beta) - np.cos(beta)) / (np.sinh(beta) - np.sin(beta))
        return (np.cosh(beta*xi) - np.cos(beta*xi) - 
                sigma * (np.sinh(beta*xi) - np.sin(beta*xi)))
    
    elif bc_type == 'FF':  # Free-Free
        sigma = (np.cosh(beta) - np.cos(beta)) / (np.sinh(beta) - np.sin(beta))
        return (np.cosh(beta*xi) + np.cos(beta*xi) - 
                sigma * (np.sinh(beta*xi) + np.sin(beta*xi)))

def _get_eigenvalue(self, n, bc_type):
    """Get n-th eigenvalue for given BC type."""
    if bc_type == 'SS':
        return n * np.pi
    
    eigenvals = EIGENVALUES[bc_type]
    if n <= len(eigenvals):
        return eigenvals[n-1]
    else:
        # Asymptotic approximation for higher modes
        if bc_type in ['CF']:
            return (2*n - 1) * np.pi / 2
        elif bc_type in ['CS', 'SF']:
            return (4*n + 1) * np.pi / 4
        elif bc_type in ['CC', 'FF']:
            return (2*n + 1) * np.pi / 2
```

### 3.3 Stiffness Matrix Assembly

```python
def _compute_integrals(self):
    """
    Precompute beam function integrals using numerical quadrature.
    
    For each direction, compute:
    - I0[m,p] = ∫ Xm(x) Xp(x) dx  (mass-like)
    - I1[m,p] = ∫ Xm'(x) Xp'(x) dx  (first derivative)
    - I2[m,p] = ∫ Xm''(x) Xp''(x) dx  (curvature)
    """
    from scipy.integrate import quad
    
    # X-direction integrals
    bc_x = self._get_beam_bc('x')
    self.Ix0 = np.zeros((self.M, self.M))
    self.Ix1 = np.zeros((self.M, self.M))
    self.Ix2 = np.zeros((self.M, self.M))
    
    for m in range(1, self.M + 1):
        for p in range(1, self.M + 1):
            # Use symmetry: only compute upper triangle
            if p < m:
                self.Ix0[m-1, p-1] = self.Ix0[p-1, m-1]
                self.Ix1[m-1, p-1] = self.Ix1[p-1, m-1]
                self.Ix2[m-1, p-1] = self.Ix2[p-1, m-1]
                continue
            
            # I0: ∫ Xm Xp dx
            def f0(xi):
                return (self._beam_function(xi, m, bc_x) * 
                        self._beam_function(xi, p, bc_x))
            self.Ix0[m-1, p-1], _ = quad(f0, 0, 1, limit=100)
            self.Ix0[m-1, p-1] *= self.a  # Scale by length
            
            # I1: ∫ Xm' Xp' dx (use numerical differentiation)
            def f1(xi):
                return (self._beam_deriv(xi, m, bc_x, order=1) * 
                        self._beam_deriv(xi, p, bc_x, order=1))
            self.Ix1[m-1, p-1], _ = quad(f1, 0, 1, limit=100)
            self.Ix1[m-1, p-1] /= self.a  # Scale by 1/length
            
            # I2: ∫ Xm'' Xp'' dx
            def f2(xi):
                return (self._beam_deriv(xi, m, bc_x, order=2) * 
                        self._beam_deriv(xi, p, bc_x, order=2))
            self.Ix2[m-1, p-1], _ = quad(f2, 0, 1, limit=100)
            self.Ix2[m-1, p-1] /= self.a**3  # Scale by 1/length³
    
    # Y-direction integrals (same structure)
    bc_y = self._get_beam_bc('y')
    self.Iy0 = np.zeros((self.N, self.N))
    self.Iy1 = np.zeros((self.N, self.N))
    self.Iy2 = np.zeros((self.N, self.N))
    # ... similar computation ...

def _assemble_stiffness(self):
    """
    Assemble global stiffness matrix K.
    
    K[i,j] where i = (m-1)*N + (n-1), j = (p-1)*N + (q-1)
    
    From strain energy:
    K_mnpq = D * ∫∫ [Xm''Xp'' YnYq + XmXp Yn''Yq'' + 2(1-ν) Xm'Xp' Yn'Yq'] dA
           = D * [Ix2[m,p]*Iy0[n,q] + Ix0[m,p]*Iy2[n,q] + 2(1-ν)*Ix1[m,p]*Iy1[n,q]]
    """
    K = np.zeros((self.ndof, self.ndof))
    D, nu = self.D, self.nu
    
    for m in range(1, self.M + 1):
        for n in range(1, self.N + 1):
            i = (m-1) * self.N + (n-1)
            
            for p in range(1, self.M + 1):
                for q in range(1, self.N + 1):
                    j = (p-1) * self.N + (q-1)
                    
                    # Skip lower triangle (symmetric)
                    if j < i:
                        continue
                    
                    # Stiffness contribution
                    term1 = self.Ix2[m-1,p-1] * self.Iy0[n-1,q-1]  # ∂⁴w/∂x⁴
                    term2 = self.Ix0[m-1,p-1] * self.Iy2[n-1,q-1]  # ∂⁴w/∂y⁴
                    term3 = 2 * self.Ix1[m-1,p-1] * self.Iy1[n-1,q-1]  # ∂⁴w/∂x²∂y²
                    
                    K[i,j] = D * (term1 + term2 + term3)
                    K[j,i] = K[i,j]  # Symmetric
    
    return K
```

### 3.4 Load Vector Assembly

```python
def _assemble_load_uniform(self, q0):
    """Load vector for uniform pressure q0."""
    F = np.zeros(self.ndof)
    
    bc_x = self._get_beam_bc('x')
    bc_y = self._get_beam_bc('y')
    
    for m in range(1, self.M + 1):
        for n in range(1, self.N + 1):
            i = (m-1) * self.N + (n-1)
            
            # F_mn = q0 * ∫∫ Xm(x) Yn(y) dA
            Ix = quad(lambda xi: self._beam_function(xi, m, bc_x), 0, 1)[0] * self.a
            Iy = quad(lambda eta: self._beam_function(eta, n, bc_y), 0, 1)[0] * self.b
            
            F[i] = q0 * Ix * Iy
    
    return F

def _assemble_load_rect_patch(self, q0, x1, y1, x2, y2):
    """Load vector for rectangular patch load."""
    F = np.zeros(self.ndof)
    
    bc_x = self._get_beam_bc('x')
    bc_y = self._get_beam_bc('y')
    
    # Normalized coordinates
    xi1, xi2 = x1/self.a, x2/self.a
    eta1, eta2 = y1/self.b, y2/self.b
    
    for m in range(1, self.M + 1):
        for n in range(1, self.N + 1):
            i = (m-1) * self.N + (n-1)
            
            # F_mn = q0 * ∫∫_patch Xm(x) Yn(y) dA
            Ix = quad(lambda xi: self._beam_function(xi, m, bc_x), xi1, xi2)[0] * self.a
            Iy = quad(lambda eta: self._beam_function(eta, n, bc_y), eta1, eta2)[0] * self.b
            
            F[i] = q0 * Ix * Iy
    
    return F

def _assemble_load_circular_patch(self, q0, xc, yc, R):
    """Load vector for circular patch load using polar integration."""
    from scipy.integrate import dblquad
    
    F = np.zeros(self.ndof)
    
    bc_x = self._get_beam_bc('x')
    bc_y = self._get_beam_bc('y')
    
    for m in range(1, self.M + 1):
        for n in range(1, self.N + 1):
            i = (m-1) * self.N + (n-1)
            
            def integrand(theta, r):
                x = xc + r * np.cos(theta)
                y = yc + r * np.sin(theta)
                
                # Check bounds
                if x < 0 or x > self.a or y < 0 or y > self.b:
                    return 0.0
                
                xi = x / self.a
                eta = y / self.b
                
                return r * self._beam_function(xi, m, bc_x) * self._beam_function(eta, n, bc_y)
            
            result, _ = dblquad(integrand, 0, R, 0, 2*np.pi, epsabs=1e-8)
            F[i] = q0 * result
    
    return F
```

### 3.5 Solution and Post-Processing

```python
def solve(self, load_type='uniform', q0=1.0, x0=None, y0=None, 
          x1=None, y1=None, x2=None, y2=None, R=None):
    """
    Solve for deflection coefficients.
    
    Returns dict with results.
    """
    # Assemble stiffness
    K = self._assemble_stiffness()
    
    # Assemble load vector
    if load_type == 'uniform':
        F = self._assemble_load_uniform(q0)
    elif load_type == 'rect_patch':
        F = self._assemble_load_rect_patch(q0, x1, y1, x2, y2)
    elif load_type == 'circular':
        if R is None:
            R = min(self.a, self.b) / 10
        if x0 is None:
            x0 = self.a / 2
        if y0 is None:
            y0 = self.b / 2
        F = self._assemble_load_circular_patch(q0, x0, y0, R)
    
    # Solve K·A = F
    A = np.linalg.solve(K, F)
    
    # Store coefficients
    self.coeffs = A.reshape((self.M, self.N))
    
    # Compute deflection field
    nx, ny = 61, 61
    x = np.linspace(0, self.a, nx)
    y = np.linspace(0, self.b, ny)
    X, Y = np.meshgrid(x, y)
    W = self._compute_deflection_field(X, Y)
    
    # Compute derived quantities
    results = self._compute_results(X, Y, W, q0)
    results['method'] = 'Ritz'
    results['load_type'] = load_type
    results['q0'] = q0
    results['n_terms'] = (self.M, self.N)
    
    return results

def _compute_deflection_field(self, X, Y):
    """Evaluate w(x,y) on grid."""
    bc_x = self._get_beam_bc('x')
    bc_y = self._get_beam_bc('y')
    
    W = np.zeros_like(X)
    
    for m in range(1, self.M + 1):
        for n in range(1, self.N + 1):
            Amn = self.coeffs[m-1, n-1]
            
            # Evaluate basis functions on grid
            Xi = X / self.a
            Eta = Y / self.b
            
            Xm = np.vectorize(lambda xi: self._beam_function(xi, m, bc_x))(Xi)
            Yn = np.vectorize(lambda eta: self._beam_function(eta, n, bc_y))(Eta)
            
            W += Amn * Xm * Yn
    
    return W
```

---

## Phase 4: GUI Integration (3-4 hours)

### 4.1 Updated GUI Structure

```python
class TripleMethodGUI:
    """
    Extended GUI with three solver methods:
    1. Lévy (primary, fast, exact for Lévy-type BCs)
    2. FIT (approximate, for comparison)
    3. Ritz (general, independent validation)
    """
    
    def __init__(self):
        # ... existing setup ...
        
        # Add Ritz solver controls
        self.ritz_enabled = tk.BooleanVar(value=True)
        self.ritz_terms_M = tk.StringVar(value='10')
        self.ritz_terms_N = tk.StringVar(value='10')
```

### 4.2 Comparison Tab Enhancements

```python
def _update_comparison(self):
    """Update comparison display with three methods."""
    self.comp_text.delete(1.0, tk.END)
    
    L = self.levy_results
    F = self.fit_results
    R = self.ritz_results  # NEW
    bc = f"S{self.bc_y0.get()}S{self.bc_yb.get()}"
    
    bench = Benchmarks.get(bc)
    
    txt = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  TRIPLE METHOD COMPARISON: {bc}                                                           ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                           │   LÉVY METHOD   │   FIT METHOD    │   RITZ METHOD   │ BENCH  ║
╠═══════════════════════════╪═════════════════╪═════════════════╪═════════════════╪════════╣
║  Max |W| (mm)             │  {L['W_max']*1e3:13.6f}  │  {F['W_max']*1e3:13.6f}  │  {R['W_max']*1e3:13.6f}  │        ║
║  W coefficient (W·D/qa⁴)  │  {L['W_coef']:13.6f}  │  {F['W_coef']:13.6f}  │  {R['W_coef']:13.6f}  │ {bench.get('W_max_coef', 'N/A'):6} ║
║  Max |Mx| (N·m/m)         │  {L['Mx_max']:13.4e}  │  {F['Mx_max']:13.4e}  │  {R['Mx_max']:13.4e}  │        ║
║  Max Von Mises (MPa)      │  {L['vm_max']/1e6:13.4f}  │  {F['vm_max']/1e6:13.4f}  │  {R['vm_max']/1e6:13.4f}  │        ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
    self.comp_text.insert(tk.END, txt)
```

### 4.3 New Visualization Options

```python
def _update_plot(self):
    # ... existing code ...
    
    if p == 'W_three':  # New option: all three methods
        ax1 = self.fig.add_subplot(131)
        c1 = ax1.contourf(L['X'], L['Y'], L['W']*1e3, levels=25, cmap='RdYlBu_r')
        ax1.set_title(f"LÉVY: {L['W_max']*1e3:.4f} mm")
        
        ax2 = self.fig.add_subplot(132)
        c2 = ax2.contourf(F['X'], F['Y'], F['W']*1e3, levels=25, cmap='RdYlBu_r')
        ax2.set_title(f"FIT: {F['W_max']*1e3:.4f} mm")
        
        ax3 = self.fig.add_subplot(133)
        c3 = ax3.contourf(R['X'], R['Y'], R['W']*1e3, levels=25, cmap='RdYlBu_r')
        ax3.set_title(f"RITZ: {R['W_max']*1e3:.4f} mm")
    
    elif p == 'convergence':  # Ritz convergence study
        self._plot_ritz_convergence()
```

---

## Phase 5: Testing & Validation (2-3 hours)

### 5.1 Unit Tests

```python
def test_levy_sign_fix():
    """Verify sign fix doesn't break SSSS case."""
    solver = StableLevySolver(1, 1, 0.01, 2.1e11, 0.3, 'S', 'S', 100)
    results = solver.solve('uniform', 10000)
    
    expected = 0.00406
    actual = results['W_coef']
    assert abs(actual - expected) / expected < 0.01, f"SSSS failed: {actual} vs {expected}"

def test_levy_scsf_uniform():
    """Verify SCSF with uniform load matches Timoshenko."""
    solver = StableLevySolver(1, 1, 0.01, 2.1e11, 0.3, 'C', 'F', 100)
    results = solver.solve('uniform', 10000)
    
    expected = 0.01377
    actual = results['W_coef']
    assert abs(actual - expected) / expected < 0.05, f"SCSF failed: {actual} vs {expected}"

def test_ritz_ssss_uniform():
    """Verify Ritz matches Navier for SSSS."""
    solver = RitzSolver(1, 1, 0.01, 2.1e11, 0.3, 'SSSS', 10, 10)
    results = solver.solve('uniform', 10000)
    
    expected = 0.00406
    actual = results['W_coef']
    assert abs(actual - expected) / expected < 0.02, f"Ritz SSSS failed: {actual} vs {expected}"

def test_ritz_convergence():
    """Verify Ritz converges with increasing terms."""
    prev_w = None
    for N in [3, 5, 8, 12]:
        solver = RitzSolver(1, 1, 0.01, 2.1e11, 0.3, 'SSSS', N, N)
        results = solver.solve('uniform', 10000)
        w = results['W_max']
        
        if prev_w is not None:
            change = abs(w - prev_w) / prev_w
            print(f"N={N}: W_max={w:.6e}, change={change:.2%}")
        prev_w = w

def test_levy_vs_ritz_scsf():
    """Cross-validate Lévy and Ritz for SCSF uniform load."""
    levy = StableLevySolver(1, 1, 0.01, 2.1e11, 0.3, 'C', 'F', 100)
    levy_results = levy.solve('uniform', 10000)
    
    ritz = RitzSolver(1, 1, 0.01, 2.1e11, 0.3, 'SCSF', 15, 15)
    ritz_results = ritz.solve('uniform', 10000)
    
    diff = abs(levy_results['W_coef'] - ritz_results['W_coef'])
    rel_diff = diff / levy_results['W_coef']
    
    print(f"Lévy: {levy_results['W_coef']:.6f}")
    print(f"Ritz: {ritz_results['W_coef']:.6f}")
    print(f"Difference: {rel_diff:.2%}")
    
    assert rel_diff < 0.03, "Lévy and Ritz disagree by more than 3%"
```

### 5.2 Benchmark Comparison Table

| BC | Load | Lévy (expected) | Ritz (expected) | Source |
|----|------|-----------------|-----------------|--------|
| SSSS | Uniform | 0.00406 | 0.00406 | Timoshenko Table 8 |
| SCSF | Uniform | 0.01377 | 0.01377 | Szilard Table 5.9 |
| SSSF | Uniform | 0.01286 | 0.01286 | Timoshenko Table 48 |
| SSSS | Central patch | ~0.0041 | ~0.0041 | FEM validation |
| SCSF | Edge patch | — | — | FEM validation needed |

---

## Implementation Timeline

| Phase | Task | Estimated Time | Dependencies |
|-------|------|----------------|--------------|
| 1.1 | Fix Lévy sign errors | 1 hour | None |
| 1.2 | Validate fix | 1 hour | 1.1 |
| 2.1-2.2 | Design piecewise particular solution | 2 hours | 1.2 |
| 2.3-2.4 | Implement patch load in Lévy | 3 hours | 2.1-2.2 |
| 3.1-3.2 | Ritz class structure + beam functions | 2 hours | None |
| 3.3-3.4 | Stiffness and load assembly | 3 hours | 3.1-3.2 |
| 3.5 | Solution and post-processing | 2 hours | 3.3-3.4 |
| 4.1-4.3 | GUI integration | 3 hours | 2.3-2.4, 3.5 |
| 5.1-5.2 | Testing and validation | 2 hours | 4.1-4.3 |

**Total: ~19-21 hours**

---

## File Structure

```
plate_bending/
├── solvers/
│   ├── __init__.py
│   ├── levy_solver.py      # Fixed StableLevySolver with patch load
│   ├── fit_solver.py       # Existing FITSolver (minor fixes)
│   ├── ritz_solver.py      # NEW RitzSolver
│   └── beam_functions.py   # Beam function library
├── gui/
│   ├── __init__.py
│   ├── main_gui.py         # TripleMethodGUI
│   └── plotting.py         # Visualization helpers
├── validation/
│   ├── benchmarks.py       # Benchmark data
│   └── test_solvers.py     # Unit tests
└── main.py                 # Entry point
```

---

## Risk Mitigation

1. **Numerical stability in Ritz**: Beam functions can overflow for high mode numbers
   - *Mitigation*: Normalize beam functions, use moderate term counts (M,N ≤ 20)

2. **Slow Ritz performance**: Numerical integration is expensive
   - *Mitigation*: Cache integrals, use analytical forms where available (SS edges)

3. **Lévy patch load complexity**: 12×12 system may have conditioning issues
   - *Mitigation*: Use `numpy.linalg.lstsq` as fallback, monitor condition number

4. **Validation uncertainty**: Limited benchmark data for patch loads
   - *Mitigation*: Cross-validate Lévy vs Ritz; both should agree within ~2-3%

---

## Success Criteria

1. ✅ SCSF + uniform load matches Timoshenko (within 5%)
2. ✅ SSSF + uniform load matches Timoshenko (within 5%)
3. ✅ Lévy and Ritz agree within 3% for all test cases
4. ✅ Patch loads produce physically reasonable results (max deflection at load location)
5. ✅ GUI displays three-way comparison clearly
6. ✅ Convergence study shows Ritz approaching Lévy as terms increase
