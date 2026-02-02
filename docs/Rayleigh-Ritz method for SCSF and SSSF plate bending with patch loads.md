# Rayleigh-Ritz method for SCSF and SSSF plate bending with patch loads

Both SCSF and SSSF plates are **Lévy-type boundary conditions**, meaning exact single-series solutions exist—often making full double-series Rayleigh-Ritz unnecessary. However, Ritz remains valuable for complex loadings and provides a general framework. This report delivers explicit beam functions, stiffness matrix formulations, Python implementation, and validation benchmarks for practical engineering use.

## Beam characteristic functions and their eigenvalues

The Rayleigh-Ritz method approximates plate deflection as **w(x,y) = Σₘ Σₙ Aₘₙ Xₘ(x) Yₙ(y)**, where Xₘ and Yₙ are beam functions satisfying boundary conditions on each edge. These functions come from beam vibration theory, with normalized coordinate **ξ = x/L**.

**Simply supported (SS) beam functions** take the simplest form:

φₙ(ξ) = sin(nπξ), with eigenvalues βₙL = nπ (n = 1, 2, 3,...)

**Clamped-free (cantilever) beam functions** satisfy cos(βL)cosh(βL) + 1 = 0:

φₙ(ξ) = [cosh(βₙLξ) − cos(βₙLξ)] − σₙ[sinh(βₙLξ) − sin(βₙLξ)]

where σₙ = (sinh βₙL − sin βₙL)/(cosh βₙL + cos βₙL). The first five eigenvalues are **β₁L = 1.8751, β₂L = 4.6941, β₃L = 7.8548, β₄L = 10.996, β₅L = 14.137**. For n > 3, the approximation βₙL ≈ (2n−1)π/2 holds within 0.5%.

**Simply supported-free (S-F) beam functions** satisfy tan(βL) = tanh(βL):

φₙ(x) = Aₙ[sin(βₙx) − (sin βₙL/sinh βₙL)·sinh(βₙx)]

The eigenvalues are **β₁L = 3.9266, β₂L = 7.0686, β₃L = 10.210, β₄L = 13.352, β₅L = 16.493**. These identical roots also apply to clamped-simply supported (C-S) beams, though with different eigenfunctions.

**Clamped-simply supported (C-S) beam functions** share the same characteristic equation:

φₙ(x) = [sinh(βₙx) − sin(βₙx)] − σₙ[cosh(βₙx) − cos(βₙx)]

with σₙ = (sinh βₙL − sin βₙL)/(cosh βₙL − cos βₙL).

| Boundary condition | Characteristic equation | β₁L | β₂L | β₃L | Asymptotic formula |
|--------------------|------------------------|-------|-------|-------|-------------------|
| Simply supported | sin(βL) = 0 | 3.142 | 6.283 | 9.425 | nπ |
| Clamped-free | cos(βL)cosh(βL) = −1 | 1.875 | 4.694 | 7.855 | (2n−1)π/2 |
| Clamped-clamped | cos(βL)cosh(βL) = 1 | 4.730 | 7.853 | 10.996 | (2n+1)π/2 |
| Clamped-pinned / Pinned-free | tan(βL) = tanh(βL) | 3.927 | 7.069 | 10.210 | (4n+1)π/4 |

## Stiffness matrix formulation from strain energy

The strain energy for a Kirchhoff thin plate is:

**U = (D/2) ∫∫ [(∂²w/∂x²)² + (∂²w/∂y²)² + 2ν(∂²w/∂x²)(∂²w/∂y²) + 2(1−ν)(∂²w/∂x∂y)²] dA**

where D = Eh³/12(1−ν²) is flexural rigidity. Substituting the series form and applying the Ritz minimization ∂U/∂Aₘₙ = 0 yields the linear system **K·A = F**.

The stiffness coefficient connecting modes (m,n) and (p,q) is:

**Kₘₙ,ₚq = D[I″ₘₚ J⁰ₙq/a³b + I⁰ₘₚ J″ₙq/ab³ + ν(I″ₘₚ J⁰ₙq/ab + I⁰ₘₚ J″ₙq/ab)/ab + 2(1−ν) I′ₘₚ J′ₙq/a²b²]**

where the fundamental integrals are:

- I⁰ₘₚ = ∫₀ᵃ Xₘ(x)Xₚ(x) dx (orthogonality integral)
- I′ₘₚ = ∫₀ᵃ X′ₘ(x)X′ₚ(x) dx (first derivative product)
- I″ₘₚ = ∫₀ᵃ X″ₘ(x)X″ₚ(x) dx (curvature product)

For **simply supported functions**, orthogonality gives I⁰ₘₚ = (a/2)δₘₚ, and the stiffness matrix becomes **diagonal**—a major computational simplification. For clamped or free boundaries, cross-coupling integrals are non-zero when (m+p) is even, creating a sparse but non-diagonal structure.

**Tabulated integral values** from Young & Felgar (1949) and Blevins for clamped-clamped beams (normalized to unit length):

| m | ∫φₘ²dξ | ∫(φ″ₘ)²dξ | ∫(φ′ₘ)²dξ |
|---|--------|-----------|-----------|
| 1 | 1.0 | 500.6 | 12.30 |
| 2 | 1.0 | 3803.5 | 46.05 |
| 3 | 1.0 | 14617 | 118.9 |
| 4 | 1.0 | 39944 | 231.4 |

Cross-coupling integrals ∫φ″ₘφ″ₙdξ for clamped beams: modes (1,3) give −228.7, modes (2,4) give −1869.4. These non-zero off-diagonal terms create the coupling that requires solving the full system.

## Handling rectangular and circular patch loads

For a rectangular patch load of intensity q₀ over region [x₁,x₂] × [y₁,y₂], the generalized force vector separates as:

**Fₘₙ = q₀ · ∫ˣ²ₓ₁ Xₘ(x)dx · ∫ʸ²ᵧ₁ Yₙ(y)dy**

For simply supported functions, the integral has closed form: ∫ˣ²ₓ₁ sin(mπx/a)dx = (a/mπ)[cos(mπx₁/a) − cos(mπx₂/a)].

**Circular patch loads** require numerical integration. The most efficient approach uses polar coordinates centered on the patch: x = xc + r·cos(θ), y = yc + r·sin(θ), with Jacobian r dr dθ. Integration bounds become 0 ≤ r ≤ R, 0 ≤ θ ≤ 2π, clipped to plate boundaries.

For quick estimates, an equivalent-area square approximation works: side length s = √π·R gives a square with identical area, centered at the same point.

## Python implementation for Ritz plate bending

```python
import numpy as np
from scipy.integrate import dblquad
from scipy.linalg import solve

class RitzPlateSS:
    """Rayleigh-Ritz solver for simply supported rectangular plate."""
    
    def __init__(self, a, b, h, E, nu, M, N):
        self.a, self.b, self.h = a, b, h
        self.E, self.nu = E, nu
        self.M, self.N = M, N
        self.D = E * h**3 / (12 * (1 - nu**2))
        self.ndof = M * N
    
    def idx(self, m, n):
        """Map (m,n) starting at 1 to linear index."""
        return (m - 1) * self.N + (n - 1)
    
    def mode(self, i):
        """Recover (m,n) from linear index."""
        return i // self.N + 1, i % self.N + 1
    
    def stiffness_matrix(self):
        """Diagonal stiffness for SSSS plate."""
        K = np.zeros((self.ndof, self.ndof))
        for i in range(self.ndof):
            m, n = self.mode(i)
            alpha = m * np.pi / self.a
            beta = n * np.pi / self.b
            # Analytical result exploiting orthogonality
            K[i, i] = self.D * (self.a * self.b / 4) * (alpha**2 + beta**2)**2
        return K
    
    def load_vector_rect_patch(self, q0, x1, y1, x2, y2):
        """Load vector for rectangular patch."""
        F = np.zeros(self.ndof)
        for i in range(self.ndof):
            m, n = self.mode(i)
            # Closed-form integrals of sine functions
            Ix = (self.a / (m * np.pi)) * (
                np.cos(m * np.pi * x1 / self.a) - 
                np.cos(m * np.pi * x2 / self.a))
            Iy = (self.b / (n * np.pi)) * (
                np.cos(n * np.pi * y1 / self.b) - 
                np.cos(n * np.pi * y2 / self.b))
            F[i] = q0 * Ix * Iy
        return F
    
    def load_vector_circular_patch(self, q0, xc, yc, R):
        """Load vector for circular patch via numerical integration."""
        F = np.zeros(self.ndof)
        for i in range(self.ndof):
            m, n = self.mode(i)
            def integrand(theta, r):
                x = xc + r * np.cos(theta)
                y = yc + r * np.sin(theta)
                if 0 <= x <= self.a and 0 <= y <= self.b:
                    return r * np.sin(m*np.pi*x/self.a) * np.sin(n*np.pi*y/self.b)
                return 0.0
            result, _ = dblquad(integrand, 0, R, 0, 2*np.pi, epsabs=1e-8)
            F[i] = q0 * result
        return F
    
    def solve(self, F):
        """Solve for mode coefficients."""
        return solve(self.stiffness_matrix(), F)
    
    def displacement(self, A, x, y):
        """Compute deflection at point."""
        w = 0.0
        for i, Ai in enumerate(A):
            m, n = self.mode(i)
            w += Ai * np.sin(m*np.pi*x/self.a) * np.sin(n*np.pi*y/self.b)
        return w

# Example: 1m × 1m steel plate, 10mm thick, central patch load
plate = RitzPlateSS(a=1.0, b=1.0, h=0.01, E=200e9, nu=0.3, M=10, N=10)
F = plate.load_vector_rect_patch(q0=10000, x1=0.4, y1=0.4, x2=0.6, y2=0.6)
A = plate.solve(F)
w_center = plate.displacement(A, 0.5, 0.5)
print(f"Central deflection: {w_center*1000:.4f} mm")
```

For **clamped or mixed boundaries**, replace the sine functions with appropriate beam functions and compute stiffness integrals numerically:

```python
def beam_CF(xi, m):
    """Clamped-free beam function."""
    beta = [1.8751, 4.6941, 7.8548, 10.996, 14.137]
    b = beta[m-1] if m <= 5 else (2*m - 1) * np.pi / 2
    sigma = (np.sinh(b) - np.sin(b)) / (np.cosh(b) + np.cos(b))
    return (np.cosh(b*xi) - np.cos(b*xi) - 
            sigma * (np.sinh(b*xi) - np.sin(b*xi)))

def beam_SF(xi, m):
    """Simply supported-free beam function."""
    beta = [3.9266, 7.0686, 10.210, 13.352, 16.493]
    b = beta[m-1] if m <= 5 else (4*m + 1) * np.pi / 4
    return np.sin(b*xi) - (np.sin(b)/np.sinh(b)) * np.sinh(b*xi)
```

## Convergence behavior and practical term counts

Convergence depends strongly on both boundary conditions and load type:

| Configuration | Load type | Terms needed (M=N) | Relative error |
|--------------|-----------|-------------------|----------------|
| SSSS | Uniform | 3–5 | < 1% |
| SSSS | Central patch | 8–12 | < 1% |
| SSSS | Point load | 15–25 | < 2% at distance |
| CCCC | Uniform | 5–8 | < 2% |
| SCSF/SSSF | Patch | 10–15 | < 2% |

**Key insight**: Deflection converges faster than moments. For accurate stress results, double the term count. Series diverge at concentrated load points—always model point loads as small patches (R ≈ h for a "practical point load").

A convergence study should track relative change:
```python
for N in [3, 5, 8, 12, 16, 20]:
    plate = RitzPlateSS(a, b, h, E, nu, N, N)
    w_max = compute_max_deflection(plate, load)
    if N > 3:
        rel_change = abs(w_max - w_prev) / w_prev
        if rel_change < 0.01:  # 1% convergence
            break
    w_prev = w_max
```

## Simpler alternatives to full Ritz for SCSF and SSSF plates

Both SCSF and SSSF belong to the six **Lévy-type boundary conditions** (two opposite edges simply supported), allowing exact single-series solutions that converge faster than double-series Ritz.

**Lévy method** assumes w(x,y) = Σₘ Yₘ(y)·sin(mπx/a), automatically satisfying simply supported edges at x = 0 and x = a. The remaining boundary conditions on y = 0 and y = b reduce to solving a fourth-order ODE for each Yₘ(y):

Yₘ(y) = Aₘ cosh(αₘy) + Bₘ sinh(αₘy) + Cₘ cos(γₘy) + Dₘ sin(γₘy)

Coefficients come from four boundary equations. For SSSF: simply supported at y = 0 (w = 0, Myy = 0) and free at y = b (Myy = 0, Vy = 0). This yields algebraic equations solvable term-by-term.

**Finite integral transform method** applies Fourier transforms directly to the governing equation and boundary conditions, converting the PDE to linear algebra without assuming trial functions. Li Rui's group (Dalian University) has published extensive solutions for plates with free edges, though **500+ terms may be needed** for accurate results at free corners.

**Gorman's superposition method** decomposes the problem into Lévy-type "building blocks"—auxiliary problems with simpler boundaries. Superimposing solutions satisfies all original conditions. This converges faster than Ritz (40×40 matrices vs. 100×100) and provides upper or lower bounds depending on block selection. Reference: D.J. Gorman, *Vibration Analysis of Plates by the Superposition Method* (World Scientific, 1999).

**Recommendation**: For SCSF and SSSF under patch loads, start with Lévy solution. Only use full Ritz if edges have elastic restraints or non-standard conditions.

## Validation benchmarks for implementation verification

**SSSS square plate under uniform load** (Timoshenko & Woinowsky-Krieger, Table 35):

| Parameter | Formula | Value (a/b=1, ν=0.3) |
|-----------|---------|---------------------|
| Max deflection | wmax = α·pa⁴/D | α = 0.00406 |
| Max moment Mx | Mx,max = β·pa² | β = 0.0479 |
| Max moment My | My,max = γ·pa² | γ = 0.0479 |

**SSSS square plate under central point load**:
- wmax = 0.01160 · Pa²/D

**SSSF plate under uniform load** (Roark's Formulas, free edge ratio = 1.0):
- Deflection coefficient: **c₁ = 0.140** in wmax = c₁·pa⁴/(Eh³)
- Stress coefficient: c₂ = 0.67 in σmax = c₂·pa²/h²

**SCSF plate** (Gao et al., 2019): For hydrostatic loading, dimensionless deflection w′ = wD/(q₀b⁴) at the free edge corners provides benchmark values. FEM validation with ABAQUS shows series truncation dominates error.

**Convergence verification for SSSS uniform load**:

| Terms | wmax×D/(pa⁴) | Error |
|-------|-------------|-------|
| 1×1 | 0.00416 | +2.5% |
| 3×3 | 0.00406 | +0.1% |
| 5×5 | 0.004062 | exact |

These values should match your implementation output. Discrepancies beyond 1% indicate coding errors in stiffness assembly or load vector computation.

## Conclusion and implementation guidance

The Rayleigh-Ritz method provides a versatile framework for plate bending analysis, but **SCSF and SSSF are special cases with exact Lévy solutions**—consider these first. For general implementation:

- Use **sine functions for simply supported edges** (diagonal stiffness, fast convergence)
- Implement **beam functions numerically** for clamped/free boundaries using tabulated eigenvalues
- Model patch loads with **separable integrals** for rectangles, **polar integration** for circles
- Start with **M = N = 10** terms and verify convergence to 1%
- Validate against Timoshenko tables before tackling novel geometries

The Python code provided handles simply supported plates directly. For SCSF, use Xₘ(x) = sin(mπx/a) in x and Yₙ(y) from C-S beam functions in y; for SSSF, use S-F beam functions in y. Numerical integration of the stiffness integrals via scipy.integrate.quad replaces the closed-form diagonal results.