"""Detailed continuity check at patch load boundaries."""

import numpy as np
from plate_bending.solvers.levy_solver import StableLevySolver

# Parameters
a, b, h = 1.0, 1.0, 0.01
E, nu = 2.1e11, 0.3
q0 = 10000

print("="*70)
print("PATCH LOAD CONTINUITY CHECK")
print("="*70)
print()

# Use SSSS with asymmetric patch for clear test
levy = StableLevySolver(a, b, h, E, nu, 'S', 'S', 100)
y1, y2 = 0.3, 0.6  # Asymmetric patch

result = levy.solve('rect_patch', q0, x1=0.3*a, y1=y1*b, x2=0.7*a, y2=y2*b)
W = result['W']
X, Y = result['X'], result['Y']

ny, nx = W.shape
dy = Y[1, 0] - Y[0, 0]

# Extract centerline
centerline_x_idx = nx // 2
w_centerline = W[:, centerline_x_idx]
y_centerline = Y[:, centerline_x_idx]

print("Deflection along y-centerline (x=0.5):")
print("-"*50)
print(f"{'y':^8} {'w (mm)':^12} {'dw/dy':^12} {'d2w/dy2':^12}")
print("-"*50)

# Compute derivatives
dw = np.gradient(w_centerline, dy)
d2w = np.gradient(dw, dy)

# Find indices near y1 and y2
for target_y in [y1, y2]:
    idx = np.argmin(np.abs(y_centerline - target_y))

    print(f"\n  Near y = {target_y:.1f} (patch boundary):")
    for i in range(max(0,idx-2), min(ny, idx+3)):
        print(f"{y_centerline[i]:8.4f} {w_centerline[i]*1e3:12.6f} {dw[i]*1e3:12.6f} {d2w[i]:12.6f}")

# Check for sudden jumps in d2w (curvature)
print("\n" + "-"*50)
print("Checking for curvature jumps at boundaries:")
print("-"*50)

for target_y in [y1, y2]:
    idx = np.argmin(np.abs(y_centerline - target_y))

    # 3-point curvature difference
    curv_before = d2w[idx-1] if idx > 0 else d2w[idx]
    curv_at = d2w[idx]
    curv_after = d2w[idx+1] if idx < ny-1 else d2w[idx]

    max_curv = max(abs(curv_before), abs(curv_at), abs(curv_after))
    if max_curv > 1e-15:
        jump1 = abs(curv_at - curv_before) / max_curv
        jump2 = abs(curv_after - curv_at) / max_curv
        print(f"  y = {target_y:.1f}: jump_before={jump1:.3f}, jump_after={jump2:.3f}")

        if max(jump1, jump2) < 0.2:
            print("    --> Smooth (expected jump < 0.2)")
        else:
            print("    --> Possible discontinuity")

# Final smoothness verdict
print("\n" + "="*70)

# Check overall smoothness using max curvature jump
max_jump = 0
for i in range(1, ny-1):
    if abs(d2w[i]) > 1e-15:
        jump = abs(d2w[i+1] - d2w[i]) / (abs(d2w[i]) + 1e-30)
        if jump > max_jump:
            max_jump = jump
            max_jump_y = y_centerline[i]

print(f"Max relative curvature jump: {max_jump:.3f} at y={max_jump_y:.3f}")
if max_jump < 0.3:
    print("VERDICT: Solution is smooth - patch load continuity verified!")
else:
    print("VERDICT: Solution may have continuity issues - needs investigation")
print("="*70)
