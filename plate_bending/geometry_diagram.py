"""Generate a schematic geometry diagram for a rectangular plate."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import math

import matplotlib

# Use non-interactive backend for file output
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle


def generate_geometry_diagram(
    a: float,
    b: float,
    bc: str,
    output_path: str,
    units: str = "imperial",
    load_type: Optional[str] = None,
    load_center: Optional[Tuple[float, float]] = None,
    load_radius: Optional[float] = None,
    load_bounds: Optional[Tuple[float, float, float, float]] = None,
    h: Optional[float] = None,
) -> None:
    """Generate a plate geometry diagram and save it as a PNG.

    Parameters
    ----------
    a, b
        Plate dimensions (in display units).
    bc
        4-char boundary condition string in order: x=0, y=0, x=a, y=b.
        Characters: S=Simply Supported, C=Clamped, F=Free.
    output_path
        Path to save PNG.
    units
        'imperial' or 'metric' for labeling.
    load_type
        'uniform', 'circular', 'rect_patch', 'point', or None.
    load_center
        (x, y) tuple for circular/point loads.
    load_radius
        Radius for circular loads.
    load_bounds
        (x1, y1, x2, y2) bounds for rect_patch.
    h
        Thickness (shown in title if provided).
    """
    if not isinstance(bc, str) or len(bc) != 4:
        raise ValueError("bc must be a 4-character string like 'FCFC' or 'SSSS'.")

    bc = bc.upper()

    unit_label = "in" if units == "imperial" else "mm"

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=200)

    # Plate rectangle
    plate = Rectangle((0, 0), a, b, facecolor="#e6e6e6", edgecolor="black", linewidth=1.0)
    ax.add_patch(plate)

    # Edge styling
    _draw_edge(ax, (0, 0), (0, b), bc[0], outward=(-1, 0))  # x=0
    _draw_edge(ax, (0, 0), (a, 0), bc[1], outward=(0, -1))  # y=0
    _draw_edge(ax, (a, 0), (a, b), bc[2], outward=(1, 0))   # x=a
    _draw_edge(ax, (0, b), (a, b), bc[3], outward=(0, 1))   # y=b

    # Edge labels
    _label_edge(ax, (0, 0), (0, b), bc[0], offset=(-0.06 * a, 0), align="right")
    _label_edge(ax, (0, 0), (a, 0), bc[1], offset=(0, -0.08 * b), align="center")
    _label_edge(ax, (a, 0), (a, b), bc[2], offset=(0.06 * a, 0), align="left")
    _label_edge(ax, (0, b), (a, b), bc[3], offset=(0, 0.08 * b), align="center")

    # Dimensions
    _draw_dimension(ax, (0, -0.18 * b), (a, -0.18 * b), f"{a:.3g} {unit_label}")
    _draw_dimension(ax, (-0.18 * a, 0), (-0.18 * a, b), f"{b:.3g} {unit_label}")

    # Load visualization
    _draw_load(ax, a, b, load_type, load_center, load_radius, load_bounds)

    # Title
    title = f"BC: {bc}"
    if h is not None:
        title += f" | thickness h = {h:.3g} {unit_label}"
    ax.set_title(title, fontsize=10)

    # Axes settings
    margin_x = 0.25 * a
    margin_y = 0.25 * b
    ax.set_xlim(-margin_x, a + margin_x)
    ax.set_ylim(-margin_y, b + margin_y)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _edge_name(code: str) -> str:
    if code == "C":
        return "CLAMPED"
    if code == "S":
        return "SIMPLY SUPPORTED"
    return "FREE"


def _draw_edge(ax, p1: Tuple[float, float], p2: Tuple[float, float], code: str, outward: Tuple[float, float]) -> None:
    """Draw edge with boundary condition symbols."""
    x1, y1 = p1
    x2, y2 = p2

    if code == "C":
        # Thick solid line
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=3.0)
        _draw_clamp_hatching(ax, p1, p2, outward)
    elif code == "S":
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.5)
        _draw_roller_triangles(ax, p1, p2, outward)
    else:
        # Free edge: dashed
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.0, linestyle="--")


def _draw_clamp_hatching(ax, p1: Tuple[float, float], p2: Tuple[float, float], outward: Tuple[float, float]) -> None:
    """Draw ground hatching marks along a clamped edge."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return

    nx, ny = outward
    # hatch parameters
    hatch_len = 0.04 * length
    hatch_spacing = 0.10 * length
    count = max(2, int(length / hatch_spacing))

    for i in range(count + 1):
        t = i / count
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        # diagonal hatch
        hx1 = x + nx * hatch_len * 0.3 - ny * hatch_len * 0.3
        hy1 = y + ny * hatch_len * 0.3 + nx * hatch_len * 0.3
        hx2 = x + nx * hatch_len
        hy2 = y + ny * hatch_len
        ax.plot([hx1, hx2], [hy1, hy2], color="black", linewidth=1.0)


def _draw_roller_triangles(ax, p1: Tuple[float, float], p2: Tuple[float, float], outward: Tuple[float, float]) -> None:
    """Draw small triangular rollers along a simply supported edge."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return

    nx, ny = outward
    tri_h = 0.05 * length
    tri_w = 0.06 * length
    spacing = 0.20 * length
    count = max(2, int(length / spacing))

    # Direction along edge
    ex = (x2 - x1) / length
    ey = (y2 - y1) / length

    for i in range(1, count):
        t = i / count
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t

        base_center_x = x + nx * tri_h
        base_center_y = y + ny * tri_h

        pA = (base_center_x - ex * tri_w / 2, base_center_y - ey * tri_w / 2)
        pB = (base_center_x + ex * tri_w / 2, base_center_y + ey * tri_w / 2)
        pC = (x, y)

        tri = Polygon([pA, pB, pC], closed=True, facecolor="white", edgecolor="black", linewidth=1.0)
        ax.add_patch(tri)


def _label_edge(
    ax,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    code: str,
    offset: Tuple[float, float],
    align: str,
) -> None:
    """Place edge label near the center of the edge."""
    x = (p1[0] + p2[0]) / 2 + offset[0]
    y = (p1[1] + p2[1]) / 2 + offset[1]
    label = _edge_name(code)

    ha = "center"
    va = "center"
    if align == "left":
        ha = "left"
    elif align == "right":
        ha = "right"

    ax.text(x, y, label, fontsize=8, ha=ha, va=va)


def _draw_dimension(ax, p1: Tuple[float, float], p2: Tuple[float, float], text: str) -> None:
    """Draw a dimension line with arrowheads and text."""
    ax.annotate(
        "",
        xy=p1,
        xytext=p2,
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
    )
    x = (p1[0] + p2[0]) / 2
    y = (p1[1] + p2[1]) / 2
    ax.text(x, y, text, fontsize=8, ha="center", va="center", backgroundcolor="white")


def _draw_load(
    ax,
    a: float,
    b: float,
    load_type: Optional[str],
    load_center: Optional[Tuple[float, float]],
    load_radius: Optional[float],
    load_bounds: Optional[Tuple[float, float, float, float]],
) -> None:
    """Draw load visualization based on type."""
    if load_type is None:
        return

    load_type = load_type.lower()

    if load_type == "uniform":
        rect = Rectangle((0, 0), a, b, facecolor="red", alpha=0.15, edgecolor="none")
        ax.add_patch(rect)
        return

    if load_type == "circular" and load_center and load_radius:
        circ = Circle(load_center, load_radius, facecolor="red", alpha=0.35, edgecolor="darkred", linewidth=1.0)
        ax.add_patch(circ)
        return

    if load_type == "rect_patch" and load_bounds:
        x1, y1, x2, y2 = load_bounds
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        rect = Rectangle((x, y), w, h, facecolor="red", alpha=0.25, edgecolor="darkred", linewidth=1.0)
        ax.add_patch(rect)
        return

    if load_type == "point" and load_center:
        x, y = load_center
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(x, y + 0.15 * b),
            arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
        )
        ax.plot([x], [y], marker="o", color="red")
        return

