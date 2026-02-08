"""
Matplotlib Figure Generation for Plate Bending Analysis
=======================================================

This module generates publication-quality matplotlib figures for embedding
in LaTeX reports. Supports both imperial and metric units with proper
conversions from SI base units.

Functions:
- generate_deflection_contour: Contour plot of deflection field
- generate_stress_contours: Side-by-side stress contour plots
- generate_deflection_profile: Line plot of deflection along x-direction

All functions save PNG files at 200 DPI for high-quality output.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Any

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

# Unit conversion factors from SI
UNIT_CONVERSIONS = {
    'length': {
        'imperial': 39.37,  # m to inches
        'metric': 1000.0    # m to mm
    },
    'deflection': {
        'imperial': 39370.0,  # m to mils (0.001 in)
        'metric': 1000.0      # m to mm
    },
    'stress': {
        'imperial': 0.000145037738,  # Pa to ksi
        'metric': 1e-6               # Pa to MPa
    }
}

UNIT_LABELS = {
    'length': {
        'imperial': 'in',
        'metric': 'mm'
    },
    'deflection': {
        'imperial': 'mil',
        'metric': 'mm'
    },
    'stress': {
        'imperial': 'ksi',
        'metric': 'MPa'
    }
}


def _convert_units(data: np.ndarray, quantity: str, units: str) -> np.ndarray:
    """Convert data from SI units to specified unit system."""
    factor = UNIT_CONVERSIONS[quantity][units]
    return data * factor


def _get_unit_label(quantity: str, units: str) -> str:
    """Get unit label for specified quantity and unit system."""
    return UNIT_LABELS[quantity][units]


def generate_deflection_contour(
    results: Dict[str, Any], 
    output_path: str, 
    units: str = 'imperial',
    load_center: Optional[Tuple[float, float]] = None,
    load_radius: Optional[float] = None
) -> str:
    """
    Generate contour plot of deflection field.
    
    Parameters
    ----------
    results : dict
        Results dictionary containing 'X', 'Y', 'W' (all 2D numpy arrays)
    output_path : str
        Path where PNG file will be saved
    units : str
        Unit system ('imperial' or 'metric')
    load_center : tuple, optional
        (x, y) coordinates of load center for marking
    load_radius : float, optional
        Radius of circular load for marking
        
    Returns
    -------
    str
        The output_path where file was saved
    """
    # Extract and convert data
    X = _convert_units(results['X'], 'length', units)
    Y = _convert_units(results['Y'], 'length', units)
    W = _convert_units(results['W'], 'deflection', units)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create contour plot
    levels = 20
    cs = ax.contourf(X, Y, W, levels=levels, cmap='viridis')
    
    # Add contour lines
    cs_lines = ax.contour(X, Y, W, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, shrink=0.8)
    deflection_label = _get_unit_label('deflection', units)
    cbar.set_label(f'Deflection ({deflection_label})')
    
    # Mark load location if provided
    if load_center is not None:
        # load_center and load_radius are in display units (same as axis)
        x_load = load_center[0]
        y_load = load_center[1]
        
        if load_radius is not None:
            r_load = load_radius
            circle = plt.Circle((x_load, y_load), r_load, fill=False, 
                              color='red', linestyle='--', linewidth=2, label='Load')
            ax.add_patch(circle)
            ax.legend()
        else:
            ax.plot(x_load, y_load, 'ro', markersize=8, label='Load center')
            ax.legend()
    
    # Labels and title
    length_label = _get_unit_label('length', units)
    ax.set_xlabel(f'x ({length_label})')
    ax.set_ylabel(f'y ({length_label})')
    ax.set_title('Plate Deflection')
    ax.set_aspect('equal')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_stress_contours(
    results: Dict[str, Any], 
    output_path: str, 
    units: str = 'imperial',
    load_center: Optional[Tuple[float, float]] = None,
    load_radius: Optional[float] = None
) -> str:
    """
    Generate side-by-side contour plots of sigma_x and sigma_y stress fields.
    
    Parameters
    ----------
    results : dict
        Results dictionary containing 'X', 'Y', 'sigma_x', 'sigma_y' (all 2D numpy arrays)
    output_path : str
        Path where PNG file will be saved
    units : str
        Unit system ('imperial' or 'metric')
    load_center : tuple, optional
        (x, y) coordinates of load center for marking
    load_radius : float, optional
        Radius of circular load for marking
        
    Returns
    -------
    str
        The output_path where file was saved
    """
    # Extract and convert data
    X = _convert_units(results['X'], 'length', units)
    Y = _convert_units(results['Y'], 'length', units)
    sigma_x = _convert_units(results['sigma_x'], 'stress', units)
    sigma_y = _convert_units(results['sigma_y'], 'stress', units)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Determine common colorbar range for comparison
    stress_max = max(np.max(np.abs(sigma_x)), np.max(np.abs(sigma_y)))
    levels = np.linspace(-stress_max, stress_max, 21)
    
    # Plot sigma_x
    cs1 = ax1.contourf(X, Y, sigma_x, levels=levels, cmap='RdBu_r')
    ax1.contour(X, Y, sigma_x, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Plot sigma_y
    cs2 = ax2.contourf(X, Y, sigma_y, levels=levels, cmap='RdBu_r')
    ax2.contour(X, Y, sigma_y, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add load markers if provided
    def add_load_marker(ax):
        if load_center is not None:
            x_load = load_center[0]
            y_load = load_center[1]
            
            if load_radius is not None:
                r_load = load_radius
                circle = plt.Circle((x_load, y_load), r_load, fill=False, 
                                  color='red', linestyle='--', linewidth=2)
                ax.add_patch(circle)
            else:
                ax.plot(x_load, y_load, 'ro', markersize=6)
    
    add_load_marker(ax1)
    add_load_marker(ax2)
    
    # Labels and titles
    length_label = _get_unit_label('length', units)
    stress_label = _get_unit_label('stress', units)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel(f'x ({length_label})')
        ax.set_ylabel(f'y ({length_label})')
        ax.set_aspect('equal')
    
    ax1.set_title(f'σₓ Stress ({stress_label})')
    ax2.set_title(f'σᵧ Stress ({stress_label})')
    
    # Shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = plt.colorbar(cs1, cax=cbar_ax)
    cbar.set_label(f'Stress ({stress_label})')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_deflection_profile(
    results: Dict[str, Any], 
    output_path: str, 
    y_section: Optional[float] = None,
    units: str = 'imperial',
    load_center: Optional[Tuple[float, float]] = None
) -> str:
    """
    Generate line plot of deflection along x-direction at specified y-coordinate.
    
    Parameters
    ----------
    results : dict
        Results dictionary containing 'X', 'Y', 'W' (all 2D numpy arrays)
    output_path : str
        Path where PNG file will be saved
    y_section : float, optional
        Y-coordinate for the section (default: center of plate)
    units : str
        Unit system ('imperial' or 'metric')
    load_center : tuple, optional
        (x, y) coordinates of load center for vertical line marker
        
    Returns
    -------
    str
        The output_path where file was saved
    """
    # Extract and convert data
    X = _convert_units(results['X'], 'length', units)
    Y = _convert_units(results['Y'], 'length', units)
    W = _convert_units(results['W'], 'deflection', units)
    
    # Determine y-section
    if y_section is None:
        y_section = Y[Y.shape[0]//2, 0]  # Center of plate
    else:
        y_section = _convert_units(np.array([y_section]), 'length', units)[0]
    
    # Find closest row to y_section
    y_idx = np.argmin(np.abs(Y[:, 0] - y_section))
    actual_y = Y[y_idx, 0]
    
    # Extract profile data
    x_profile = X[y_idx, :]
    w_profile = W[y_idx, :]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot deflection profile
    ax.plot(x_profile, w_profile, 'b-', linewidth=2, label='Deflection')
    
    # Mark load center if provided
    if load_center is not None:
        x_load = _convert_units(np.array([load_center[0]]), 'length', units)[0]
        ax.axvline(x_load, color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='Load center')
        ax.legend()
    
    # Labels and title
    length_label = _get_unit_label('length', units)
    deflection_label = _get_unit_label('deflection', units)
    
    ax.set_xlabel(f'x ({length_label})')
    ax.set_ylabel(f'Deflection ({deflection_label})')
    ax.set_title(f'Deflection Profile at y = {actual_y:.3f} {length_label}')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path