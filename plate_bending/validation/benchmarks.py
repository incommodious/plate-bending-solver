"""
Benchmark Solutions for Plate Bending
=====================================

Classical solutions from Timoshenko, Szilard, etc.
All for square plates (a/b = 1) with nu = 0.3.
"""


class Benchmarks:
    """Classical benchmark solutions from Timoshenko, Szilard, etc."""

    # W = alpha * q * a^4 / D, M = beta * q * a^2  (all for nu = 0.3)
    DATA = {
        'SSSS': {
            'W_center_coef': 0.00406,
            'Mx_center_coef': 0.0479,
            'My_center_coef': 0.0479,
            'source': 'Timoshenko Table 8'
        },
        'SCSC': {
            'W_center_coef': 0.00192,
            'Mx_center_coef': 0.0244,
            'My_center_coef': 0.0332,
            'source': 'Timoshenko Table 36'
        },
        'SCSS': {
            'W_center_coef': 0.00263,
            'Mx_center_coef': 0.0340,
            'My_center_coef': 0.0380,
            'source': 'Timoshenko Table 44'
        },
        'SCSF': {
            'W_max_coef': 0.01377,
            'W_at_free': 0.01377,
            'Mx_max_coef': 0.0946,
            'My_max_coef': 0.0364,
            'source': 'Szilard Table 5.9, Timoshenko interpolation'
        },
        'SSSF': {
            'W_max_coef': 0.01286,
            'source': 'Timoshenko Table 48'
        },
        'SSFF': {
            'W_max_coef': 0.1875,
            'W_center_coef': 0.0593,
            'source': 'Xu et al. 2020, Table 5'
        },
        'SFSF': {
            'W_center_coef': 0.01309,
            'source': 'Timoshenko'
        },
    }

    @classmethod
    def get(cls, bc):
        """Get benchmark data for given BC."""
        return cls.DATA.get(bc.upper(), None)

    @classmethod
    def list_available(cls):
        """List all available benchmarks."""
        return list(cls.DATA.keys())
