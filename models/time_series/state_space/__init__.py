"""
State-space structural time-series models package.
"""

from models.time_series.state_space.ucm import UCMResults, fit_local_linear_trend

__all__ = ["fit_local_linear_trend", "UCMResults"]
