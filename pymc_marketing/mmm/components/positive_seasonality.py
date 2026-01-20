"""
Custom positive seasonality component for PyMC-Marketing.
Ensures seasonality output is always positive to prevent negative baseline.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


class PositiveSeasonality:
    """
    Seasonality component that guarantees positive output using exponential transformation.
    
    Formula: seasonality(t) = offset + exp(Σ [γᵢ * fourier_feature_i(t)])
    
    This ensures seasonality > 0 always, preventing baseline = intercept + seasonality < 0
    
    Parameters
    ----------
    n_order : int, default=2
        Number of Fourier pairs (results in n_order * 2 coefficients)
    prior_scale : float, default=0.1
        Standard deviation for Normal prior on Fourier coefficients.
        Smaller values = less seasonal variation
    min_offset : float, default=0.01
        Minimum positive offset to prevent numerical issues
    """
    
    def __init__(
        self,
        n_order: int = 2,
        prior_scale: float = 0.1,
        min_offset: float = 0.01
    ):
        self.n_order = n_order
        self.prior_scale = prior_scale
        self.min_offset = min_offset
    
    def _create_fourier_features(self, dates, n_order: int) -> np.ndarray:
        """Create Fourier features from dates."""
        # Convert to day of year
        if hasattr(dates, 'dt'):
            day_of_year = dates.dt.dayofyear.values
        else:
            day_of_year = dates.dayofyear.values
        
        periods_per_year = 365.25
        features = []
        
        for i in range(1, n_order + 1):
            # Sine component
            sin_feature = np.sin(2 * np.pi * i * day_of_year / periods_per_year)
            features.append(sin_feature)
            # Cosine component
            cos_feature = np.cos(2 * np.pi * i * day_of_year / periods_per_year)
            features.append(cos_feature)
        
        return np.column_stack(features)
    
    def apply(self, dates, coords: dict) -> pt.TensorVariable:
        """
        Apply positive seasonality transformation.
        
        Parameters
        ----------
        dates : pd.Series or pd.DatetimeIndex
            Date column for creating Fourier features
        coords : dict
            Coordinate dictionary for PyMC dimensions
        
        Returns
        -------
        positive_seasonality : TensorVariable
            Seasonality contribution (always positive)
        """
        # Create Fourier features
        fourier_matrix = self._create_fourier_features(dates, self.n_order)
        
        # Number of Fourier modes
        n_modes = self.n_order * 2
        
        # Normal prior on Fourier coefficients (can be negative)
        gamma_fourier = pm.Normal(
            "gamma_fourier",
            mu=0,
            sigma=self.prior_scale,
            shape=n_modes,
            dims="fourier_mode" if "fourier_mode" in coords else None
        )
        
        # Linear combination (can be negative)
        linear_combination = pt.dot(fourier_matrix, gamma_fourier)
        
        # Exponential transformation - ALWAYS POSITIVE!
        # Add min_offset to ensure numerical stability
        positive_seasonality = pm.Deterministic(
            "yearly_seasonality_contribution",
            self.min_offset + pt.exp(linear_combination),
            dims="date" if "date" in coords else None
        )
        
        return positive_seasonality

