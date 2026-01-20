"""
Patch to use PositiveSeasonality in MMM models.
"""

from pymc_marketing.mmm.components.positive_seasonality import PositiveSeasonality


def patch_mmm_seasonality(mmm_instance, prior_scale=0.05):
    """
    Replace standard seasonality with positive seasonality.
    
    Usage:
        >>> from pymc_marketing.mmm import MMM
        >>> from pymc_marketing.mmm.patches import patch_mmm_seasonality
        >>> 
        >>> mmm = MMM(yearly_seasonality=2, ...)
        >>> mmm = patch_mmm_seasonality(mmm, prior_scale=0.05)
        >>> mmm.fit(X, y)
    
    Parameters
    ----------
    mmm_instance : MMM
        MMM instance to patch
    prior_scale : float, default=0.05
        Scale for seasonality prior (smaller = less variation)
    
    Returns
    -------
    mmm_instance : MMM
        Patched MMM instance
    """
    if not hasattr(mmm_instance, 'yearly_seasonality') or not mmm_instance.yearly_seasonality:
        return mmm_instance
    
    original_build = mmm_instance.build_model
    
    def patched_build(X, y):
        result = original_build(X, y)
        
        with mmm_instance.model:
            pos_seasonality = PositiveSeasonality(
                n_order=mmm_instance.yearly_seasonality,
                prior_scale=prior_scale
            )
            
            dates = X[mmm_instance.date_column]
            coords = mmm_instance.model.coords
            pos_seasonality.apply(dates, coords)
        
        return result
    
    mmm_instance.build_model = patched_build
    return mmm_instance

