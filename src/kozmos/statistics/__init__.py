"""Statistics: linear fits, MCMC sampling and convergence diagnostics, robust estimators."""

from kozmos.statistics import fitting, mcmc, robust
from kozmos.statistics.fitting import (
    LinearFit,
    chi_square,
    fit_line,
    linear_least_squares,
)
from kozmos.statistics.mcmc import (
    Chain,
    autocorrelation_time,
    ensemble_sample,
    rank_normalized_rhat,
    sample_until_converged,
    split_rhat,
    stretch_factor,
)
from kozmos.statistics.robust import (
    MAD_TO_SIGMA,
    biweight_location,
    biweight_scale,
    mad_scale,
    sigma_clip,
)

__all__ = [
    "autocorrelation_time",
    "biweight_location",
    "biweight_scale",
    "Chain",
    "chi_square",
    "ensemble_sample",
    "fit_line",
    "fitting",
    "linear_least_squares",
    "LinearFit",
    "mad_scale",
    "MAD_TO_SIGMA",
    "mcmc",
    "rank_normalized_rhat",
    "robust",
    "sample_until_converged",
    "sigma_clip",
    "split_rhat",
    "stretch_factor",
]
