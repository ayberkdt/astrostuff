"""Image reduction: frame combination and calibration, cosmic-ray cleaning, infrared sky subtraction."""

from kozmos.reduction import calibration, cosmic_rays, sky
from kozmos.reduction.calibration import (
    ReducedFrame,
    combine,
    gain_and_read_noise,
    reduce_frame,
)
from kozmos.reduction.cosmic_rays import (
    detect_cosmic_rays,
)
from kozmos.reduction.sky import (
    subtract_dynamic_sky,
    subtract_static_sky,
)

__all__ = [
    "calibration",
    "combine",
    "cosmic_rays",
    "detect_cosmic_rays",
    "gain_and_read_noise",
    "reduce_frame",
    "ReducedFrame",
    "sky",
    "subtract_dynamic_sky",
    "subtract_static_sky",
]
