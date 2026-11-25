"""Access to archives and data releases: TAP queries, Gaia, 2MASS, the Bright Star Catalogue, SkyView cutouts, Pantheon+."""

from kozmos.archives import cache, gaia, heasarc, pantheon, skyview, tap, twomass
from kozmos.archives.cache import (
    USER_AGENT,
    key,
)
from kozmos.archives.gaia import (
    IRSA_TAP,
    absolute_magnitude,
    cone_search,
    nearby_stars,
)
from kozmos.archives.heasarc import (
    TAP,
    bright_stars,
    read_pipe_text,
)
from kozmos.archives.pantheon import (
    BASE_URL,
    FILES,
    PantheonData,
    load,
    read_covariance,
    read_table,
)
from kozmos.archives.skyview import (
    ATTEMPTS,
    ENDPOINT,
    SURVEYS,
    download_cutout,
)
from kozmos.archives.tap import (
    query,
    read_csv,
)
from kozmos.archives.twomass import (
    point_sources,
)

__all__ = [
    "absolute_magnitude",
    "ATTEMPTS",
    "BASE_URL",
    "bright_stars",
    "cache",
    "cone_search",
    "download_cutout",
    "ENDPOINT",
    "FILES",
    "gaia",
    "heasarc",
    "IRSA_TAP",
    "key",
    "load",
    "nearby_stars",
    "pantheon",
    "PantheonData",
    "point_sources",
    "query",
    "read_covariance",
    "read_csv",
    "read_pipe_text",
    "read_table",
    "skyview",
    "SURVEYS",
    "TAP",
    "tap",
    "twomass",
    "USER_AGENT",
]
