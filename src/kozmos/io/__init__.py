"""File formats: FITS images and binary tables."""

from kozmos.io import fits
from kozmos.io.fits import (
    BLOCK,
    CARD,
    list_hdus,
    parse_header,
    read_fits,
    read_table,
    write_fits,
)

__all__ = [
    "BLOCK",
    "CARD",
    "fits",
    "list_hdus",
    "parse_header",
    "read_fits",
    "read_table",
    "write_fits",
]
