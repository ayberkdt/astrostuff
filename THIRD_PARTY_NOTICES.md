# Third-party notices

This repository hosts no third-party source code or raw data; it contains a few figures derived from third-party data. The notices below cover the source code that some functions follow and the data on which the figures rest.

## ERFA (Essential Routines for Fundamental Astronomy)

The functions in the `kozmos.time.julian`, `kozmos.core.angles` and `kozmos.coordinates.transforms` modules were rewritten in Python by following the algorithms and coefficients in the source files of the ERFA C library (`cal2jd.c`, `jd2cal.c`, `era00.c`, `gmst06.c`, `obl06.c`, `sepp.c`, `s2c.c`, `c2s.c`, `anp.c`, `anpm.c`, `icrs2g.c`, `hd2ae.c`, `ae2hd.c`, `erfam.h`). ERFA is derived, with permission, from the IAU SOFA library. This repository took as its source not SOFA itself but a library derived from SOFA.

The copyright and licence notice in the ERFA source files:

```
Copyright (C) 2013-2023, NumFOCUS Foundation.
All rights reserved.

This library is derived, with permission, from the International
Astronomical Union's "Standards of Fundamental Astronomy" library,
available from http://www.iausofa.org.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1 Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

2 Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in
  the documentation and/or other materials provided with the
  distribution.

3 Neither the name of the Standards Of Fundamental Astronomy Board,
  the International Astronomical Union nor the names of its
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

The presence of this notice here is not a legal assessment; the licence obligations of the parts derived from ERFA must be assessed separately by the owner of the repository.

## Data

The datasets used by the analyses are not put in the repository; they are downloaded from their source at run time. The repository of the Pantheon+SH0ES data release has no licence file; the data is used with reference to the papers of Scolnic et al. (2022), Brout et al. (2022) and Riess et al. (2022).

## Survey data used in the figures

Some of the figures in the `figures/` folder are derived from third-party observational data. The reservation of rights in the repository does not cover these figures; each is subject to the terms below. The terms were read from the relevant pages on 25 November 2026.

| Figure | Data | Terms |
|---|---|---|
| `optical_infrared.png` (left panel) | Digitized Sky Survey 2, red band | CC BY-NC-SA 4.0; not public domain |
| `optical_infrared.png` (middle panel), `flame_nebula_jhk.png`, `real_image_validation.png` | 2MASS images and Point Source Catalog | Images in the image gallery are public domain; an acknowledgement text is requested |
| `optical_infrared.png` (right panel) | WISE 12 µm | An acknowledgement text is requested; the page that was read has no licence statement |
| `hr_diagram.png` | Gaia DR3 | CC BY-NC 3.0 IGO |
| `sky_map.png` | Bright Star Catalogue, 5th edition (HEASARC `bsc5p`) | See the note below |
| `hubble_diagram.png` | Pantheon+SH0ES data release | Not read |
| `cosmic_rays_real_data.png` | Hubble STIS, program 8310 (MAST); program 7245 is also used in the measurements | MAST data use page: CC0 under NASA policy; an acknowledgement text is expected |
| `infrared_reduction.png` | ESO NTT/SOFI, program 184.D-1152(F) | CC BY 4.0; the copyright is held by ESO |

### Digitized Sky Survey

The data use page of MAST (`archive.stsci.edu/publishing/data-use`) states that DSS data is not in the public domain, that it can be used freely for scientific, educational and not-for-profit purposes (including web sites), and that it is bound by the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) licence. For this reason the figure `optical_infrared.png` is also shared under CC BY-NC-SA 4.0 and cannot be used for commercial purposes.

The rights holder varies with the plate used: UK SERC/PPARC (today STFC) for southern surveys before 1993, the Anglo-Australian Observatory Board for southern surveys after 1993, the California Institute of Technology for the POSS-II northern surveys, and the Association of Universities for Research in Astronomy, Inc. for the remaining material. Which plate the field in the figure (δ = −1.9°) came from was not read from the SkyView header; the rights holder is one of these four.

### 2MASS

The 2MASS frequently asked questions page of IRSA states that the images in the image galleries are in the public domain and requests the following acknowledgement texts:

> This publication makes use of data products from the Two Micron All Sky Survey, which is a joint project of the University of Massachusetts and the Infrared Processing and Analysis Center/California Institute of Technology, funded by the National Aeronautics and Space Administration and the National Science Foundation.

> Atlas Image mosaic obtained as part of the Two Micron All Sky Survey (2MASS), a joint project of the University of Massachusetts and the Infrared Processing and Analysis Center/California Institute of Technology, funded by the National Aeronautics and Space Administration and the National Science Foundation.

On the page, the public domain statement applies to the image galleries. The cutouts in this repository were produced not from the gallery but from FITS files that SkyView resampled from the Atlas images; that the statement also covers these cutouts has not been verified separately.

### WISE

The introduction of the WISE All-Sky Release Explanatory Supplement requests the following acknowledgement text:

> This publication makes use of data products from the Wide-field Infrared Survey Explorer, which is a joint project of the University of California, Los Angeles, and the Jet Propulsion Laboratory/California Institute of Technology, funded by the National Aeronautics and Space Administration.

The reason for the region in the figure that carries no data is in the caveat notes of the same supplement (section I.4.c.vii): saturated pixels are masked in single exposures and do not enter the Atlas image; in bright regions above the saturation limit, the intensity and uncertainty maps carry the value NaN.

### IRSA

The Gaia and 2MASS catalog queries were made through IRSA. The "Acknowledging IRSA" page of IRSA requests the following text:

> This research has made use of the NASA/IPAC Infrared Science Archive, which is funded by the National Aeronautics and Space Administration and operated by the California Institute of Technology.

### Gaia

The Gaia licence page of ESA (`cosmos.esa.int/web/gaia-users/license`) states that Gaia data is distributed under the CC BY-NC 3.0 IGO licence; for commercial use it refers to the archive terms of ESA. The catalog data is not hosted in this repository. The citation page of Gaia says that each data release gives its own acknowledgement text; the acknowledgement text of DR3 has not been read and has not been added here.

### Bright Star Catalogue

The licence note on the `bsc5p` page of HEASARC states that the catalog is offered only for scientific research, that the copyright is not held by HEASARC, and that HEASARC cannot explain the terms of commercial use; the copyright of the last printed edition (4th edition) is held by Yale University Observatory. The catalog is not hosted in this repository; `sky_map.png` is drawn from the positions and magnitudes in the catalog.

### Hubble (MAST)

The data use page of MAST states that MAST data, apart from exceptions such as DSS, is in the public domain and is licensed under CC0 under NASA policy; it expects the mission and STScI to be mentioned in publications. The Hubble text on the mission acknowledgements page of MAST:

> This research is based on observations made with the NASA/ESA Hubble Space Telescope obtained from the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS5-26555. These observations are associated with programs 8310 and 7245.

### ESO

The ESO Data Access Policy (`archive.eso.org/cms/eso-data-access-policy.html`) states that the copyright of all data in the archive remains with ESO and that the data is distributed under Creative Commons Attribution 4.0 (CC BY 4.0); it must be stated that the data originates from ESO, and the file headers must be kept. The text requested for publications that use raw data:

> Based on observations collected at the European Southern Observatory under ESO programme 184.D-1152(F).

The raw frames are not hosted in this repository; `infrared_reduction.png` is derived from them and is shared under the CC BY 4.0 terms.

### SkyView and Pantheon+

The image cutouts were downloaded from the NASA SkyView service; the acknowledgement page of SkyView itself could not be reached (the request returned "File Not Found"). The terms of use of the Pantheon+SH0ES data release have not been read; the data is not hosted in this repository, and `hubble_diagram.png` is derived from it.

## Publications

The publications on which the functions rest are listed in `bibliography/references.bib`; the equations were taken from these publications, and their texts were not copied.
