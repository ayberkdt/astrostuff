# Plan folder

The plan of the astronomy and astrophysics calculation library (`kozmos`). Prepared on 25 November 2026. The first version of the library was written the same day; the current list of implemented functions is in [STATUS.md](STATUS.md).

## Reading order

1. [01_vision_and_scope.md](01_vision_and_scope.md): purpose, scope, open decisions.
2. [02_field_survey/](02_field_survey/): existing software, data and current state.
   - [01_software_ecosystem.md](02_field_survey/01_software_ecosystem.md): libraries by field, their versions and their roles in the project
   - [02_hubble_and_cosmology_data.md](02_field_survey/02_hubble_and_cosmology_data.md): the Pantheon+SH0ES data release, other datasets, the state of `H0` measurements
   - [03_data_archives.md](02_field_survey/03_data_archives.md): archives, access protocols, IVOA standards
   - [04_limits_of_the_survey.md](02_field_survey/04_limits_of_the_survey.md): how each piece of information was obtained and how reliable it is
3. [03_mathematics/](03_mathematics/)
   - [01_spherical_astronomy_and_transforms.md](03_mathematics/01_spherical_astronomy_and_transforms.md)
   - [02_time_scales.md](03_mathematics/02_time_scales.md)
   - [03_numerical_methods.md](03_mathematics/03_numerical_methods.md)
   - [04_statistics_and_inference.md](03_mathematics/04_statistics_and_inference.md)
   - [05_image_and_signal_processing.md](03_mathematics/05_image_and_signal_processing.md)
4. [04_physics/](04_physics/)
   - [01_radiation_and_photometry.md](04_physics/01_radiation_and_photometry.md)
   - [02_orbits_and_celestial_mechanics.md](04_physics/02_orbits_and_celestial_mechanics.md)
   - [03_stellar_astrophysics.md](04_physics/03_stellar_astrophysics.md)
   - [04_galaxies_and_dynamics.md](04_physics/04_galaxies_and_dynamics.md)
   - [05_cosmology.md](04_physics/05_cosmology.md)
   - [06_detector_and_atmosphere.md](04_physics/06_detector_and_atmosphere.md)
   - [07_relativity_and_high_energy.md](04_physics/07_relativity_and_high_energy.md)
   - [08_black_holes_stars_and_star_formation.md](04_physics/08_black_holes_stars_and_star_formation.md): the three implemented physics modules and what was found while reading the sources
5. [05_data_science/](05_data_science/)
   - [01_data_access_and_formats.md](05_data_science/01_data_access_and_formats.md)
   - [02_image_reduction_pipeline.md](05_data_science/02_image_reduction_pipeline.md)
   - [03_photometry_astrometry_spectroscopy.md](05_data_science/03_photometry_astrometry_spectroscopy.md)
   - [04_time_series.md](05_data_science/04_time_series.md)
   - [05_data_management_and_reproducibility.md](05_data_science/05_data_management_and_reproducibility.md)
   - [06_optical_and_infrared_observing.md](05_data_science/06_optical_and_infrared_observing.md): the implemented observing pipeline, the state of the services, infrared to-do items
6. [06_architecture/](06_architecture/)
   - [01_package_layout.md](06_architecture/01_package_layout.md)
   - [02_code_standards_and_source_citation.md](06_architecture/02_code_standards_and_source_citation.md)
   - [03_testing_and_verification.md](06_architecture/03_testing_and_verification.md)
7. [07_roadmap.md](07_roadmap.md): phases and acceptance criteria.
8. [08_example_projects/](08_example_projects/)
   - [01_hubble_diagram.md](08_example_projects/01_hubble_diagram.md)
   - [02_other_projects.md](08_example_projects/02_other_projects.md)
9. [09_function_inventory.md](09_function_inventory.md): the functions to be written, module by module, with their units and sources.
10. [STATUS.md](STATUS.md): implemented functions, constants and source verification status (generated from the registry).
11. [bibliography/](../bibliography/): `references.bib`, `packages.txt`, the verification script and its record.

## Citation rule

The keys in square brackets (`[Hogg1999]`) point to entries in `bibliography/references.bib`. The key can be followed by the location inside the source (`[Meeus1998, Section 7]`).

## Confidence level of the information

- The bibliographic entries and the package versions were tested against external records; the result is in [bibliography/VERIFICATION_SUMMARY.md](../bibliography/VERIFICATION_SUMMARY.md).
- Maintenance status, archive versions and current measured values come from a web survey and are marked as such in the documents.
- The equations, coefficients and equation numbers in the plan documents are information recalled from memory. For implemented functions, the status is recorded per function: for those marked `read`, the equation was taken from the text of the source while the code was written (arXiv TeX files, the NIST constants table, ERFA source code); for those marked `pending`, the source has not yet been opened.

Details: [02_field_survey/04_limits_of_the_survey.md](02_field_survey/04_limits_of_the_survey.md).

## Re-running the verification

```
python bibliography/verify.py
```

The script compares the bibliographic entries with the Crossref, DataCite, arXiv and Open Library records and the package versions with PyPI, and checks that every citation key in the plan documents and the source code is defined in the bibliography. If the exit code is not zero, there is a mismatched entry or an undefined key.
