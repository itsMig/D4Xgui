# Changelog

## [1.0.0] - 2025-12-01 – Initial commit

+ Initial release of D4Xgui
+ Data I/O functionality for uploading replicate and intensity data
+ Baseline correction using m/z47.5 half-mass cup
+ Processing capabilities for D47, D48, and D49 clumped isotopes
+ Standardization results visualization
+ Dual clumped space plotting
+ Interactive data discovery and analysis
+ D47crunch integration for data processing
+ Database management for storing and retrieving pre-processed d45-d49 data
+ Save and reload functionality for session states
+ Support for multiple calibration methods using D47calib
+ Temperature calculation from D47 values
+ Excel export functionality for results

## [1.0.1] - 2025-12-04

+ exclude `Session` and `Sample` from smart_numeric_conversion()
+ Publish D4Xgui to PyPI

## [1.0.2] - 2025-12-08

+ no `scaling_factors` without PBL correction
+ typo in `STDs` output
+ Python≥3.12 (D47crunch dependency)
+ installation from PyPI