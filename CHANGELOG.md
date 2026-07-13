# Changelog

## [1.0.5] - 2026-07-13

+ PyPI update checker in Settings (one-click upgrade)
+ Removed ETH-3 and ETH3oxi from bulk isotope standards
+ Version resolution via `importlib.metadata` (entrypoint-independent)
+ Optional `test` dependency group with pytest
+ switched default Streamlit theme from dark to light
+ D95eq integration on Dual Clumped Space page: 95% confidence ellipses, equilibrium curve/band, and Δ₉₅ thermometry (Teq, p-values) for D47+D48 axes
+ new `tools/d95eq_plotly.py` geometry/bridge module for D95eq Plotly rendering
+ new **"Together"** plot level combining sample means and replicate overview
+ per-sample Pearson ρ(Δ₄₇, Δ₄₈) estimated from replicates (significance-tested) to tilt confidence ellipses, with a per-sample ρ expander table
+ uncertainty-display toggle (error bars / ellipses / both), disequilibrium-slope input, equilibrium p-cutoff, and per-sample equilibrium projection arrows
+ per-isotope uncertainty columns in Processing (`SD`/`SE`/`95% CL` → `*_D47`/`*_D48`/`*_D49`) preserving joint Δ₄₇–Δ₄₈ covariance
+ Fiebig (2024) calibration constants and curve rendering consolidated in `TemperatureCalculator` (added reprocessed full-precision values and a Hill×affine coefficient helper)
+ new dependencies: `D95eq>=1.2.4`, `correldata`

## [1.0.4] - 2026-03-05

+ force `Sessions` col to string
+ advanced datetime handling

## [1.0.3] - 2026-02-24

+ `BasePage` class for consistent page setup (authentication, config, sidebar)
+ persistent **Settings** page (`98_Settings.py`) backed by `user_settings.json`
+ new modules: `constants.py`, `config.py`, `filters.py`, `base_page.py`
+ configurable isotopic constants (R13_VPDB, R17_VSMOW, R18_VSMOW, λ₁₇) and working-gas ratios via Settings
+ sanitize `Session` column in addition to `Sample`; block commas and semicolons
+ 50 °C equilibrated-gas target values for ∆₄₇, ∆₄₈, ∆₄₉
+ removed deprecated calibration sets from `init_params`
+ removed `08_D47crunch_plots.py` page (functionality merged elsewhere)
+ updated `SampleDatabase.xlsx` with extended example data
+ refactored `SessionStateManager` to use shared `db_connection`
+ added `render_plotly_chart` helper in `commons`


## [1.0.2] - 2025-12-08

+ no `scaling_factors` without PBL correction
+ typo in `STDs` output
+ Python≥3.12 (D47crunch dependency)
+ installation from PyPI


## [1.0.1] - 2025-12-04

+ exclude `Session` and `Sample` from smart_numeric_conversion()
+ Publish D4Xgui to PyPI


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