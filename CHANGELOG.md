# Changelog

## [1.0.9] - 2026-11-12

+ `98_Settings`: choose one or more mineral δ¹⁸O acid fractionations (calcite and aragonite, Kim, Mucci & Taylor 2007; dolomite and siderite, Rosenbaum & Sheppard 1986; siderite, Fernandez et al. 2016). Default remains calcite (Kim 2007)
+ `04_Processing`: acid factor applied on VPDB, then the mineral value projected to VSMOW. Summary gains `d18O_{mineral}_VPDB` / `d18O_{mineral}_VSMOW` per selection (citation tag added when two equations share a mineral)
+ `04_Processing`: replicate sample SDs for δ₄₇/δ₄₈/δ₄₉ (`SD_d47`/`SD_d48`/`SD_d49`); summary columns written in a fixed export order
+ `04_Processing` / `tools/commons`: FAIR metadata and `proc_params` values stored as strings so Streamlit Arrow and Excel export no longer break on mixed bools; settings path recorded as the filename only; standards δ¹³C and δ¹⁸O no longer swapped in the run-parameters sheet
+ pooled standardization uses vectorized residuals (`tools/d47crunch_fast.py`) with the same drift model as D47crunch. A 32-session, ~9000-replicate run dropped from about 40 min to about 7 min. Reported Δ and 2SE are unchanged. Independent-session correction is still D47crunch
+ standardization-error contours no longer build a matplotlib mesh or an unbounded `np.arange`. Ill-conditioned sessions (typical of noisy Δ₄₉) are skipped instead of SIGKILL-ing the app
+ `Pysotope_fork`: an empty standards dict now really opts that mass out of the baseline optimizer
+ `06_Dual_Clumped_Space`: D95eq results cached as plain arrays instead of full uncertainty graphs; Teq PDF windows are clamped so one bad point cannot abort the overlay; runs above 75 points ask for confirmation before starting
+ floating **Report a bug** button uses `st.iframe` and no longer leaves a speck in the sidebar
+ `streamlit>=1.56.0`

## [1.0.8] - 2026-09-16

+ `run.py`: put the package directory on `sys.path` so `uvx d4xgui` / the `d4xgui` console script can import `tools` (fixes `ModuleNotFoundError: No module named 'tools'`)

## [1.0.7] - 2026-09-02

+ `03_Baseline_correction`: per-scale PBL column assignment — enable Δ₄₇/Δ₄₈/Δ₄₉ correction independently and map each scale to its own negative-baseline signal (e.g. m/z 47.5 → Δ₄₇, 48.5 → Δ₄₈)
+ `01_Data_IO`: preserve all paired `raw_s…` / `raw_r…` PBL columns from uploads (not limited to 47.5 / 48.5)
+ `Pysotope_fork`: per-mass `half_mass_cups` mapping for scaling-factor keys and baseline correction math
+ `04_Processing`: derive per-mass baseline-correction status from non-zero `{mz}b_*` scaling-factor keys
+ `tools/commons`: `discover_baseline_signal_suffixes`, `baseline_signal_column_label`, `default_pbl_suffix_for_mass` helpers
+ auto-create `.streamlit/secrets.toml` from shipped `secrets.toml.example` on first run; exclude personal `user_settings.json` from releases

## [1.0.6] - 2026-07-16

+ `03_Baseline_correction`: per-mass opt-in checkboxes (Δ₄₇/Δ₄₈/Δ₄₉); unchecked masses skip the PBL optimizer (`bg = raw`)
+ `Pysotope_fork.correctBaseline`: empty `D{mz}std` short-circuits the optimizer and zeros the scaling factor
+ `04_Processing`: `Baseline correction 47/48/49` rows in `proc_params` derived from `sss.scaling_factors` instead of widget state
+ `04_Processing`: dropped duplicated `D47/D48/D49 processed` rows from `proc_params`
+ `04_Processing`: FAIR metadata merged into `proc_params`; separate `Metadata` sheet removed from Excel export
+ `tools/commons.build_fair_metadata`: removed `Δ₄₇/Δ₄₈/Δ₄₉ standards` rows (duplicated `proc_params`)
+ `06_Dual_Clumped_Space`: D95eq ellipses and Teq / T_kinetic hover bounds now follow the "Error determination" toggle (1SE → 68% CI ±1σ, 2SE → 95% CI ±2σ)
+ floating **"Report a bug"** button on every page (screenshot to clipboard + prefilled `mailto:` with version/OS context)
+ configurable `bug_report_email` in Settings
+ single source-of-truth versioning via `D4Xgui/__init__.py` (`pyproject.toml`, Sphinx, `Welcome.py` all read from it)
+ new `tools/version.py` helper for launch-agnostic version resolution
+ `datetime_parsing`: use `pd.api.types.is_numeric_dtype`; drop deprecated `infer_datetime_format=True`
+ `01_Data_IO`: fix pandas FutureWarning from chained `Series.replace(..., inplace=True)`
+ reverted default Streamlit theme back to dark

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