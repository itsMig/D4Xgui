## Welcome to D4Xgui v1.0.2!

[D4Xgui](https://github.com/itsMig/D4Xgui) is developed to enable easy access to state-of-the-art CO₂ clumped isotope (∆₄₇, ∆₄₈ and ∆₄₉) data processing.
A recently developed optimizer algorithm allows pre-processing of mass spectrometric raw intensities utilizing a m/z47.5 half-mass Faraday cup correction to account for the effect of a negative pressure baseline, which is essential for accurate and highest precision clumped isotope analysis of CO₂ ([Bernecker et al., 2023](https://doi.org/10.1016/j.chemgeo.2023.121803)).
It is backed with the recently published processing tool [D47crunch (v.2.4.3)](https://github.com/mdaeron/D47crunch) (following the methodology outlined in [Daeron, 2021](https://doi.org/10.1029/2020GC009592)), which allows standardization under consideration of full error propagation and has been used for the InterCarb community effort ([Bernasconi et al., 2021](https://doi.org/10.1029/2020GC009588)).
This web-app allows users to discover replicate- or sample-based processing results in interactive spreadsheets and plots.

<br>

Example data is accessible from the Data-IO page, or can be downloaded from [GitHub](https://github.com/itsMig/D4Xgui/tree/main/D4Xgui/static).
Please check `INSTALLATION.md` to find help setting up D4Xgui.

<br>

In order to process post-background corrected data (Data-IO page, Upload δ⁴⁵-δ⁴⁹ replicates tab), the following columns need to be provided (`.xlsx`, `.csv`):

| `UID` | `Sample` | `Session` | `Timetag` | `d45` | `d46` | `d47` | `d48` | `d49` |
|----|----|----|----|----|----|----|----|----|

$~$

Baseline correction can be performed using a m/z47.5 half-mass cup. For this purpose either a set of equilibrated gases (via heated-gas-line), or carbonate standards (via target values) is used to determine m/z47, m/z48 and m/z49-specific scaling factors. Please upload a cycle-based spreadsheet (`.xlsx`, `.csv`) including the following columns (Data-IO page, Upload m/z44-m/z49 intensities tab):

| `UID` | `Sample` | `Session` | `Timetag` | `Replicate` |
|----|----|----|----|----|

| `raw_r44` | `raw_r45` | `raw_r46` | `raw_r47` | `raw_r48` | `raw_r49` | `raw_r47.5` |
|----|----|----|----|----|----|----|

| `raw_s44` | `raw_s45` | `raw_s46` | `raw_s47` | `raw_s48` | `raw_s49` | `raw_s47.5` |
|----|----|----|----|----|----|----|


<br><br>
Please find the [code documentation here](https://itsmig.github.io/D4Xgui/index.html).

