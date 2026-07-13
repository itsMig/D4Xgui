#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import base64
import hashlib
import io
import itertools
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import optimize as so

import tools.Pysotope_fork as tP
from tools.base_page import BasePage
from tools.commons import PLOT_PARAMS, PlotlyConfig
from tools.constants import KELVIN_OFFSET, SAMPLE_DB_PATH
from tools.filters import filter_dataframe, render_sample_filter_sidebar
from tools import config as app_config
from tools import d95eq_plotly as d95p
from tools.calc_temperature import TemperatureCalculator

_CALIBRATION_TEMP_MARKERS_C = [
    8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200,
    250, 300, 350, 400, 450, 500, 700, 900, 1100,
]


class DualClumpedSpacePage(BasePage):
    """Manages the Dual Clumped Space visualization page."""

    PAGE_NUMBER = 6
    PAGE_TITLE = "Dual Clumped Space"

    def __init__(self):
        """Initialize the DualClumpedSpacePage."""
        self.symbols = PLOT_PARAMS.SYMBOLS
        super().__init__()
        self._add_custom_css()
        self._validate_data_requirements()

    def _add_custom_css(self) -> None:
        """Add custom CSS styling to the page."""
        custom_css = """
        <style>
            .button-red {
                background-color: red;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
            }
            .button-grey {
                background-color: grey;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
            }
            .stPlotlyChart {
                align-content: stretch;
            }
            .main {
                align-content: center;
                height: auto;
                margin: -80px auto 0px auto;
            }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

    def _validate_data_requirements(self) -> None:
        """Validate that required data is available and processed."""
        if "correction_output_summary" not in self.sss:
            st.markdown(
                r"Please upload and process a dataset for at least two metrics "
                r"(i.e., $\Delta_{47}$ & $\Delta_{48}$) in order to discover "
                r"results in dual clumped space."
            )
            st.page_link(
                "pages/04_Processing.py", 
                label=r"$\rightarrow  \textit{Processing}$  page"
            )
            st.stop()

        if not self.sss.params_last_run["process_D47"]:
            st.markdown(r"Please process $\Delta_{{47}}$ as well to show dual clumped space!")
            st.page_link(
                "pages/04_Processing.py", 
                label=r"$\rightarrow  \textit{Processing}$  page"
            )
            st.stop()

        if not (self.sss.params_last_run["process_D48"] or self.sss.params_last_run["process_D49"]):
            st.markdown(
                r"Just $\Delta_{{47}}$ data processed. Please process $\Delta_{{48}}$ "
                r"and/or $\Delta_{{49}}$ as well to display results in dual clumped space!"
            )
            st.page_link(
                "pages/04_Processing.py", 
                label=r"$\rightarrow  \textit{Processing}$  page"
            )
            st.stop()

    def run(self) -> None:
        """Run the main application page."""
        self._setup_filtering_options()
        self._setup_plot_controls()
        self._apply_filters()
        self._display_plot()

    def _setup_filtering_options(self) -> None:
        """Set up filtering options in the sidebar."""
        # Check if sample database exists
        has_sample_db = os.path.exists(SAMPLE_DB_PATH)
        
        if has_sample_db:
            st.sidebar.toggle(
                "Select filter functionality",
                key="filter_mode",
                value=False,
            )
        else:
            self.sss["filter_mode"] = False

        if self.sss["filter_mode"] and has_sample_db:
            self._setup_database_filters()
        else:
            self._setup_text_filters()

    def _setup_database_filters(self) -> None:
        """Set up database-based filtering options."""
        col01, col02, col03, col04 = st.sidebar.columns(4)
        
        df_filter = pd.read_excel(SAMPLE_DB_PATH, engine="openpyxl")
        
        # Normalize filter data
        for col in ["Type", "Project", "Mineralogy", "Publication"]:
            df_filter[col] = df_filter[col].str.lower()
        
        # Filter to only include samples in current dataset
        all_samples = list(self.sss.correction_output_summary["Sample"].unique())
        df_filter = df_filter[df_filter["Sample"].isin(all_samples)]
        self.sss["df_filter"] = df_filter
        
        # Create filter options
        filter_options = {}
        for col in ["Type", "Project", "Mineralogy", "Publication"]:
            filter_options[col] = self._get_unique_split_values(df_filter, col)
        
        # Render filter controls
        with col01:
            st.sidebar.multiselect(
                "Project:", filter_options["Project"], None, key="Project"
            )
        with col02:
            st.sidebar.multiselect(
                "Sample type:", filter_options["Type"], None, key="Type"
            )
        with col03:
            st.sidebar.multiselect(
                "Publication:", filter_options["Publication"], None, key="Publication"
            )
        with col04:
            st.sidebar.multiselect(
                "Mineralogy:", filter_options["Mineralogy"], None, key="Mineralogy"
            )

    def _setup_text_filters(self) -> None:
        """Set up text-based filtering options."""
        render_sample_filter_sidebar("06", use_columns=True)

    def _setup_plot_controls(self) -> None:
        """Set up plot control options in the sidebar."""
        st.sidebar.radio(
            "Choose plot level:",
            ("Sample mean ±err", "Overview replicates", "Together"),
            key="level_plot",
        )

        xy_options = self._get_available_axes()
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.radio("x-axis", xy_options, 1, key="x_axis")  # Default to D48
        with col2:
            st.radio("y-axis", xy_options, 0, key="y_axis")  # Default to D47

        if self._is_d95eq_axes():
            if self._d95eq_import_ok():
                st.sidebar.radio(
                    "Uncertainty display:",
                    ("Error bars", "Ellipses", "Both"),
                    key="d95eq_uncertainty_display",
                )
                st.sidebar.number_input(
                    "ρ(Δ₄₇, Δ₄₈) significance α",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.05,
                    step=0.01,
                    format="%.3f",
                    key="d95eq_rho_alpha",
                    help=(
                        "Per-sample Pearson correlation ρ(Δ₄₇, Δ₄₈) is estimated "
                        "from the sample's own replicates. If the two-sided "
                        "t-test on ρ has p < α (and n ≥ 3), that ρ is used to "
                        "tilt the sample's confidence ellipse via the covariance "
                        "`[[σ₄₇², ρ·σ₄₇·σ₄₈], [ρ·σ₄₇·σ₄₈, σ₄₈²]]`. Otherwise the "
                        "sample is treated as ρ = 0 and its ellipse stays "
                        "axis-aligned. This is needed because Δ₄₇ and Δ₄₈ are "
                        "processed in separate D47crunch pipelines, so their "
                        "joint covariance is not otherwise preserved."
                    ),
                )
            else:
                st.sidebar.warning(
                    "D95eq is not installed. Run `pip install D95eq>=1.2.4` to enable "
                    "confidence ellipses and Δ₉₅ thermometry."
                )

        if self._uses_sample_means():
            error_dualClumped = st.sidebar.radio(
                "Error determination:",
                ("fully propagated 2SE", "fully propagated 1SE", "via long-term repeatability"),
                help=(
                    "Affects error-bar length and hover values. "
                    "**Confidence ellipses** always take a 1σ input (2SE columns "
                    "are divided by 2) and are drawn at p = 0.95, so toggling "
                    "1SE ↔ 2SE does **not** change the ellipse size. "
                    "Switching to *long-term repeatability* reads a different "
                    "uncertainty column and will change the ellipse size."
                ),
            )
            error_mapping = {
                "fully propagated 2SE": "2SE_{mz}",
                "fully propagated 1SE": "SE_{mz}",
                "via long-term repeatability": "{mz} 2SE (longterm)",
            }
            self.sss.error_dualClumped = error_mapping[error_dualClumped]

        if (
            self._is_d95eq_axes()
            and self._d95eq_import_ok()
            and self._show_d95eq_ellipses()
        ):
            st.sidebar.checkbox(
                label=r"Calculate $T_{\mathrm{eq}}$ through D95eq",
                key="d95eq_show_teq",
            )
            if self.sss.get("d95eq_show_teq", False):
                self._setup_d95eq_controls()

        st.sidebar.checkbox("Hide legend", key="06_hide_legend")
        st.sidebar.checkbox("Lock x/y ratio", key="fix_ratio")

        st.sidebar.checkbox(
            label="re-process calibration",
            value=False,
            key="reprocCalib",
            help="D4Xgui uses the method of Fiebig(2021) to process ∆47/∆48 calibrations, "
                 "which uses the theoretical Hill(2014) polynoms which are scaled and "
                 "shifted linearly to match the data.",
        )

        st.sidebar.checkbox(
            label="Display CO$_{2}$ equilibrium",
            value=False,
            key="CO2_poly",
        )



    def _is_d95eq_axes(self) -> bool:
        """True when both axes are D47 and D48 (either orientation)."""
        axes = {self.sss.get("x_axis"), self.sss.get("y_axis")}
        return axes == {"D47", "D48"}

    def _uses_sample_means(self) -> bool:
        """True when the plot includes sample-mean data (mean-only or together)."""
        level_plot = self.sss.get("level_plot", "")
        return "mean" in level_plot.lower() or level_plot == "Together"

    def _d95eq_import_ok(self) -> bool:
        """Check whether D95eq is importable."""
        try:
            import D95eq  # noqa: F401
            return True
        except ImportError:
            return False

    def _setup_d95eq_controls(self) -> None:
        """Sidebar controls for D95eq thermometry (D47+D48 axes only)."""
        if not self._is_d95eq_axes() or not self._d95eq_import_ok():
            return

        st.sidebar.number_input(
            d95p.d95eq_kinetic_slope_label(self.sss.x_axis, self.sss.y_axis),
            value=-1.0,
            format="%.2f",
            key="d95eq_diseq_slope",
            help=d95p.d95eq_kinetic_slope_help(self.sss.x_axis, self.sss.y_axis),
        )
        st.sidebar.number_input(
            f"{d95p.d95eq_kinetic_slope_label(self.sss.x_axis, self.sss.y_axis)} SE",
            value=0.1,
            format="%.2f",
            min_value=0.0,
            key="d95eq_diseq_slope_se",
        )
        st.sidebar.number_input(
            "Equilibrium p cutoff",
            value=0.05,
            format="%.3f",
            min_value=0.0,
            max_value=1.0,
            key="d95eq_p_cutoff",
        )

    def _show_d95eq_error_bars(self) -> bool:
        display = self.sss.get("d95eq_uncertainty_display", "Error bars")
        return display in ("Error bars", "Both")

    def _show_d95eq_ellipses(self) -> bool:
        display = self.sss.get("d95eq_uncertainty_display", "Error bars")
        return display in ("Ellipses", "Both")

    def _get_available_axes(self) -> List[str]:
        """Get list of available axes based on processed data."""
        xy_options = []
        if self.sss.params_last_run["process_D47"]:
            xy_options.append("D47")
        if self.sss.params_last_run["process_D48"]:
            xy_options.append("D48")
        if self.sss.params_last_run["process_D49"]:
            xy_options.append("D49")
        return xy_options

    def _get_unique_split_values(self, df: pd.DataFrame, column: str) -> List[str]:
        """Get unique values from a column that may contain comma-separated values."""
        return sorted(set(
            strip.strip(" ")
            for strip in itertools.chain(*[
                str(value).lower().split(", ")
                for value in df[column].dropna().unique()
            ])
        ))

    def _apply_filters(self) -> None:
        """Apply filters to the data."""
        corrected = self.sss.correction_output_full_dataset
        summary = self.sss.correction_output_summary
        
        self.sss._06_filtered_reps = self._filter_dataframe(corrected, "Sample")
        self.sss._06_filtered_summary = self._filter_dataframe(summary, "Sample")

    def _filter_dataframe(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply filtering logic to a DataFrame."""
        if not self.sss["filter_mode"]:
            return self._apply_text_filters(df, column)
        else:
            return self._apply_database_filters(df, column)

    def _apply_text_filters(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply text-based filters to DataFrame."""
        return filter_dataframe(
            df,
            include_str=self.sss.get("06_sample_contains", ""),
            exclude_str=self.sss.get("06_sample_not_contains", ""),
            column=column,
        )

    def _apply_database_filters(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply database-based filters to DataFrame."""
        df_filter = self.sss.get("df_filter")
        if df_filter is None:
            return df
        
        filter_mask = pd.Series([False] * len(df))
        
        for filter_type in ["Project", "Type", "Mineralogy", "Publication"]:
            selected_values = self.sss.get(filter_type, [])
            if selected_values:
                for value in selected_values:
                    # Create regex pattern for safe matching
                    value_regex = value.replace('.', r'\.').replace('(', r'\(').replace(')', r'\)')
                    
                    # Find matching samples in filter database
                    matching_samples = df_filter.loc[
                        df_filter[filter_type].str.contains(
                            f"(?i){value_regex}", regex=True, na=False
                        ), 'Sample'
                    ]
                    
                    # Update filter mask
                    sample_mask = df['Sample'].isin(matching_samples)
                    filter_mask = filter_mask | sample_mask
        
        return df[filter_mask] if filter_mask.any() else df

    def _display_plot(self) -> None:
        """Display the dual clumped space plot."""
        fig = self._create_dual_clumped_plot()
        
        st.plotly_chart(
            fig,
            config=PlotlyConfig.CONFIG,
            use_container_width=True,
        )

        if (
            self._is_d95eq_axes()
            and self._d95eq_import_ok()
            and self._show_d95eq_ellipses()
        ):
            self._render_sample_rho_expander()

        # Provide download link
        download_link = self._create_html_download_link(fig)
        st.markdown(download_link, unsafe_allow_html=True)

    def _render_sample_rho_expander(self) -> None:
        """Show the per-sample Pearson r(Δ47, Δ48) actually used for ellipses."""
        rho_map = self.sss.get("_d95eq_sample_rho_map") or {}
        if not rho_map:
            return
        alpha = float(self.sss.get("d95eq_rho_alpha", 0.05))
        rows: List[Dict[str, Any]] = []
        for sample, entry in sorted(rho_map.items()):
            rows.append({
                "Sample": sample,
                "n": entry.get("n", 0),
                "r(Δ47, Δ48)": None if entry.get("r") is None else round(entry["r"], 3),
                "p-value": None if entry.get("p") is None else float(f"{entry['p']:.3g}"),
                f"significant (α={alpha:g})": bool(entry.get("significant", False)),
                "ρ used for ellipse": round(entry.get("used_rho", 0.0), 3),
            })
        with st.expander(":rainbow[Per-sample ρ(Δ₄₇, Δ₄₈) from replicates]"):
            st.caption(
                "Pearson r estimated on each sample's replicate (Δ₄₇, Δ₄₈) pairs. "
                "Samples whose two-sided t-test on r has p ≥ α (or with fewer "
                "than 3 replicates) are drawn with ρ = 0 (axis-aligned)."
            )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    def _create_dual_clumped_plot(self) -> go.Figure:
        """Create the main dual clumped space plot."""
        level_plot = self.sss.level_plot

        if level_plot == "Together":
            fig = self._create_together_plot()
        elif "mean" in level_plot:
            fig = self._create_mean_plot()
        else:
            fig = self._create_replicate_plot()

        d95eq_active = self._is_d95eq_axes() and self._d95eq_import_ok()
        if d95eq_active:
            try:
                self._ensure_sample_legendgroups(fig)
                self._add_d95eq_calibration(fig)
                self._add_uncertainty_geometry(fig)
                if (
                    self.sss.get("d95eq_show_teq", False)
                    and self._show_d95eq_ellipses()
                ):
                    self._add_d95eq_thermometry(fig)
            except Exception as exc:
                st.sidebar.warning(f"D95eq overlay failed: {exc}")

        if not d95eq_active:
            self._add_calibration_curves(fig, reprocessed=False)

        if self.sss.get("reprocCalib", False):
            if "reprocessed_poly" not in self.sss:
                self._reprocess_calibration()
            self._add_calibration_curves(fig, reprocessed=True)
        
        # Add CO2 equilibrium if requested
        if self.sss.get("CO2_poly", False):
            self._add_co2_equilibrium(fig)
        
        # Apply layout settings
        self._apply_plot_layout(fig)
        
        return fig

    def _create_mean_plot(self) -> go.Figure:
        """Create a plot showing sample means with error bars."""
        summary = self.sss._06_filtered_summary
        
        if len(summary) == 0:
            st.write('### Please set filter to match the available samples!')
            available_samples = sorted(
                list(self.sss.correction_output_summary['Sample'].unique())
            )
            st.markdown("  \n  ".join(available_samples))
            st.stop()
        
        # Prepare hover data
        hover_data = self._prepare_hover_data()
        scatter_kwargs = dict(
            x=self.sss.x_axis,
            y=self.sss.y_axis,
            text="Sample",
            color="Sample",
            hover_data=hover_data,
            symbol="Sample",
            symbol_sequence=self.symbols,
            category_orders={"Sample": sorted(summary["Sample"].unique())},
        )
        if (not self._is_d95eq_axes()) or self._show_d95eq_error_bars():
            scatter_kwargs["error_x"] = self.sss.error_dualClumped.format(mz=self.sss.x_axis)
            scatter_kwargs["error_y"] = self.sss.error_dualClumped.format(mz=self.sss.y_axis)

        fig = px.scatter(summary, **scatter_kwargs).update_traces(mode="lines+markers")
        
        # Update marker properties
        fig.update_traces(marker=dict(size=11))
        
        # Reduce error bar thickness
        for trace in fig.data:
            if hasattr(trace, 'error_y'):
                trace.error_y.thickness = 0.75
        
        return fig

    def _create_together_plot(self) -> go.Figure:
        """Create a plot showing replicates and sample means together."""
        fig = self._create_replicate_plot()
        fig.add_traces(self._create_mean_plot().data)
        return fig

    def _create_replicate_plot(self) -> go.Figure:
        """Create a plot showing individual replicates."""
        level_plot = self.sss.level_plot
        df = (self.sss._06_filtered_summary if "mean" in level_plot 
              else self.sss._06_filtered_reps)
        
        # Ensure numeric data types
        try:
            df[self.sss.x_axis] = pd.to_numeric(df[self.sss.x_axis])
            df[self.sss.y_axis] = pd.to_numeric(df[self.sss.y_axis])
        except (KeyError, ValueError):
            df[self.sss.x_axis] = pd.to_numeric(df[f"{self.sss.x_axis} CDES"])
            df[self.sss.y_axis] = pd.to_numeric(df[f"{self.sss.y_axis} CDES"])
        
        hover_data = ["Session", "Timetag", "d13C_VPDB", "d18O_VSMOW"]
        if 'n_acqu' in df:
            hover_data.append('n_acqu')

        scatter_kwargs = dict(
            data_frame=df,
            x=self.sss.x_axis,
            y=self.sss.y_axis,
            color="Sample",
            symbol="Sample",
            symbol_sequence=self.symbols,
            hover_data=hover_data,
            category_orders={"Sample": sorted(df["Sample"].unique())},
        )
        if (not self._is_d95eq_axes()) or self._show_d95eq_error_bars():
            x_se = f"SE_{self.sss.x_axis}"
            y_se = f"SE_{self.sss.y_axis}"
            if x_se in df.columns and y_se in df.columns:
                scatter_kwargs["error_x"] = x_se
                scatter_kwargs["error_y"] = y_se

        fig = px.scatter(**scatter_kwargs)
        
        return fig

    def _prepare_hover_data(self) -> List[str]:
        """Prepare hover data for mean plots."""
        
        hover_data = ["N", "d13C_VPDB", "d18O_CO2_VSMOW"]
        
        if not self.sss.params_last_run["process_D47"]:
            return hover_data
        
        for calib in self.sss["04_used_calibs"]:
            error_type = "2SE" if "2" in self.sss.error_dualClumped.format(mz="D47") else "1SE"
            hover_data.extend([
                f"T(min, {error_type}), {calib}",
                f"T(mean), {calib}",
                f"T(max, {error_type}), {calib}"
            ])
        return hover_data

    def _get_d95eq_plot_df(self) -> pd.DataFrame:
        """DataFrame used for D95eq geometry (mean or replicate level)."""
        if self._uses_sample_means():
            return self.sss._06_filtered_summary
        return self.sss._06_filtered_reps

    def _d95eq_sample_rho_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute per-sample Pearson r(Δ47, Δ48) on the filtered replicate rows
        and stash it in session_state. Non-significant / underpowered samples
        yield ``used_rho = 0`` (flat, axis-aligned ellipse).
        """
        alpha = float(self.sss.get("d95eq_rho_alpha", 0.05))
        reps = self.sss.get("_06_filtered_reps")
        if reps is None or reps.empty or not {"D47", "D48", "Sample"}.issubset(reps.columns):
            rho_map: Dict[str, Dict[str, Any]] = {}
        else:
            rho_map = d95p.sample_pearson_map(reps, sig_level=alpha)
        self.sss["_d95eq_sample_rho_map"] = rho_map
        return rho_map

    def _d95eq_used_rho_by_sample(self) -> Dict[str, float]:
        return {
            sample: entry.get("used_rho", 0.0)
            for sample, entry in self._d95eq_sample_rho_map().items()
        }

    def _resolve_d95eq_se_columns(self) -> Tuple[str, str, float, float]:
        """Return D47/D48 SE column names and divisors for 1σ."""
        if self._uses_sample_means():
            col_d47, div_d47 = d95p.resolve_se_columns(self.sss.error_dualClumped, "D47")
            col_d48, div_d48 = d95p.resolve_se_columns(self.sss.error_dualClumped, "D48")
            return col_d47, col_d48, div_d47, div_d48
        se_d47, se_d48 = d95p.replicate_se_columns()
        return se_d47, se_d48, 1.0, 1.0

    def _validate_d95eq_columns(self, df: pd.DataFrame) -> bool:
        se_d47, se_d48, _, _ = self._resolve_d95eq_se_columns()
        required = ["D47", "D48", se_d47, se_d48]
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.sidebar.warning(
                f"D95eq requires columns {', '.join(required)}; missing: {', '.join(missing)}"
            )
            return False
        return True

    def _d95eq_cache_key(self, df: pd.DataFrame) -> str:
        se_d47, se_d48, div_d47, div_d48 = self._resolve_d95eq_se_columns()
        used_rho = self._d95eq_used_rho_by_sample()
        payload = "|".join([
            str(len(df)),
            str(sorted(df["Sample"].astype(str).tolist())) if "Sample" in df.columns else "",
            str(df[["D47", "D48", se_d47, se_d48]].round(6).to_csv(index=False)),
            str(self.sss.get("d95eq_diseq_slope", -1.0)),
            str(self.sss.get("d95eq_diseq_slope_se", 0.1)),
            str(self.sss.get("d95eq_p_cutoff", 0.05)),
            str(self.sss.get("d95eq_rho_alpha", 0.05)),
            str(sorted((s, round(r, 6)) for s, r in used_rho.items())),
            str(self.sss.get("x_axis")),
            str(self.sss.get("y_axis")),
            str(div_d47),
            str(div_d48),
            str(TemperatureCalculator.hill_affine_to_d4x_coefs(
                TemperatureCalculator.POLY_63_COEFFS,
                TemperatureCalculator.FIEBIG2024_D47_SCALING,
                TemperatureCalculator.FIEBIG2024_D47_OFFSET,
            )),
            str(TemperatureCalculator.hill_affine_to_d4x_coefs(
                TemperatureCalculator.POLY_64_COEFFS,
                TemperatureCalculator.FIEBIG2024_D48_SCALING,
                TemperatureCalculator.FIEBIG2024_D48_OFFSET,
            )),
        ])
        return hashlib.md5(payload.encode()).hexdigest()

    def _compute_d95eq_thermometry(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run Engine thermometry with session-state caching."""
        cache_key = self._d95eq_cache_key(df)
        cached = self.sss.get("_d95eq_thermometry_cache")
        if cached and cached.get("key") == cache_key:
            return cached["result"]

        import uncertainties as uc

        se_d47, se_d48, div_d47, div_d48 = self._resolve_d95eq_se_columns()
        d47_u, d48_u, row_index = d95p.df_to_d95eq_uarrays(
            df,
            se_d47=se_d47,
            se_d48=se_d48,
            se_div_d47=div_d47,
            se_div_d48=div_d48,
            sample_rho=self._d95eq_used_rho_by_sample(),
        )
        if d47_u.size == 0:
            raise ValueError("No valid D47/D48 pairs with uncertainties for D95eq.")

        n_samples = d47_u.size
        compute = lambda: self._run_d95eq_engine(d47_u, d48_u, uc)
        if n_samples > 20:
            with st.spinner(f"Computing D95eq thermometry for {n_samples} points..."):
                result = compute()
        else:
            result = compute()

        result["row_index"] = row_index
        result["samples"] = df.loc[row_index, "Sample"].tolist()
        self.sss["_d95eq_thermometry_cache"] = {"key": cache_key, "result": result}
        return result

    @staticmethod
    def _get_d95eq_engine():
        from D95eq import Engine
        d47_coefs, d48_coefs = d95p.fiebig2024_engine_coefs()
        return Engine(D47_coefs=d47_coefs, D48_coefs=d48_coefs)

    def _run_d95eq_engine(self, d47_u, d48_u, uc) -> Dict[str, Any]:
        engine = self._get_d95eq_engine()
        d47eq, d48eq, p_values = engine.nearest_D47eq(d47_u, d48_u)
        plot_slope = self.sss.get("d95eq_diseq_slope", -1.0)
        plot_slope_se = self.sss.get("d95eq_diseq_slope_se", 0.1)
        d95eq_slope, d95eq_slope_se = d95p.plot_kinetic_slope_to_d95eq(
            plot_slope,
            plot_slope_se,
            self.sss.x_axis,
            self.sss.y_axis,
        )
        slope = uc.ufloat(d95eq_slope, d95eq_slope_se)
        d47p, d48p = engine.projected_D47eq(d47_u, d48_u, slope)
        teq = engine.T_as_function_of_D47(d47eq)
        tkp = engine.T_as_function_of_D47(d47p)
        teq_asym = d95p.teq_asymmetric_from_pdf(engine, d47eq)
        tkp_asym = d95p.teq_asymmetric_from_pdf(engine, d47p)
        return {
            "engine": engine,
            "d47_u": d47_u,
            "d48_u": d48_u,
            "d47eq": d47eq,
            "d48eq": d48eq,
            "p_values": p_values,
            "d47p": d47p,
            "d48p": d48p,
            "teq": teq,
            "tkp": tkp,
            "teq_asym": teq_asym,
            "tkp_asym": tkp_asym,
        }

    def _add_d95eq_calibration(self, fig: go.Figure) -> None:
        """Add D95eq carbonate equilibrium curve and optional 95% confidence band."""
        engine = self._get_d95eq_engine()
        show_confidence = d95p.coefs_have_calibration_uncertainties(engine.D47_coefs)
        data, _ = engine.plot_D95_equilibrium(
            Tmarkers=_CALIBRATION_TEMP_MARKERS_C,
            Tmax=1100,
            NT=1101,
            show_Tmarker_labels=False,
            show_confidence=show_confidence,
            confidence_pvalue=0.95,
            ax=None,
        )
        calib_style = d95p.calibration_style(app_config.get("theme", "Dark") == "Dark")
        if show_confidence:
            band = engine.plot_D95_confidence_band(p=0.95, plot=False)
            d95p.add_confidence_band(
                fig,
                band,
                self.sss.x_axis,
                self.sss.y_axis,
                fillcolor=calib_style["band_fillcolor"],
                line_color=calib_style["band_line_color"],
            )
        d95p.add_equilibrium_curve(
            fig,
            data,
            self.sss.x_axis,
            self.sss.y_axis,
            line_color=calib_style["line_color"],
            calibration_temps_c=_CALIBRATION_TEMP_MARKERS_C,
        )
        d95p.add_equilibrium_temperature_markers(
            fig,
            data,
            self.sss.x_axis,
            self.sss.y_axis,
            marker_color=calib_style["marker_color"],
            temp_markers_c=_CALIBRATION_TEMP_MARKERS_C,
        )

    @staticmethod
    def _ensure_sample_legendgroups(fig: go.Figure) -> None:
        """Bind sample marker traces to legend groups named after the sample."""
        for trace in fig.data:
            if trace.name and trace.mode and "markers" in trace.mode:
                trace.legendgroup = trace.name

    def _add_uncertainty_geometry(self, fig: go.Figure) -> None:
        if not self._show_d95eq_ellipses():
            return

        df = self._get_d95eq_plot_df()
        if not self._validate_d95eq_columns(df):
            return

        se_d47, se_d48, div_d47, div_d48 = self._resolve_d95eq_se_columns()
        d47_u, d48_u, row_index = d95p.df_to_d95eq_uarrays(
            df,
            se_d47=se_d47,
            se_d48=se_d48,
            se_div_d47=div_d47,
            se_div_d48=div_d48,
            sample_rho=self._d95eq_used_rho_by_sample(),
        )
        if d47_u.size == 0:
            return

        ellipses = d95p.compute_conf_ellipses(d47_u, d48_u)
        sample_names = df.loc[row_index, "Sample"].tolist()
        colors = d95p.sample_color_map(fig)
        d95p.add_conf_ellipses(
            fig,
            ellipses,
            color=colors,
            sample_names=sample_names,
            x_axis=self.sss.x_axis,
            y_axis=self.sss.y_axis,
        )

    def _add_d95eq_thermometry(self, fig: go.Figure) -> None:
        df = self._get_d95eq_plot_df()
        if not self._validate_d95eq_columns(df):
            return

        result = self._compute_d95eq_thermometry(df)
        p_cutoff = self.sss.get("d95eq_p_cutoff", 0.05)
        p_values = result["p_values"]
        eq_mask = p_values >= p_cutoff
        diseq_mask = ~eq_mask

        diseq_color = "rgb(204, 0, 102)"
        sample_names = result["samples"]
        colors = d95p.sample_color_map(fig)

        self._add_projection_ellipses(
            fig, result["d47eq"], result["d48eq"], eq_mask, colors, sample_names
        )
        self._add_projection_ellipses(
            fig, result["d47p"], result["d48p"], diseq_mask, colors, sample_names
        )
        self._merge_d95eq_hover_into_traces(fig, df, result, p_cutoff)
        self._add_d95eq_projection_arrows(fig, result, p_cutoff, diseq_color)

    def _add_projection_ellipses(
        self,
        fig: go.Figure,
        d47_u,
        d48_u,
        mask: np.ndarray,
        colors: Dict[str, str],
        sample_names: List[str],
    ) -> None:
        if not np.any(mask):
            return
        for i in np.where(mask)[0]:
            point_mask = np.zeros(len(sample_names), dtype=bool)
            point_mask[i] = True
            subset_d47 = d95p.subset_uarray(d47_u, point_mask)
            subset_d48 = d95p.subset_uarray(d48_u, point_mask)
            if subset_d47.size == 0:
                continue
            ellipses = d95p.compute_conf_ellipses(subset_d47, subset_d48)
            d95p.add_conf_ellipses(
                fig,
                ellipses,
                color=colors,
                sample_names=[sample_names[i]],
                opacity=0.65,
                line_width=1.25,
                x_axis=self.sss.x_axis,
                y_axis=self.sss.y_axis,
            )

    def _build_d95eq_row_hover_map(
        self,
        result: Dict[str, Any],
        p_cutoff: float,
    ) -> Dict[Any, str]:
        """Map dataframe row index to D95eq thermometry hover lines."""
        row_index = result["row_index"]
        samples = result.get("samples", [])
        teq_asym = result.get("teq_asym", [])
        tkp_asym = result.get("tkp_asym", [])
        p_values = result["p_values"]
        rho_map = self.sss.get("_d95eq_sample_rho_map") or {}
        mapping: Dict[Any, str] = {}
        for i, p_val in enumerate(p_values):
            sample = samples[i] if i < len(samples) else None
            corr_line = self._format_p_corr_line(rho_map.get(str(sample), {}))
            if p_val >= p_cutoff:
                label = d95p.format_asym_temperature(*teq_asym[i])
                mapping[row_index[i]] = (
                    f"Teq = {label}<br>"
                    f"p_equ = {p_val:.3f}"
                    f"{corr_line}"
                )
            else:
                label = d95p.format_asym_temperature(*tkp_asym[i])
                mapping[row_index[i]] = (
                    f"T_kinetic = {label}<br>"
                    f"p_equ = {p_val:.2e}"
                    f"{corr_line}"
                )
        return mapping

    @staticmethod
    def _format_p_corr_line(rho_entry: Dict[str, Any]) -> str:
        """Render one hover line describing the per-sample ρ(Δ47, Δ48) test."""
        p_corr = rho_entry.get("p")
        n = rho_entry.get("n")
        used_rho = rho_entry.get("used_rho", 0.0)
        significant = rho_entry.get("significant", False)
        if p_corr is None:
            n_txt = f" (n={n})" if n is not None else ""
            return f"<br>p_corr = n/a{n_txt}, ρ = 0"
        tag = f"ρ = {used_rho:.2f}" if significant else f"n.s., ρ = 0"
        return f"<br>p_corr = {p_corr:.3g}, {tag}"

    def _merge_d95eq_hover_into_traces(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        result: Dict[str, Any],
        p_cutoff: float,
    ) -> None:
        """Append D95eq thermometry fields to existing sample hover tooltips."""
        row_hover = self._build_d95eq_row_hover_map(result, p_cutoff)
        if not row_hover:
            return

        together_means_only = self.sss.get("level_plot") == "Together"
        for trace in fig.data:
            if not trace.name or not trace.mode or "markers" not in trace.mode:
                continue
            if together_means_only and "lines" not in trace.mode:
                continue

            sample_df = df[df["Sample"] == trace.name]
            if sample_df.empty or len(sample_df) != len(trace.x):
                continue

            suffixes = [
                f"<br>{row_hover[idx]}" if idx in row_hover else ""
                for idx in sample_df.index
            ]
            if not any(suffixes):
                continue
            self._append_trace_hover_suffixes(trace, suffixes)

    @staticmethod
    def _append_trace_hover_suffixes(trace: go.Scatter, suffixes: List[str]) -> None:
        """Extend a Plotly Express trace hovertemplate with extra customdata."""
        cd = trace.customdata
        if cd is None:
            cd = np.empty((len(trace.x), 0), dtype=object)
        else:
            cd = np.asarray(cd, dtype=object)
            if cd.ndim == 1:
                cd = cd.reshape(-1, 1)
        suffix_col = np.array(suffixes, dtype=object).reshape(-1, 1)
        trace.customdata = np.hstack([cd, suffix_col])
        col_idx = trace.customdata.shape[1] - 1
        template = trace.hovertemplate or "%{x}<br>%{y}<extra></extra>"
        insert = f"%{{customdata[{col_idx}]}}"
        if "<extra>" in template:
            trace.hovertemplate = template.replace("<extra>", f"{insert}<extra>")
        else:
            trace.hovertemplate = template + insert

    def _add_d95eq_projection_arrows(
        self,
        fig: go.Figure,
        result: Dict[str, Any],
        p_cutoff: float,
        diseq_color: str,
    ) -> None:
        p_values = result["p_values"]
        sample_names = result["samples"]
        for i, p_val in enumerate(p_values):
            if p_val >= p_cutoff:
                proj_d47 = result["d47eq"][i]
                proj_d48 = result["d48eq"][i]
            else:
                proj_d47 = result["d47p"][i]
                proj_d48 = result["d48p"][i]

            x0, y0 = d95p.map_d47d48_point_to_axes(
                result["d47_u"][i].n,
                result["d48_u"][i].n,
                self.sss.x_axis,
                self.sss.y_axis,
            )
            x1, y1 = d95p.map_d47d48_point_to_axes(
                proj_d47.n,
                proj_d48.n,
                self.sss.x_axis,
                self.sss.y_axis,
            )
            color = "rgb(0, 128, 0)" if p_val >= p_cutoff else diseq_color
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.55,
                    legendgroup=sample_names[i],
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    def _add_fiebig24_curve(
        self,
        fig: go.Figure,
        scaling_47: float,
        offset_47: float,
        scaling_48: float,
        offset_48: float,
        curve_name: str,
        line_color: str = "Grey",
        line_dash: str = "solid",
        legendgroup: str = "legend_calib",
        show_markers: bool = True,
        marker_color: str = "Black",
        marker_symbol: str = "diamond-open",
        temp_label_suffix: str = "",
        temp_label_position: str = "top left",
    ) -> None:
        """Add a Hill×affine Fiebig2024 carbonate equilibrium curve to the plot."""
        scaling_49, offset_49 = 1.02, 0.56
        x_axis, y_axis = self.sss.x_axis, self.sss.y_axis

        scaling_funcs = {
            "D47": (scaling_47, offset_47, tP.K47_t),
            "D48": (scaling_48, offset_48, tP.K48_t),
            "D49": (scaling_49, offset_49, tP.K49_t),
        }

        scaling_x, offset_x, x_func = scaling_funcs[x_axis]
        scaling_y, offset_y, y_func = scaling_funcs[y_axis]

        temps_c = _CALIBRATION_TEMP_MARKERS_C
        temps_k = np.array([1 / (t + KELVIN_OFFSET) for t in temps_c])
        temps_y = y_func(temps_k, scaling_y, offset_y)
        temps_x = x_func(temps_k, scaling_x, offset_x)

        if show_markers:
            marker_hover = d95p.axis_calibration_hover_labels(
                temps_c, temps_x, temps_y, x_axis, y_axis,
            )
            fig.add_trace(go.Scatter(
                x=temps_x,
                y=temps_y,
                mode="markers",
                legendgroup=legendgroup,
                name="",
                marker=dict(color=marker_color, symbol=marker_symbol),
                showlegend=False,
                text=marker_hover,
                hovertemplate="%{text}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=temps_x,
                y=temps_y,
                mode="text",
                legendgroup=legendgroup,
                name="",
                showlegend=False,
                text=[f"{lbl}{temp_label_suffix}" for lbl in d95p.calibration_temp_labels(temps_c)],
                textposition=temp_label_position,
            ))

        calib_range = np.array([1 / (t + KELVIN_OFFSET) for t in range(0, 1100, 1)])
        calib_y = y_func(calib_range, scaling_y, offset_y)
        calib_x = x_func(calib_range, scaling_x, offset_x)
        calib_temps = list(range(0, 1100, 1))
        curve_hover = d95p.axis_calibration_hover_labels(
            calib_temps, calib_x, calib_y, x_axis, y_axis,
        )

        fig.add_trace(go.Scatter(
            x=np.round(calib_x, 6),
            y=np.round(calib_y, 6),
            legendgroup=legendgroup,
            mode="lines",
            name=curve_name,
            text=curve_hover,
            hovertemplate="%{text}<extra></extra>",
            line=dict(color=line_color, dash=line_dash, width=1.5),
        ))

    def _add_calibration_curves(self, fig: go.Figure, reprocessed: bool = False) -> None:
        """Add carbonate equilibrium calibration curves to the plot."""
        if reprocessed:
            if not self._reprocess_calibration():
                return
            scaling_47 = self.sss["reprocessed_poly"][47]["a"]
            offset_47 = self.sss["reprocessed_poly"][47]["b"]
            scaling_48 = self.sss["reprocessed_poly"][48]["a"]
            offset_48 = self.sss["reprocessed_poly"][48]["b"]
            curve_name = self._get_calibration_curve_name(reprocessed, self.sss.x_axis, self.sss.y_axis)
            self._add_fiebig24_curve(
                fig,
                scaling_47=scaling_47,
                offset_47=offset_47,
                scaling_48=scaling_48,
                offset_48=offset_48,
                curve_name=curve_name,
                line_color="Red",
                legendgroup="legend_calib",
                show_markers=True,
                marker_color="Red",
                temp_label_suffix=" (new)" if reprocessed else "",
            )
            return

        curve_name = self._get_calibration_curve_name(False, self.sss.x_axis, self.sss.y_axis)
        self._add_fiebig24_curve(
            fig,
            scaling_47=TemperatureCalculator.FIEBIG2024_D47_SCALING,
            offset_47=TemperatureCalculator.FIEBIG2024_D47_OFFSET,
            scaling_48=TemperatureCalculator.FIEBIG2024_D48_SCALING,
            offset_48=TemperatureCalculator.FIEBIG2024_D48_OFFSET,
            curve_name=curve_name,
            line_color="Grey",
            legendgroup="legend_calib",
            show_markers=True,
            marker_color="Black",
        )

    def _get_calibration_curve_name(self, reprocessed: bool, x_axis: str, y_axis: str) -> str:
        """Get the appropriate name for the calibration curve."""
        if reprocessed:
            return "Carbonate equilibrium (reprocessed)"
        elif x_axis == "D49" or y_axis == "D49":
            return "Carbonate equilibrium (Bernecker2023/Fiebig2024)"
        else:
            return "Carbonate equilibrium (Fiebig2024)"

    def _add_co2_equilibrium(self, fig: go.Figure) -> None:
        """Add CO2 equilibrium curve to the plot."""
        def delta_47_equilibrium(t_celsius: np.ndarray) -> np.ndarray:
            """
            Calculate CO2 equilibrium Δ47 (in ‰) using Cao & Liu (2012).
            Input: t_celsius (temperature in degrees Celsius)
            Output: Δ47 (per mil, ‰)
            """
            # t_kelvin = t_celsius + 273.15
            # return 25932 / (t_kelvin ** 2) + 266.6 / t_kelvin - 0.2446
            _="""
            Calculate CO2 equilibrium Δ47 (in ‰) using Wang et al. (2004).
            Input: t_celsius (temperature in degrees Celsius)
            Output: Δ47 (per mil, ‰)
            """
            # t_kelvin = t_celsius + 273.15
            #return 24952 / (t_kelvin ** 2) + 325.6 / t_kelvin - 0.365
            #return 0.003 * (1000. / t_kelvin)** 4 - 0.0438 * (1000. / t_kelvin)** 3 + 0.2443 * (1000. / t_kelvin)** 2 - 0.2195 * (
            #             1000. / t_kelvin) + 0.06161
            x = 1000 / (t_celsius + KELVIN_OFFSET)
            return (
                    0.003 * x ** 4
                    - 0.0438 * x ** 3
                    + 0.2553 * x ** 2
                    - 0.2195 * x
                    + 0.0616
            )

        def delta_48_equilibrium(t_celsius: np.ndarray) -> np.ndarray:
            """
            Calculate CO2 equilibrium Δ48 (in ‰) using Cao & Liu (2012).
            Input: t_celsius (temperature in degrees Celsius)
            Output: Δ48 (per mil, ‰)
            """
            t_kelvin = t_celsius + KELVIN_OFFSET
            # factor = 1e6 / (t_kelvin ** 2)
            # return (
            #     -1.0316e-4 * (factor ** 3)
            #     + 4.2175e-3 * (factor ** 2)
            #     - 3.7502e-3 * factor
            # )
            _="""
            Calculate CO2 equilibrium Δ48 (in ‰) using Wang et al. (2004).
            Input: t_celsius (temperature in degrees Celsius)
            Output: Δ48 (per mil, ‰)
            """
            #t_kelvin = t_celsius + 273.15
            factor = 1e6 / (t_kelvin ** 2)
            # return (
            #    -9.154e-5 * (factor ** 3)
            #    + 3.707e-3 * (factor ** 2)
            #    - 3.522e-3 * factor
            # )
            term1 = -1.0345e-4 * (1e6 / t_kelvin ** 2) ** 3
            term2 = 4.22629e-3 * (1e6 / t_kelvin ** 2) ** 2
            term3 = -3.76112e-3 * (1e6 / t_kelvin ** 2)
            return term1 + term2 + term3
            # return -1.0345 * 10 ** -4 * (10 ** 6. / (t_kelvin ** 2))** 3 + 4.22629 * 10 ** -3 * (
            #             10 ** 6. / (t_kelvin ** 2))** 2 - 3.76112 * 10 ** -3 * (10 ** 6. / (t_kelvin ** 2))
            #
        temp_range = np.arange(0, 1200, 1)
        d47_values = delta_47_equilibrium(temp_range)
        d48_values = delta_48_equilibrium(temp_range)
        
        # Add CO2 equilibrium curve
        fig.add_trace(go.Scatter(
            x=d48_values, 
            y=d47_values, 
            mode='lines', 
            name='CO2 equilibrium (Dennis2011, Fiebig2019 after Wang2004)',
            text=[f"{t}°C" for t in temp_range],
        ))
        
        # Add temperature labels for CO2 equilibrium
        temp_labels = np.array([0, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                               150, 200, 250, 300, 350, 400, 450, 500, 700, 900, 1100, 1200])
        
        fig.add_trace(go.Scatter(
            x=delta_48_equilibrium(temp_labels),
            y=delta_47_equilibrium(temp_labels),
            mode="text",
            legendgroup="legend_calib",
            name="",
            marker=dict(color="Blue"),
            showlegend=False,
            text=[f"{t}°C" for t in temp_labels],
            textposition="bottom right",
        ))

    def _apply_plot_layout(self, fig: go.Figure) -> None:
        """Apply layout settings to the plot."""
        # Fix x/y ratio if requested
        if self.sss.get("fix_ratio", False):
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        # Set axis ranges based on filtered data
        if "_06_filtered_reps" in self.sss:
            x_data = self.sss._06_filtered_reps[self.sss.x_axis]
            y_data = self.sss._06_filtered_reps[self.sss.y_axis]
            
            x_pad = 0.07
            y_pad = 0.02
            
            fig.update_layout(
                xaxis=dict(range=[x_data.min() - x_pad, x_data.max() + x_pad]),
                yaxis=dict(range=[y_data.min() - y_pad, y_data.max() + y_pad]),
            )
        
        # Apply general layout settings
        scale = self.sss.params_last_run['scale']
        x_title = f"∆<sub>{self.sss.x_axis.replace('D','')}, {scale}</sub> [‰]"
        y_title = f"∆<sub>{self.sss.y_axis.replace('D','')}, {scale}</sub> [‰]"
        
        fig.update_layout(
            height=750,
            margin=dict(r=40, t=40),
            xaxis=dict(title=x_title),
            yaxis=dict(title=y_title),
            hoverlabel=dict(font_size=20),
            legend=dict(font_size=15),
            legend_title=dict(font_size=25),
            showlegend=not self.sss.get("06_hide_legend", False),
        )
        
        # Update trace and axis styling
        fig.update_traces(textfont_size=15)
        fig.update_xaxes(
            showline=True, linewidth=2, linecolor="grey", mirror=True,
            title_font=dict(size=25), tickfont=dict(size=20)
        )
        fig.update_yaxes(
            showline=True, linewidth=2, linecolor="grey", mirror=True,
            title_font=dict(size=25), tickfont=dict(size=20)
        )

    def _reprocess_calibration(self) -> bool:
        """Reprocess calibration data using available calibration samples."""
        df = self.sss.correction_output_summary
        self.sss["reprocessed_poly"] = {}
        
        # Predefined calibration temperatures
        preset_temperatures = {
            "ETH-1-1100": 1100, "ETH-2-1100": 1100, "LGB-2": 7.9,
            "DHC2-8": 33.7, "DHC2-3": 33.7, "DVH-2": 33.7,
            "CA120": 120, "CA170": 170, "CA200": 200,
            "CA250A": 250, "CA250B": 250, "CM351": 727,
            "DH11": 33.7, "DH11-109_4": 33.7, "DH11-141_6": 33.7,
            "DH11-187": 33.7, "DH11-19-7": 33.7, "DH11-201_3": 33.7,
            "DH11-44_5": 33.7, "DH11-73": 33.7,
            'ETH1-800': 800, 'ETH2-800_72h': 800, 'MERCK-800_48h': 800,
        }
        
        # Filter to calibration samples only
        calib_df = df.loc[df["Sample"].isin(preset_temperatures)]
        
        if len(calib_df) == 0:
            info_msg = (f'None of the pre-defined calibration samples included in the results: '
                       f'{", ".join(preset_temperatures.keys())}')
            with st.expander(":rainbow[Calibration results]"):
                st.write(info_msg, unsafe_allow_html=True)
            return False
        
        # Add temperature data
        calib_df = calib_df.copy()
        calib_df["T_C"] = calib_df["Sample"].map(preset_temperatures)
        calib_df["T_1K"] = 1 / (calib_df["T_C"] + KELVIN_OFFSET)
        
        info_msg = 'The following calibration samples are included in the results:<br>'
        used_temps = {sample: preset_temperatures[sample] 
                     for sample in calib_df["Sample"] if sample in preset_temperatures}
        for sample, temp in used_temps.items():
            info_msg += f"{temp}°C = {sample}<br>"
        
        # Fit polynomials for D47 and D48
        for mz in (47, 48):
            popt, info_msg = self._fit_calibration_polynomial(calib_df, mz, info_msg)
            self.sss["reprocessed_poly"][mz] = {"a": popt[0], "b": popt[1]}
        
        with st.expander(":rainbow[Calibration results]"):
            st.write(info_msg, unsafe_allow_html=True)
        
        return True

    def _fit_calibration_polynomial(self, df: pd.DataFrame, mz: int, info_msg: str) -> Tuple[np.ndarray, str]:
        """Fit polynomial calibration for a specific mass."""
        # Hill et al. 2014 polynomial coefficients
        poly_coeffs = {
            47: (-5.896755e00, -3.520888e03, 2.391274e07, -3.540693e09),
            48: (6.001624e00, -1.298978e04, 8.995634e06, -7.422972e08),
            49: (-6.741e00, -1.950e04, 5.845e07, -8.093e09),
        }
        
        poly = poly_coeffs[mz]
        x = df["T_1K"]
        y = df[f"D{mz}"]
        sigma = df[f"SE_D{mz}"]
        
        def calibration_function(x_vals: np.ndarray, a: float, b: float) -> np.ndarray:
            """Calibration function using Hill polynomial."""
            poly_vals = (poly[0] * x_vals + poly[1] * x_vals**2 + 
                        poly[2] * x_vals**3 + poly[3] * x_vals**4)
            return (poly_vals * a) + b
        
        # Perform curve fitting
        popt, pcov = so.curve_fit(
            calibration_function,
            xdata=x,
            ydata=y,
            sigma=sigma,
        )
        
        # Calculate R²
        a, b = popt
        n = len(x)
        y_pred = calibration_function(x, a, b)
        r2 = 1.0 - (sum((y - y_pred) ** 2) / ((n - 1.0) * np.var(y, ddof=1)))
        
        info_msg += (f"<br>∆{mz}<br>Optimal Values: a={a:.6f} b={b:.6f}   "
                    f"R²: {r2:.4f}")
        
        return popt, info_msg

    def _create_html_download_link(self, fig: go.Figure) -> str:
        """Create a download link for the plot as HTML."""
        import plotly.io as pio
        
        #pio.templates.default = "plotly"
        html_buffer = io.StringIO()
        fig.write_html(html_buffer)
        
        bytes_buffer = io.BytesIO(html_buffer.getvalue().encode())
        b64 = base64.b64encode(bytes_buffer.read()).decode()
        
        return (f'<a href="data:text/html;charset=utf-8;base64,{b64}" '
                f'download="dual_clumped_plot.html">Download plot</a>')

    @staticmethod
    def k47_temperature_function(d47: float) -> float:
        """Calculate temperature from D47 values using calibration function."""
        scaling_47 = TemperatureCalculator.FIEBIG2024_D47_SCALING
        offset_47 = TemperatureCalculator.FIEBIG2024_D47_OFFSET
        
        def polynomial_4th_order(coeffs: Tuple[float, ...], x: float) -> float:
            """Calculate 4th order polynomial."""
            return (coeffs[0] * x + coeffs[1] * x**2 + 
                   coeffs[2] * x**3 + coeffs[3] * x**4)
        
        poly_63 = (-5.896755e00, -3.520888e03, 2.391274e07, -3.540693e09)
        poly_vals = polynomial_4th_order(poly_63, d47)
        
        return (poly_vals * scaling_47) + offset_47

    @staticmethod
    def find_d47_temperature(temp: float, args: Dict[str, float]) -> float:
        """Find D47 temperature using optimization."""
        return abs(args["D47 CDES"] - DualClumpedSpacePage.k47_temperature_function(1 / (temp + KELVIN_OFFSET)))


if __name__ == "__main__":
    page = DualClumpedSpacePage()
    page.run()