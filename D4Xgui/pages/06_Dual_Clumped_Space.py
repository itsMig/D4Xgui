#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import base64
import io
import itertools
import os
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import optimize as so
from scipy.stats import chi2

import tools.Pysotope_fork as tP
from tools.base_page import BasePage
from tools.commons import PLOT_PARAMS, PlotlyConfig
from tools.constants import KELVIN_OFFSET, SAMPLE_DB_PATH
from tools.ellipse import ellipse_boundary_points, ellipse_params_from_uncertainty
from tools.filters import filter_dataframe, render_sample_filter_sidebar


def _rgba_with_alpha(color_str: str, alpha: float) -> str:
    """Best-effort conversion of a Plotly colour string to rgba with *alpha*."""
    rgb_match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color_str)
    if rgb_match:
        r, g, b = rgb_match.groups()
        return f"rgba({r},{g},{b},{alpha})"
    rgba_match = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)", color_str)
    if rgba_match:
        r, g, b = rgba_match.groups()
        return f"rgba({r},{g},{b},{alpha})"
    hex_match = re.match(r"^#([0-9a-fA-F]{6})$", color_str)
    if hex_match:
        h = hex_match.group(1)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(128,128,128,{alpha})"


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
        # Plot level selection
        level_plot = st.sidebar.radio(
            "Choose plot level:", 
            ("Sample mean", "All replicates", "Together"), 
            key="level_plot"
        )
        
        # Error determination for mean plots (and "Both" mode)
        if level_plot in ("Sample mean", "Together"):
            error_dualClumped = st.sidebar.radio(
                "Error determination:",
                ("fully propagated 2SE", "fully propagated 1SE", "via long-term repeatability"),
            )
            error_mapping = {
                "fully propagated 2SE": "2SE_{mz}",
                "fully propagated 1SE": "SE_{mz}",
                "via long-term repeatability": "{mz} 2SE (longterm)",
            }
            self.sss.error_dualClumped = error_mapping[error_dualClumped]

            st.sidebar.radio(
                "Uncertainty display:",
                ("Error ellipses", "Error bars", "Both"),
                key="uncertainty_style",
            )

            if self.sss.get("uncertainty_style") in ("Error ellipses", "Both"):
                st.sidebar.radio(
                    "Covariance source:",
                    ("From replicates", "Zero (axis-aligned)"),
                    key="covariance_source",
                    help=(
                        "**From replicates** estimates the Δx–Δy correlation per sample "
                        "from paired replicate measurements (hybrid: ρ from replicates, "
                        "marginal SEs from D47crunch propagation). "
                        "**Zero** assumes no cross-isotope covariance, giving axis-aligned ellipses. "
                        "Samples with fewer than 3 replicates are excluded from ellipse rendering."
                    ),
                )

        # Additional options
        st.sidebar.checkbox("Hide legend", key="06_hide_legend")
        st.sidebar.checkbox("Lock x/y ratio", key="fix_ratio")
        
        st.sidebar.checkbox(
            label="re-process calibration", 
            value=False, 
            key="reprocCalib",
            help="D4Xgui uses the method of Fiebig(2021) to process ∆47/∆48 calibrations, "
                 "which uses the theoretical Hill(2014) polynoms which are scaled and "
                 "shifted linearly to match the data."
        )
        
        st.sidebar.checkbox(
            label="Display CO$_{2}$ equilibrium",
            value=False, 
            key="CO2_poly"
        )
        
        # Axis selection
        xy_options = self._get_available_axes()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.radio("x-axis", xy_options, 1, key="x_axis")  # Default to D48
        with col2:
            st.radio("y-axis", xy_options, 0, key="y_axis")  # Default to D47

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
            part.strip()
            for part in itertools.chain.from_iterable(
                str(value).lower().split(", ")
                for value in df[column].dropna().unique()
            )
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
        
        filter_mask = pd.Series(False, index=df.index)
        
        for filter_type in ["Project", "Type", "Mineralogy", "Publication"]:
            selected_values = self.sss.get(filter_type, [])
            if selected_values:
                for value in selected_values:
                    matching_samples = df_filter.loc[
                        df_filter[filter_type].str.contains(
                            re.escape(value), case=False, regex=True, na=False
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
        
        # Provide download link
        download_link = self._create_html_download_link(fig)
        st.markdown(download_link, unsafe_allow_html=True)

    def _create_dual_clumped_plot(self) -> go.Figure:
        """Create the main dual clumped space plot."""
        level_plot = self.sss.level_plot
        
        if level_plot == "Together":
            fig = self._create_replicate_plot()
            self._overlay_mean_traces(fig)
        elif level_plot == "Sample mean":
            fig = self._create_mean_plot()
        else:
            fig = self._create_replicate_plot()
        
        # Add calibration curves
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

    def _compute_replicate_covariance(self, x_col: str, y_col: str) -> Dict[str, Optional[float]]:
        """Compute per-sample Pearson rho from paired replicate measurements.

        Returns a dict mapping sample name to rho (or None when N < 3 or
        the correlation is undefined).
        """
        reps = self.sss.correction_output_full_dataset
        rho_map: Dict[str, Optional[float]] = {}
        for sample, grp in reps.groupby("Sample"):
            sub = grp[[x_col, y_col]].dropna()
            if len(sub) < 3:
                rho_map[sample] = None
            else:
                r = sub[[x_col, y_col]].corr().iloc[0, 1]
                rho_map[sample] = None if np.isnan(r) else float(r)
        return rho_map

    def _create_mean_plot(self) -> go.Figure:
        """Create a plot showing sample means with error bars or ellipses."""
        summary = self.sss._06_filtered_summary
        
        if len(summary) == 0:
            st.write('### Please set filter to match the available samples!')
            available_samples = sorted(
                list(self.sss.correction_output_summary['Sample'].unique())
            )
            st.markdown("  \n  ".join(available_samples))
            st.stop()

        uncertainty_style = self.sss.get("uncertainty_style", "Error bars")
        use_ellipses = uncertainty_style in ("Error ellipses", "Both")
        use_error_bars = uncertainty_style in ("Error bars", "Both")

        x_axis, y_axis = self.sss.x_axis, self.sss.y_axis
        err_template = self.sss.error_dualClumped
        x_err_col = err_template.format(mz=x_axis)
        y_err_col = err_template.format(mz=y_axis)

        # In ellipse mode, compute rho and attach to the summary for hover
        rho_map: Dict[str, Optional[float]] = {}
        if use_ellipses:
            covariance_source = self.sss.get("covariance_source", "Zero (axis-aligned)")
            if covariance_source == "From replicates":
                rho_map = self._compute_replicate_covariance(x_axis, y_axis)
            summary = summary.copy()
            summary["ρ (rho)"] = summary["Sample"].map(
                lambda s: round(rho_map[s], 3) if rho_map.get(s) is not None else None
            )

        # Prepare hover data
        hover_data = self._prepare_hover_data(ellipse_mode=use_ellipses)

        scatter_kwargs: Dict[str, Any] = dict(
            data_frame=summary,
            x=x_axis,
            y=y_axis,
            text="Sample",
            color="Sample",
            hover_data=hover_data,
            symbol="Sample",
            symbol_sequence=self.symbols,
            category_orders={"Sample": sorted(summary["Sample"].unique())},
        )

        if use_error_bars:
            scatter_kwargs["error_x"] = x_err_col
            scatter_kwargs["error_y"] = y_err_col

        fig = px.scatter(**scatter_kwargs).update_traces(mode="lines+markers")
        fig.update_traces(marker=dict(size=11))

        for trace in fig.data:
            trace.legendgroup = trace.name

        if use_error_bars:
            for trace in fig.data:
                if hasattr(trace, "error_y"):
                    trace.error_y.thickness = 0.75

        if not use_ellipses:
            return fig

        # --- Ellipse rendering ---
        # 95 % confidence ellipse: scale = sqrt(chi2_ppf(0.95, df=2)) ≈ 2.4477
        n_sigma = float(np.sqrt(chi2.ppf(0.95, 2)))
        col_is_2se = err_template != "SE_{mz}"

        trace_color_map: Dict[str, str] = {}
        for trace in fig.data:
            if trace.name:
                trace_color_map[trace.name] = trace.marker.color

        ellipse_traces = []
        for _, row in summary.iterrows():
            sample = row["Sample"]

            sx = row.get(x_err_col, np.nan)
            sy = row.get(y_err_col, np.nan)
            if np.isnan(sx) or np.isnan(sy) or sx <= 0 or sy <= 0:
                continue

            # For 2SE / longterm columns the value is already 2*SE; normalise
            # back to 1-SE for the covariance matrix, then apply n_sigma.
            if col_is_2se:
                sx_1se, sy_1se = sx / 2.0, sy / 2.0
            else:
                sx_1se, sy_1se = sx, sy

            # Use replicate rho when available (N >= 3); fall back to rho=0
            rho = rho_map.get(sample)
            cov_xy = rho * sx_1se * sy_1se if rho is not None else 0.0

            semi_major, semi_minor, angle = ellipse_params_from_uncertainty(
                sx_1se, sy_1se, cov_xy=cov_xy, n_sigma=n_sigma,
            )
            xs, ys = ellipse_boundary_points(
                row[x_axis], row[y_axis], semi_major, semi_minor, angle,
            )

            base_color = trace_color_map.get(sample, "grey")
            fill_color = _rgba_with_alpha(base_color, 0.05)

            ellipse_traces.append(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=base_color, width=1),
                fill="toself",
                fillcolor=fill_color,
                legendgroup=sample,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Add ellipse traces then reorder so they render behind the points
        n_existing = len(fig.data)
        for trace in ellipse_traces:
            fig.add_trace(trace)
        n_total = len(fig.data)
        new_order = list(range(n_existing, n_total)) + list(range(n_existing))
        fig.data = tuple(fig.data[i] for i in new_order)

        return fig

    def _overlay_mean_traces(self, fig: go.Figure) -> None:
        """Add mean ± uncertainty traces (error bars or ellipses) onto *fig*.

        Used in "Both" mode: replicates are already plotted, and this method
        layers the sample-mean markers with their uncertainty on top.
        """
        mean_fig = self._create_mean_plot()

        # Collect existing legend groups from the replicate figure so we
        # can suppress duplicate legend entries for the mean traces.
        existing_groups = {t.legendgroup or t.name for t in fig.data}

        for trace in mean_fig.data:
            if trace.name:
                trace.legendgroup = trace.name
                if trace.name in existing_groups:
                    trace.showlegend = False
            # Increase marker size slightly so means stand out over replicates
            if hasattr(trace, "marker") and trace.marker is not None:
                trace.marker.size = 14
                trace.marker.line = dict(width=1.5, color="black")
            fig.add_trace(trace)

    def _create_replicate_plot(self) -> go.Figure:
        """Create a plot showing individual replicates."""
        df = self.sss._06_filtered_reps.copy()
        
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
        
        fig = px.scatter(
            data_frame=df,
            x=self.sss.x_axis,
            y=self.sss.y_axis,
            color="Sample",
            symbol="Sample",
            symbol_sequence=self.symbols,
            hover_data=hover_data,
            category_orders={"Sample": sorted(df["Sample"].unique())},
        )

        for trace in fig.data:
            trace.legendgroup = trace.name

        return fig

    def _prepare_hover_data(self, ellipse_mode: bool = False) -> List[str]:
        """Prepare hover data for mean plots."""
        
        hover_data = ["N", "d13C_VPDB", "d18O_CO2_VSMOW"]

        if ellipse_mode:
            err_template = self.sss.error_dualClumped
            hover_data.append(err_template.format(mz=self.sss.x_axis))
            hover_data.append(err_template.format(mz=self.sss.y_axis))
            hover_data.append("ρ (rho)")

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

    def _add_calibration_curves(self, fig: go.Figure, reprocessed: bool = False) -> None:
        """Add carbonate equilibrium calibration curves to the plot."""
        if reprocessed:
            if not self._reprocess_calibration():
                return
            scaling_47, offset_47 = (
                self.sss["reprocessed_poly"][47]["a"],
                self.sss["reprocessed_poly"][47]["b"],
            )
            scaling_48, offset_48 = (
                self.sss["reprocessed_poly"][48]["a"],
                self.sss["reprocessed_poly"][48]["b"],
            )
        else:
            # Fiebig2024 calibration parameters
            scaling_47, offset_47 = 1.038, 0.1848
            scaling_48, offset_48 = 1.038, 0.1214
        
        # D49 calibration (Bernecker2023)
        scaling_49, offset_49 = 1.02, 0.56
        
        # Select appropriate scaling and functions based on axes
        x_axis, y_axis = self.sss.x_axis, self.sss.y_axis
        
        scaling_funcs = {
            "D47": (scaling_47, offset_47, tP.K47_t),
            "D48": (scaling_48, offset_48, tP.K48_t),
            "D49": (scaling_49, offset_49, tP.K49_t),
        }
        
        scaling_x, offset_x, x_func = scaling_funcs[x_axis]
        scaling_y, offset_y, y_func = scaling_funcs[y_axis]
        
        # Temperature points for calibration
        temps_c = [8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 
                   250, 300, 350, 400, 450, 500, 700, 900, 1100]
        temps_k = np.array([1 / (t + KELVIN_OFFSET) for t in temps_c])
        
        temps_y = y_func(temps_k, scaling_y, offset_y)
        temps_x = x_func(temps_k, scaling_x, offset_x)
        
        # Add temperature markers
        fig.add_trace(go.Scatter(
            x=temps_x,
            y=temps_y,
            mode="markers",
            legendgroup="legend_calib",
            name="",
            marker=dict(color="Red" if reprocessed else "Black"),
            showlegend=False,
            text=[f"{t}°C" for t in temps_c],
        ))
        
        # Add temperature labels
        fig.add_trace(go.Scatter(
            x=temps_x,
            y=temps_y,
            mode="text",
            legendgroup="legend_calib",
            name="",
            marker=dict(color="Red" if reprocessed else "Black"),
            showlegend=False,
            text=[f"{t}°C (new)" if reprocessed else f"{t}°C" for t in temps_c],
            textposition="bottom right",
        ))
        
        # Add calibration curve
        calib_range = np.array([1 / (t + KELVIN_OFFSET) for t in range(0, 1100, 1)])
        calib_y = y_func(calib_range, scaling_y, offset_y)
        calib_x = x_func(calib_range, scaling_x, offset_x)
        
        curve_name = self._get_calibration_curve_name(reprocessed, x_axis, y_axis)
        
        fig.add_trace(go.Scatter(
            x=np.round(calib_x, 6),
            y=np.round(calib_y, 6),
            legendgroup="legend_calib",
            mode="lines",
            name=curve_name,
            text=[f"{t}°C" for t in range(0, 1100, 1)],
            texttemplate="%.3f",
            line=dict(color="Red" if reprocessed else "Grey"),
        ))

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
            """CO2 equilibrium Δ47 (‰) after Wang et al. (2004)."""
            x = 1000 / (t_celsius + KELVIN_OFFSET)
            return (
                    0.003 * x ** 4
                    - 0.0438 * x ** 3
                    + 0.2553 * x ** 2
                    - 0.2195 * x
                    + 0.0616
            )

        def delta_48_equilibrium(t_celsius: np.ndarray) -> np.ndarray:
            """CO2 equilibrium Δ48 (‰) after Wang et al. (2004)."""
            factor = 1e6 / (t_celsius + KELVIN_OFFSET) ** 2
            return (
                -1.0345e-4 * factor ** 3
                + 4.22629e-3 * factor ** 2
                - 3.76112e-3 * factor
            )
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
        html_buffer = io.StringIO()
        fig.write_html(html_buffer)
        
        bytes_buffer = io.BytesIO(html_buffer.getvalue().encode())
        b64 = base64.b64encode(bytes_buffer.read()).decode()
        
        return (f'<a href="data:text/html;charset=utf-8;base64,{b64}" '
                f'download="dual_clumped_plot.html">Download plot</a>')


if __name__ == "__main__":
    page = DualClumpedSpacePage()
    page.run()