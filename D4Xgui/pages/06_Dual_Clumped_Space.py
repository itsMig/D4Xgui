#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import base64
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
from tools import sidebar_logo
from tools.commons import PLOT_PARAMS, PlotlyConfig
from tools.page_config import set_page_config
from tools.sidebar_logo import SidebarLogoManager
from tools.authenticator import Authenticator


class DualClumpedSpacePage:
    """Manages the Dual Clumped Space visualization page."""

    def __init__(self):
        """Initialize the DualClumpedSpacePage."""
        self.sss = st.session_state
        self.symbols = PLOT_PARAMS.SYMBOLS
        self._setup_page()
        self._validate_data_requirements()

    def _setup_page(self) -> None:
        """Set up page configuration, logo, and authentication."""
        st.title("Dual Clumped Space")
        
        set_page_config(6)
        
        logo_manager = SidebarLogoManager()
        logo_manager.add_logo()
        
        if "PYTEST_CURRENT_TEST" not in os.environ:
            authenticator = Authenticator()
            if not authenticator.require_authentication():
                st.stop()


        
        # Add custom CSS
        self._add_custom_css()

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
                overflow: hidden;
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
        has_sample_db = os.path.exists("static/SampleDatabase.xlsx")
        
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
        
        df_filter = pd.read_excel("static/SampleDatabase.xlsx", engine="openpyxl")
        
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
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.sidebar.text_input(
                "Sample name contains (KEEP):",
                help="Set multiple keywords by separating them through semicolons ;",
                key="06_sample_contains",
                value="",
            )
        
        with col2:
            st.sidebar.text_input(
                "Sample name contains (DROP):",
                help="Set multiple keywords by separating them through semicolons ;",
                key="06_sample_not_contains",
                value="",
            )

    def _setup_plot_controls(self) -> None:
        """Set up plot control options in the sidebar."""
        # Plot level selection
        level_plot = st.sidebar.radio(
            "Choose plot level:", 
            ("Sample mean ±err", "Overview replicates"), 
            key="level_plot"
        )
        
        self.sss._05_level_plot = "rep" if "replicates" in level_plot else "mean"
        
        # Error determination for mean plots
        if "mean" in level_plot:
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
        
        # Additional options
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
        # Include filter
        if (self.sss.get("06_sample_contains") and 
            self.sss["06_sample_contains"].strip()):
            include_terms = self.sss["06_sample_contains"].split(";")
            df = df[df[column].apply(
                lambda x: any(term.strip() in x for term in include_terms)
            )]
        
        # Exclude filter
        if (self.sss.get("06_sample_not_contains") and 
            self.sss["06_sample_not_contains"].strip()):
            exclude_terms = self.sss["06_sample_not_contains"].split(";")
            df = df[~df[column].apply(
                lambda x: any(term.strip() in x for term in exclude_terms)
            )]
        
        return df

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
        )
        
        # Provide download link
        download_link = self._create_html_download_link(fig)
        st.markdown(download_link, unsafe_allow_html=True)

    def _create_dual_clumped_plot(self) -> go.Figure:
        """Create the main dual clumped space plot."""
        level_plot = self.sss.level_plot
        
        if "mean" in level_plot:
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
        #st.write(summary)
        fig = px.scatter(
            summary,
            x=self.sss.x_axis,
            y=self.sss.y_axis,
            error_x=self.sss.error_dualClumped.format(mz=self.sss.x_axis),
            error_y=self.sss.error_dualClumped.format(mz=self.sss.y_axis),
            text="Sample",
            color="Sample",
            hover_data=hover_data,
            symbol="Sample",
            symbol_sequence=self.symbols,
        ).update_traces(mode="lines+markers")
        
        # Update marker properties
        fig.update_traces(marker=dict(size=11))
        
        # Reduce error bar thickness
        for trace in fig.data:
            if hasattr(trace, 'error_y'):
                trace.error_y.thickness = 0.75
        
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
        
        fig = px.scatter(
            data_frame=df,
            x=self.sss.x_axis,
            y=self.sss.y_axis,
            color="Sample",
            symbol="Sample",
            symbol_sequence=self.symbols,
            hover_data=hover_data
        )
        
        return fig

    def _prepare_hover_data(self) -> List[str]:
        """Prepare hover data for mean plots."""
        
        hover_data = ["N", "d13C_VPDB", "d18O_CO2_VSMOW"]
        
        if not self.sss.params_last_run["process_D47"]:
            return hover_data
        
        for calib in self.sss["04_used_calibs"]:
            #st.write(calib)
            error_type = "2SE" if "2" in self.sss.error_dualClumped.format(mz="D47") else "1SE"
            hover_data.extend([
                f"T(min, {error_type}), {calib}",
                f"T(mean), {calib}",
                f"T(max, {error_type}), {calib}"
            ])
        #st.write(hover_data)
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
        temps_k = np.array([1 / (t + 273.15) for t in temps_c])
        
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
        calib_range = np.array([1 / (t + 273.15) for t in range(0, 1100, 1)])
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
            x = 1000 / (t_celsius+ 273.15)
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
            t_kelvin = t_celsius + 273.15
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
            xaxis=dict(title=x_title),
            yaxis=dict(title=y_title),
            hoverlabel=dict(font_size=20),
            legend=dict(font_size=15),
            legend_title=dict(font_size=25)
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
        calib_df["T_1K"] = 1 / (calib_df["T_C"] + 273.15)
        
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
        scaling_47, offset_47 = 1.0383389103254164, 0.18482382659656857
        
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
        return abs(args["D47 CDES"] - DualClumpedSpacePage.k47_temperature_function(1 / (temp + 273.15)))


if __name__ == "__main__":
    page = DualClumpedSpacePage()
    page.run()