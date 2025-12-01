#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import linregress, t
from typing import List, Tuple, Optional

from tools.page_config import set_page_config
from tools import sidebar_logo
from tools.commons import PlotParameters, modify_plot_text_sizes, PlotlyConfig
from tools.authenticator import Authenticator

# Configure page
set_page_config(5)

# Add logo
sidebar_logo.add_logo()




class DataFilter:
    """Handles filtering of DataFrame data based on sample name patterns."""
    
    def __init__(self):
        self.session_state = st.session_state
    
    def apply_filters(self, df, column: str = "Sample"):
        """
        Apply include/exclude filters to DataFrame based on sample names.
        
        Args:
            df: DataFrame to filter
            column: Column name to apply filters on
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Apply include filter
        include_keywords = self._get_filter_keywords("05_sample_contains")
        if include_keywords:
            filtered_df = self._include_samples(filtered_df, column, include_keywords)
        
        # Apply exclude filter
        exclude_keywords = self._get_filter_keywords("05_sample_not_contains")
        if exclude_keywords:
            filtered_df = self._exclude_samples(filtered_df, column, exclude_keywords)
        
        return filtered_df
    
    def _get_filter_keywords(self, key: str) -> List[str]:
        """Extract filter keywords from session state."""
        value = self.session_state.get(key, "")
        if value and isinstance(value, str):
            return [keyword.strip() for keyword in value.split(";") if keyword.strip()]
        return []
    
    def _include_samples(self, df, column: str, keywords: List[str]):
        """Include samples containing any of the specified keywords."""
        return df[df[column].apply(lambda x: any(keyword in x for keyword in keywords))]
    
    def _exclude_samples(self, df, column: str, keywords: List[str]):
        """Exclude samples containing any of the specified keywords."""
        return df[~df[column].apply(lambda x: any(keyword in x for keyword in keywords))]


class PlotGenerator:
    """Generates various plots for standardization results visualization."""
    
    def __init__(self):
        self.session_state = st.session_state
        self.symbols = PlotParameters.SYMBOLS
    
    def to_subscript(self, number_str):
        subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return number_str.translate(subscript_map)


    def create_delta_plot(self, df_data, standards_list, show_residuals: bool = False) -> go.Figure:
        """
        Create delta isotope plots (D47, D48, D49).
        
        Args:
            df_data: DataFrame containing the data
            standards_list: List of standards or samples to plot
            show_residuals: Whether to show residuals or absolute values
            
        Returns:
            Plotly figure
        """
        mz = str(self.session_state.result_mz)
        plotting_sessions = self._get_plotting_sessions(standards_list)
        df_filtered = df_data[df_data["Session"].isin(plotting_sessions)]
        
        # Setup plot
        y_label = f"∆∆{self.to_subscript(mz)}" if show_residuals else f"∆{self.to_subscript(mz)}"
        scale = self.session_state.params_last_run['scale']
        
        fig = go.Figure()
        fig.update_layout(
            xaxis_title=self.session_state.xval,
            yaxis_title=f"{y_label} ({scale}) [‰]",
            legend_title="Samples",
            hoverlabel=dict(font=dict(family="sans-serif", size=18)),
            height=600
        )
        
        # Track residuals for confidence intervals
        residuals_data = []
        degrees_freedom = 0
        has_data = False
        
        # Plot each standard/sample
        for idx, sample_name in enumerate(sorted(standards_list)):
            if sample_name not in df_filtered["Sample"].unique():
                continue
                
            has_data = True
            sample_df = df_filtered[df_filtered["Sample"] == sample_name]
            
            # Calculate y-values
            if show_residuals:
                y_values = (sample_df[f"D{mz}"] - sample_df[f"D{mz}"].mean()).round(4)
                if len(sample_df) > 1:
                    degrees_freedom += len(y_values)
                    residuals_data.extend(y_values)
            else:
                y_values = sample_df[f"D{mz}"].round(4)
            
            # Add scatter trace
            self._add_scatter_trace(fig, sample_df, y_values, sample_name, idx, mz)
            
            # Add reference lines for standards
            if not show_residuals and self._is_standard(sample_name, mz):
                self._add_reference_line(fig, sample_df, sample_name, mz)
        
        # Add confidence interval lines for residuals
        if show_residuals and residuals_data:
            self._add_confidence_intervals(fig, df_filtered, residuals_data, degrees_freedom)
        
        # Add session interval lines
        if self.session_state.xval == "Timetag" and has_data:
            self._add_session_intervals(fig)
        
        self._apply_plot_styling(fig)
        

        
        return fig
    
    def create_bulk_isotope_plot(self, df_data, standards_list, show_residuals: bool = False, 
                               over_time: bool = False, bulk_isotope: Optional[str] = None) -> go.Figure:
        """
        Create bulk isotope plots (δ18O vs δ13C or over time).
        
        Args:
            df_data: DataFrame containing the data
            standards_list: List of standards or samples to plot
            show_residuals: Whether to show residuals
            over_time: Whether to plot over time
            bulk_isotope: "13" or "18" for specific isotope when over_time=True
            
        Returns:
            Plotly figure
        """
        plotting_sessions = self._get_plotting_sessions(standards_list)
        df_filtered = df_data[df_data["Session"].isin(plotting_sessions)]
        
        # Determine axis labels
        prefix = "∆" if show_residuals else ""
        if over_time:
            x_label = self.session_state.xval
            if bulk_isotope == "13":
                y_label = f"{prefix}δ¹³C VPDB [‰]"
            else:
                y_label = f"{prefix}δ¹⁸O VSMOW [‰]"
        else:
            x_label = f"{prefix}δ¹⁸O VSMOW [‰]"
            y_label = f"{prefix}δ¹³C VPDB [‰]"
        
        fig = go.Figure()
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            hoverlabel=dict(font=dict(family="sans-serif", size=18)),
            height=600
        )
        
        # Track residual ranges for session lines
        residual_min, residual_max = 100, 0
        
        # Plot each sample
        for idx, sample_name in enumerate(sorted(standards_list)):
            sample_df = df_filtered[df_filtered["Sample"] == sample_name]
            if sample_df.empty:
                continue
            
            # Calculate x and y values
            x_vals, y_vals = self._calculate_bulk_coordinates(
                sample_df, show_residuals, over_time, bulk_isotope
            )
            
            # Update residual ranges
            if show_residuals and over_time:
                residual_min = min(residual_min, y_vals.min())
                residual_max = max(residual_max, y_vals.max())
            
            # Create marker configuration
            marker_config = self._create_marker_config(df_filtered, sample_df, idx)
            
            # Add scatter trace
            self._add_bulk_scatter_trace(fig, x_vals, y_vals, sample_name, 
                                       sample_df, marker_config)
        
        # Add session intervals for time plots
        if over_time and self.session_state.xval == "Timetag":
            y_range = (residual_min, residual_max) if show_residuals else None
            self._add_session_intervals_bulk(fig, df_filtered, y_range)
        
        # Apply color bar layout if enabled
        if self.session_state.get('05_cbar', False):
            self._apply_colorbar_layout(fig)
        
        self._apply_plot_styling(fig)
        return fig
    
    def create_heated_gas_line_plot(self, gas_name: str) -> Tuple[go.Figure, str]:
        """
        Create heated gas line plot with linear regression.
        
        Args:
            gas_name: Name of the gas standard (e.g., '25C', '1000C')
            
        Returns:
            Tuple of (Plotly figure, regression equation string)
        """
        mz = self.session_state.result_mz
        df_full = self.session_state.correction_output_full_dataset
        
        # Filter data
        df_filtered = df_full[
            (df_full["Session"].isin(self.session_state.plotting_sessions)) &
            (df_full["Sample"] == gas_name)
        ]
        
        if df_filtered.empty:
            return go.Figure(), "No data available"
        
        # Setup plot
        scale = self.session_state.params_last_run['scale']
        fig = go.Figure()
        fig.update_layout(
            xaxis_title=f"δ{mz} [‰]",
            yaxis_title=f"∆{self.to_subscript(str(mz))} {scale} [‰]",
            hoverlabel=dict(font_size=16),
            legend=dict(font_size=15)
        )
        
        # Add scatter plot
        color = "red" if gas_name == "1000C" else "blue"
        self._add_gas_scatter_trace(fig, df_filtered, gas_name, color, mz)
        
        # Perform linear regression
        x_data = df_filtered[self.session_state.delta]
        y_data = df_filtered[self.session_state.Delta]
        regression_result = linregress(x_data, y_data)
        
        # Add regression line
        self._add_regression_line(fig, x_data, regression_result, color)
        
        # Create equation string and annotation
        equation = self._format_regression_equation(regression_result)
        self._add_regression_annotation(fig, equation)
        
        return fig, equation
    
    def _get_plotting_sessions(self, standards_list):
        """Get list of sessions to plot."""
        if hasattr(standards_list, 'unique'):  # It's a pandas Series/Index
            return self.session_state.get('plotting_sessions', standards_list.unique())
        return self.session_state.get('plotting_sessions', [])
    
    def _add_scatter_trace(self, fig, sample_df, y_values, sample_name, idx, mz):
        """Add scatter trace to delta plot."""
        sample_df= sample_df.sort_values(by='Sample')
        hover_text = self._create_hover_text(sample_df, sample_name, mz)
        
        fig.add_trace(go.Scatter(
            x=sample_df[self.session_state.xval],
            y=y_values,
            mode="markers",
            name=sample_name,
            opacity=0.75,
            legendgroup=sample_name,
            marker=dict(size=12, line=dict(width=0.5), symbol=self.symbols[idx]),
            text=hover_text,
            textfont_size=15
        ))
    
    def _add_reference_line(self, fig, sample_df, sample_name, mz):
        """Add reference line for standard values."""
        standards = self.session_state["standards_nominal"]
        scale = self.session_state.params_last_run["scale"]
        
        if sample_name in standards[scale][str(mz)]:
            reference_value = standards[scale][str(mz)][sample_name]
            x_range = [sample_df[self.session_state.xval].min(), 
                      sample_df[self.session_state.xval].max()]
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=[reference_value, reference_value],
                mode="lines",
                legendgroup=sample_name,
                name=None,
                showlegend=False
            ))
            
    
    def _add_confidence_intervals(self, fig, df_filtered, residuals_data, degrees_freedom):
        """Add 95% confidence interval lines for residuals."""
        if degrees_freedom > 0:
            student_t = t.ppf(0.975, degrees_freedom)
            confidence_95 = np.array(residuals_data).std() * student_t
            
            x_range = [df_filtered[self.session_state.xval].min(),
                      df_filtered[self.session_state.xval].max()]
            
            for ci_value in [confidence_95, -confidence_95]:
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[ci_value, ci_value],
                    mode="lines",
                    line_color="grey",
                    opacity=0.75,
                    name=None,
                    showlegend=False,
                    text=f"95% CI = ±{confidence_95:.4f}"
                ))
    
    def _add_session_intervals(self, fig):
        """Add vertical lines for session intervals."""
        if not hasattr(self.session_state, 'session_intervals'):
            return
        
        shapes = []
        for session_name, start_time, end_time in self.session_state.session_intervals:
            if session_name not in self.session_state.plotting_sessions:
                continue
            
            # Add start and end lines
            for time_point in [start_time, end_time]:
                shapes.append({
                    "type": "line",
                    "xref": "x",
                    "yref": "paper",
                    "y0": 0,
                    "y1": 1,
                    "x0": time_point,
                    "x1": time_point,
                    "line": {"color": "rgb(55, 128, 191)", "width": 1}
                })
        
        fig.update_layout(shapes=shapes)
    
    def _calculate_bulk_coordinates(self, sample_df, show_residuals, over_time, bulk_isotope):
        """Calculate x and y coordinates for bulk isotope plots."""
        if show_residuals:
            if over_time:
                x_vals = sample_df[self.session_state.xval]
                if bulk_isotope == "13":
                    y_vals = sample_df["d13C_VPDB"] - sample_df["d13C_VPDB"].mean()
                else:
                    y_vals = sample_df["d18O_VSMOW"] - sample_df["d18O_VSMOW"].mean()
            else:
                x_vals = sample_df["d18O_VSMOW"] - sample_df["d18O_VSMOW"].mean()
                y_vals = sample_df["d13C_VPDB"] - sample_df["d13C_VPDB"].mean()
        else:
            x_vals = sample_df["d18O_VSMOW"]
            y_vals = sample_df["d13C_VPDB"]
        
        return x_vals, y_vals
    
    def _create_marker_config(self, df_filtered, sample_df, idx):
        """Create marker configuration for bulk isotope plots."""
        base_config = dict(size=12, line=dict(width=0.5), symbol=self.symbols[idx])
        
        if self.session_state.get('05_cbar', False):
            z_axis = self.session_state.get("05_dd_zaxis", "D47")
            base_config.update({
                "showscale": True,
                "cmax": df_filtered[z_axis].max(),
                "cmin": df_filtered[z_axis].min(),
                "colorbar": {"title": z_axis},
                "color": sample_df[z_axis].values,
                "colorscale": [
                    [0, "rgba(204, 121, 167, 1)"],    # Reddish Purple
                    [0.28, "rgba(213, 94, 0, 1)"],    # Vermilion
                    [0.6, "rgba(0, 158, 115, 1)"],    # Green
                    [1, "rgba(86, 180, 233, 1)"]      # Sky Blue
                ]
            })
        
        return base_config
    
    def _add_bulk_scatter_trace(self, fig, x_vals, y_vals, sample_name, sample_df, marker_config):
        """Add scatter trace for bulk isotope plots."""
        mz = str(self.session_state.result_mz)
        hover_text = self._create_hover_text(sample_df, sample_name, mz)
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            opacity=0.75,
            name=sample_name,
            legendgroup=sample_name,
            marker=marker_config,
            text=hover_text,
            hoverlabel=dict(font=dict(family="sans-serif", size=18))
        ))
    
    def _add_session_intervals_bulk(self, fig, df_filtered, y_range):
        """Add session interval lines for bulk isotope plots."""
        if not hasattr(self.session_state, 'session_intervals'):
            return
        
        # Determine y-range
        if y_range:
            y_min, y_max = y_range
        else:
            mz = str(self.session_state.result_mz)
            y_min = df_filtered[f"D{mz}"].min()
            y_max = df_filtered[f"D{mz}"].max()
        
        for session_name, start_time, end_time in self.session_state.session_intervals:
            if session_name not in self.session_state.plotting_sessions:
                continue
            
            # Add start and end lines
            for time_point, label in [(start_time, "Begin"), (end_time, "End")]:
                fig.add_trace(go.Scatter(
                    x=[time_point, time_point],
                    y=[y_min, y_max],
                    mode="lines",
                    line_color="grey",
                    opacity=0.75,
                    name=None,
                    showlegend=False,
                    text=f"Session: {session_name}<br>{label}: {time_point}"
                ))
    
    def _apply_colorbar_layout(self, fig):
        """Apply layout adjustments for colorbar."""
        fig.update_layout(
            legend=dict(
                font=dict(size=14),
                orientation="v",
                yanchor="auto",
                y=1,
                xanchor="right",
                x=-0.1
            )
        )
    
    def _add_gas_scatter_trace(self, fig, df_filtered, gas_name, color, mz):
        """Add scatter trace for heated gas line plot."""
        hover_text = self._create_hover_text(df_filtered, gas_name, mz)
        
        fig.add_trace(go.Scatter(
            x=df_filtered[self.session_state.delta],
            y=df_filtered[self.session_state.Delta],
            mode="markers",
            opacity=0.75,
            name=gas_name,
            marker=dict(size=12, line=dict(width=0.5), color=color),
            text=hover_text
        ))
    
    def _add_regression_line(self, fig, x_data, regression_result, color):
        """Add regression line to heated gas plot."""
        # Calculate confidence intervals
        degrees_freedom = len(x_data) - 2
        t_value = abs(t.ppf(0.025, degrees_freedom))  # 95% confidence
        
        y_pred = regression_result.intercept + regression_result.slope * x_data
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_pred,
            mode="lines",
            line_color=color,
            opacity=0.75,
            showlegend=False,
            text=(f"Slope (95%): {regression_result.slope:.4f} "
                  f"±{t_value * regression_result.stderr:.4f}\n"
                  f"Intercept (95%): {regression_result.intercept:.4f} "
                  f"±{t_value * regression_result.intercept_stderr:.4f}")
        ))
    
    def _format_regression_equation(self, regression_result):
        """Format regression equation with scientific notation."""
        def to_scientific_html(num):
            """Convert number to scientific notation HTML."""
            parts = f"{num:.2e}".split('e')
            base = parts[0]
            exponent = int(parts[1])
            return f"{base} × 10<sup>{exponent}</sup>"
        
        slope_str = to_scientific_html(regression_result.slope)
        slope_err_str = to_scientific_html(regression_result.stderr)
        intercept_str = to_scientific_html(regression_result.intercept)
        intercept_err_str = to_scientific_html(regression_result.intercept_stderr)
        
        sign = "+" if regression_result.intercept >= 0 else ""
        
        return (f"f(x) = ({slope_str} ± {slope_err_str})x "
                f"{sign}{intercept_str} ± {intercept_err_str}")
    
    def _add_regression_annotation(self, fig, equation):
        """Add regression equation annotation to plot."""
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.06,
            showarrow=False,
            text=f"<b>{equation}</b>",
            font=dict(size=18, color="grey")
        )
    
    def _create_hover_text(self, sample_df, sample_name, mz):
        """Create hover text for data points."""
        xval = self.session_state.xval
        scale = self.session_state.params_last_run['scale']
        
        hover_texts = []
        for _, row in sample_df.iterrows():
            x_val = row[xval]
            x_display = f"{x_val:.3f} ‰" if isinstance(x_val, float) else str(x_val)
            
            text = (f"{sample_name}<br>"
                   f"UID={row['UID']}<br>"
                   f"{xval}={x_display}<br>"
                   f"∆{self.to_subscript(str(mz))} {scale}={row[f'D{mz}']:.3f} ‰<br>"
                   f"δ¹⁸O_VSMOW(CO₂)={row['d18O_VSMOW']:.2f} ‰<br>"
                   f"δ¹³C_VPDB={row['d13C_VPDB']:.2f} ‰")
            hover_texts.append(text)
        
        return hover_texts
    
    def _is_standard(self, sample_name, mz):
        """Check if sample is a standard."""
        standards = self.session_state["standards_nominal"]
        scale = self.session_state.params_last_run["scale"]
        return sample_name in standards[scale][str(mz)]
    
    def _apply_plot_styling(self, fig):
        """Apply consistent styling to plots."""
        fig.update_traces(textfont_size=15)
        fig.update_xaxes(showline=True, linewidth=2, linecolor="white", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="white", mirror=True)


class StandardizationResultsPage:
    """Main class for the Standardization Results page."""
    
    def __init__(self):
        """Initialize the page with required components."""
        self.session_state = st.session_state
        self.data_filter = DataFilter()
        self.plot_generator = PlotGenerator()
        
        self._setup_page()
        self._initialize_session_state()
    
    def _setup_page(self):
        """Set up page configuration and sidebar controls."""
        # Page title
        st.title("Stable and clumped isotope results")
        
        if "PYTEST_CURRENT_TEST" not in os.environ:
            authenticator = Authenticator()
            if not authenticator.require_authentication():
                st.stop()

        # Sidebar filters
        #self._render_sidebar_filters()
    
    def _initialize_session_state(self):
        """Initialize session state variables with defaults."""
        defaults = {
            '05_sample_contains': '',
            '05_sample_not_contains': '',
            '05_cbar': False,
            '05_dd_zaxis': 'D47'
        }
        
        for key, default_value in defaults.items():
            if key not in self.session_state:
                self.session_state[key] = default_value
    
    def _render_sidebar_filters(self):
        """Render sidebar filter controls."""
        st.sidebar.text_input(
            "Sample name contains (KEEP):",
            help="Set multiple keywords by separating them through semicolons ;",
            key="05_sample_contains",
            value=""
        )
        
        st.sidebar.text_input(
            "Sample name contains (DROP):",
            help="Set multiple keywords by separating them through semicolons ;",
            key="05_sample_not_contains",
            value=""
        )
        
        st.sidebar.toggle('Scaled marker colors', key='05_cbar')
    
    def run(self):
        """Main function to run the standardization results page."""
        if not self._validate_input_data():
            self._show_data_requirements()
            return
        
        if not self._validate_processed_data():
            self._show_processing_requirements()
            return
        
        self._render_sidebar_filters()
        # Apply data filters
        self._apply_data_filters()
        
        # Setup colorbar controls if enabled
        self._setup_colorbar_controls()
        
        # Render main controls and results
        self._render_main_controls()
        self._render_results_tabs()
    
    def _validate_input_data(self) -> bool:
        """Validate that input data is available."""
        return ("input_rep" in self.session_state and 
                len(self.session_state.get("input_rep", [])) > 0)
    
    def _validate_processed_data(self) -> bool:
        """Validate that processed data is available."""
        return "correction_output_summary" in self.session_state
    
    def _show_data_requirements(self):
        """Show data upload requirements."""
        st.markdown("Please upload a dataset and process it, before you can explore results:")
        st.markdown("- Either, upload raw intensity data to perform a baseline correction "
                   "(:violet[Upload m/z44-m/z49 intensities] tab).")
        st.markdown("- Or, directly upload δ⁴⁵-δ⁴⁹ replicate data "
                   "(:violet[Upload δ⁴⁵-δ⁴⁹ replicates] tab).")
        st.page_link("pages/01_Data_IO.py", label=r"$\rightarrow  \textit{Data-IO}$  page")
        st.markdown(" ")
        st.markdown("Finally, process the dataset")
        st.page_link("pages/04_Processing.py", label=r"$\rightarrow  \textit{Processing}$  page")
    
    def _show_processing_requirements(self):
        """Show processing requirements."""
        st.markdown("Please process the dataset, before you can explore results "
                   "(:violet[Processing] page).")
        st.stop()
    
    def _apply_data_filters(self):
        """Apply data filters to datasets."""
        if "correction_output_summary" in self.session_state:
            self.df_rep = self.data_filter.apply_filters(
                self.session_state.correction_output_full_dataset, "Sample"
            )
            self.df_summary = self.data_filter.apply_filters(
                self.session_state.correction_output_summary, "Sample"
            )
    
    def _setup_colorbar_controls(self):
        """Setup colorbar controls if enabled."""
        if self.session_state.get('05_cbar', False):
            numeric_columns = self.df_rep.select_dtypes([np.number]).columns
            default_index = 0
            
            # Try to find D47 as default
            if "D47" in numeric_columns:
                default_index = list(numeric_columns).index("D47")
            
            st.sidebar.selectbox(
                "Metric for marker color in ¹⁸O/¹³C plots",
                numeric_columns,
                key="05_dd_zaxis",
                index=default_index
            )
    
    def _render_main_controls(self):
        """Render main control widgets."""
        # Get available isotope options
        isotope_options = self._get_available_isotopes()
        
        # Isotope selection
        selected_isotope = st.sidebar.radio(
            "Show results for:", isotope_options, horizontal=True
        )
        
        # X-axis selection
        self.session_state.xval = st.sidebar.radio(
            "Choose x-axis:", ("Timetag", "UID"), horizontal=True
        )
        
        # Setup session intervals for time plots
        if self.session_state.xval == "Timetag":
            self._setup_session_intervals()
        
        # Set isotope-specific variables
        self._set_isotope_variables(selected_isotope)
        
        # Session selection
        self._setup_session_selection()
        
        # Sample type selection
        self.sample_display_type = st.radio(
            "What to display:",
            ("All together", "Samples only", "Standards only"),
            horizontal=True
        )
        
        self.sample_type_code = self._get_sample_type_code(self.sample_display_type)
    
    def _get_available_isotopes(self) -> List[str]:
        """Get list of available isotopes based on processing parameters."""
        options = []
        params = self.session_state.params_last_run
        
        if params.get("process_D47", False):
            options.append("∆47")
        if params.get("process_D48", False):
            options.append("∆48")
        if params.get("process_D49", False):
            options.append("∆49")
        
        return options
    
    def _setup_session_intervals(self):
        """Setup session intervals for time-based plots."""
        session_groups = self.session_state.correction_output_full_dataset.groupby("Session")
        self.session_state.session_intervals = [
            (name, group["Timetag"].min(), group["Timetag"].max())
            for name, group in session_groups
        ]
    
    def _set_isotope_variables(self, selected_isotope: str):
        """Set session state variables based on selected isotope."""
        isotope_map = {
            "∆47": (47, "D47", "d47"),
            "∆48": (48, "D48", "d48"),
            "∆49": (49, "D49", "d49")
        }
        
        if selected_isotope in isotope_map:
            mz, delta_var, delta_lower = isotope_map[selected_isotope]
            self.session_state.result_mz = mz
            self.session_state.Delta = delta_var
            self.session_state.delta = delta_lower
    
    def _setup_session_selection(self):
        """Setup session selection multiselect."""
        standards_data = self.session_state.standards
        all_sessions = sorted(standards_data["Session"].unique())
        
        self.session_state.plotting_sessions = st.sidebar.multiselect(
            "Select sessions:", all_sessions, all_sessions
        )
    
    def _get_sample_type_code(self, display_type: str) -> str:
        """Convert display type to internal code."""
        if "tandard" in display_type:
            return "std"
        elif "ample" in display_type:
            return "sample"
        else:
            return "all"
    
    def _render_results_tabs(self):
        """Render the main results tabs."""
        # Check if heated gas standards are available
        standards_data = self.session_state.standards
        has_heated_gas = "25C" in standards_data["Sample"].values
        
        # Create tabs
        if has_heated_gas:
            tabs = st.tabs([
                f"∆{PlotGenerator.to_subscript('',str(self.session_state.result_mz))}/{self.session_state.xval}",
                f"∆∆{PlotGenerator.to_subscript('',str(self.session_state.result_mz))} residuals/{self.session_state.xval}",
                "δ¹⁸O/δ¹³C",
                "∆δ¹⁸O/∆δ¹³C",
                f"∆¹⁸O residuals/{self.session_state.xval}",
                f"∆¹³C residuals/{self.session_state.xval}",
                "Heated gas line(s)"
            ])
        else:
            tabs = st.tabs([
                f"∆{PlotGenerator.to_subscript('',str(self.session_state.result_mz))}/{self.session_state.xval}",
                f"∆∆{PlotGenerator.to_subscript('',str(self.session_state.result_mz))} residuals/{self.session_state.xval}",
                "δ¹⁸O/δ¹³C",
                "∆δ¹⁸O/∆δ¹³C",
                f"∆¹⁸O residuals/{self.session_state.xval}",
                f"∆¹³C residuals/{self.session_state.xval}"
            ])
        
        # Get standards list for plotting
        standards_list = self._get_standards_list()
        
        # Render each tab
        with tabs[0]:  # Delta values
            fig = self.plot_generator.create_delta_plot(
                self.df_rep, standards_list, show_residuals=False
            )
            st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)
        
        with tabs[1]:  # Delta residuals
            fig = self.plot_generator.create_delta_plot(
                self.df_rep, standards_list, show_residuals=True
            )
            st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)
        
        with tabs[2]:  # Bulk isotopes
            fig = self.plot_generator.create_bulk_isotope_plot(
                self.df_rep, standards_list, show_residuals=False
            )
            st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)
        
        with tabs[3]:  # Bulk isotope residuals
            fig = self.plot_generator.create_bulk_isotope_plot(
                self.df_rep, standards_list, show_residuals=True
            )
            st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)
        
        with tabs[4]:  # δ18O residuals over time
            fig = self.plot_generator.create_bulk_isotope_plot(
                self.df_rep, standards_list, show_residuals=True, over_time=True, bulk_isotope="18"
            )
            st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)
        
        with tabs[5]:  # δ13C residuals over time
            fig = self.plot_generator.create_bulk_isotope_plot(
                self.df_rep, standards_list, show_residuals=True, over_time=True, bulk_isotope="13"
            )
            st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)
        
        # Heated gas lines tab (if available)
        if has_heated_gas:
            with tabs[6]:
                for gas_name in ["25C", "1000C"]:
                    fig, _ = self.plot_generator.create_heated_gas_line_plot(gas_name)
                    st.plotly_chart(modify_plot_text_sizes(fig))
    
    def _get_standards_list(self):
        """Get list of standards or samples based on selection."""
        mz = str(self.session_state.result_mz)
        
        if self.sample_type_code == "std":
            return list(self.session_state["standards_nominal"]
                       [self.session_state.params_last_run["scale"]][mz].keys())
        elif self.sample_type_code == "sample":
            all_samples = self.df_rep["Sample"].unique()
            standards = self.session_state["standards_nominal"][
                self.session_state.params_last_run["scale"]][mz]
            return [sample for sample in all_samples if sample not in standards]
        else:  # "all"
            return self.df_rep["Sample"].unique()


# Main execution
if __name__ == "__main__":
    page = StandardizationResultsPage()
    page.run()