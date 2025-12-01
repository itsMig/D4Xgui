#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ogls import Polynomial
from scipy import stats

from tools.page_config import PageConfigManager
from tools.sidebar_logo import SidebarLogoManager
from tools.authenticator import Authenticator
from tools.commons import PLOT_PARAMS, modify_text_label_sizes, PlotlyConfig


class DataType(Enum):
    """Enumeration for different data types available for plotting."""
    SAMPLE_MEAN = "Sample mean"
    ALL_REPLICATES = "All replicates"
    RAW_DATA = "Raw data"


class FitErrorType(Enum):
    """Enumeration for different error handling methods in fitting."""
    HARMONIC = "harmonic"
    COLUMN = "column"
    NONE = "None"


@dataclass
class PlotConfig:
    """Configuration for plot settings."""
    x_column: str
    y_column: str
    text_label: Optional[str] = None
    fit_enabled: bool = False
    fit_order: int = 1
    fit_error_type: FitErrorType = FitErrorType.NONE
    error_x: float = 0.001
    error_y: float = 0.001
    error_column_x: Optional[str] = None
    error_column_y: Optional[str] = None


@dataclass
class FilterConfig:
    """Configuration for data filtering."""
    sample_contains: str = ""
    sample_not_contains: str = ""


class DataFilter:
    """Handles data filtering operations."""
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, column: str, filter_config: FilterConfig) -> pd.DataFrame:
        """Apply include and exclude filters to a DataFrame.
        
        Args:
            df: DataFrame to filter
            column: Column name to apply filters on
            filter_config: Filter configuration
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Apply include filters
        if filter_config.sample_contains:
            include_terms = [term.strip() for term in filter_config.sample_contains.split(";")]
            filtered_df = filtered_df[
                filtered_df[column].apply(
                    lambda x: any(term in str(x) for term in include_terms)
                )
            ]
        
        # Apply exclude filters
        if filter_config.sample_not_contains:
            exclude_terms = [term.strip() for term in filter_config.sample_not_contains.split(";")]
            filtered_df = filtered_df[
                ~filtered_df[column].apply(
                    lambda x: any(term in str(x) for term in exclude_terms)
                )
            ]
        
        return filtered_df


class DataFitter:
    """Handles data fitting operations."""
    
    @staticmethod
    def fit_polynomial_simple(x_data: np.ndarray, y_data: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """Perform simple polynomial fitting without error consideration.
        
        Args:
            x_data: X values
            y_data: Y values
            order: Polynomial order
            
        Returns:
            Tuple of (x_fit, y_fit, equation_string)
        """
        coeffs = np.polyfit(x_data, y_data, order)
        x_min, x_max = x_data.min(), x_data.max()
        x_fit = np.linspace(x_min, x_max, 100)
        y_fit = np.polyval(coeffs, x_fit)
        
        equation = "f(x) = "
        for i, coeff in enumerate(coeffs):
            power = len(coeffs) - i - 1
            if power > 0:
                equation += f"{coeff:.2e}x<sup>{power}</sup> + "
            else:
                equation += f"{coeff:.2e}"
        
        return x_fit, y_fit, equation
    
    @staticmethod
    def fit_polynomial_with_errors(
        x_data: np.ndarray, y_data: np.ndarray, 
        sx: np.ndarray, sy: np.ndarray, order: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, object]:
        """Perform polynomial fitting with error consideration using OGLS.
        
        Args:
            x_data: X values
            y_data: Y values
            sx: X errors
            sy: Y errors
            order: Polynomial order
            
        Returns:
            Tuple of (x_fit, y_fit, confidence_interval, equation_string, ogls_object)
        """
        ogls = Polynomial(
            X=x_data,
            Y=y_data,
            sY=sy,
            sX=sx,
            degrees=list(range(order + 1))
        )
        ogls.regress(verbose=True)
        
        x_min, x_max = x_data.min(), x_data.max()
        x_range = x_max - x_min
        x_fit = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100)
        y_fit = ogls.bff(x_fit)
        
        # Calculate confidence interval
        pcov = ogls.bfp_CM
        y_fit_se = np.sqrt(
            np.sum(
                (np.array([x_fit ** i for i in range(len(ogls.bfp))]).T @ pcov) *
                np.array([x_fit ** i for i in range(len(ogls.bfp))]).T,
                axis=1
            )
        )
        
        confidence = 0.95
        n = len(x_data)
        dof = n - len(ogls.bfp)
        t_value = stats.t.ppf((1 + confidence) / 2, dof)
        ci = t_value * y_fit_se
        
        # Format equation
        equation = "f(x) = "
        for i in range(len(ogls.bfp) - 1, -1, -1):
            coeff = ogls.bfp[f'a{i}']
            err = ogls.bfp_se[f'a{i}']
            
            coeff_mag = int(np.floor(np.log10(abs(coeff))))
            err_mag = int(np.floor(np.log10(abs(err))))
            
            coeff_formatted = coeff / 10 ** coeff_mag
            err_formatted = err / 10 ** coeff_mag
            
            if i > 0:
                equation += f"{coeff_formatted:.2f}(±{err_formatted:.2f})×10<sup>{coeff_mag}</sup>x<sup>{i}</sup> + "
            else:
                equation += f"{coeff_formatted:.2f}(±{err_formatted:.2f})×10<sup>{coeff_mag}</sup>"
        
        return x_fit, y_fit, ci, equation, ogls


class PlotGenerator:
    """Handles plot generation and customization."""
    
    def __init__(self):
        self.symbols = PLOT_PARAMS.SYMBOLS
    
    def create_scatter_plot(
        self, df: pd.DataFrame, plot_config: PlotConfig,
        error_x: Optional[Union[List, np.ndarray]] = None,
        error_y: Optional[Union[List, np.ndarray]] = None
    ) -> go.Figure:
        """Create a scatter plot with the given configuration.
        
        Args:
            df: DataFrame to plot
            plot_config: Plot configuration
            error_x: X-axis error bars
            error_y: Y-axis error bars
            
        Returns:
            Plotly figure object
        """
        fig = px.scatter(
            df,
            x=plot_config.x_column,
            y=plot_config.y_column,
            error_x=error_x,
            error_y=error_y,
            opacity=0.75,
            text=plot_config.text_label,
            color="Sample",
            symbol="Sample",
            symbol_sequence=self.symbols,
        )
        
        return self._customize_plot(fig)
    
    def add_fit_line(
        self, fig: go.Figure, x_fit: np.ndarray, y_fit: np.ndarray,
        equation: str, confidence_interval: Optional[np.ndarray] = None
    ) -> go.Figure:
        """Add a fit line to the plot.
        
        Args:
            fig: Plotly figure
            x_fit: X values for fit line
            y_fit: Y values for fit line
            equation: Equation string for legend
            confidence_interval: Optional confidence interval
            
        Returns:
            Updated figure
        """
        # Add fit line
        fig.add_trace(
            go.Scatter(
                x=x_fit, y=y_fit, mode='lines',
                name=equation,
                line=dict(color='red')
            )
        )
        
        # Add confidence interval if provided
        if confidence_interval is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_fit, y=y_fit + confidence_interval,
                    mode='lines', line=dict(width=0), showlegend=False
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_fit, y=y_fit - confidence_interval,
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(255,0,0,0.2)',
                    name='95% CI', showlegend=False
                )
            )
        
        return fig
    
    def _customize_plot(self, fig: go.Figure) -> go.Figure:
        """Apply custom styling to the plot.
        
        Args:
            fig: Plotly figure
            
        Returns:
            Customized figure
        """
        fig.update_layout(
            hoverlabel=dict(font_size=20),
            legend=dict(font_size=15),
            margin=dict(l=10, r=10, t=35, b=10),
        )
        fig.update_traces(marker=dict(size=12))
        fig = modify_text_label_sizes(fig)
        
        return fig


class DiscoverResultsPage:
    """Main class for the Discover Results page."""
    
    def __init__(self):
        """Initialize the DiscoverResultsPage."""
        self.sss = st.session_state
        self.data_filter = DataFilter()
        self.data_fitter = DataFitter()
        self.plot_generator = PlotGenerator()
        self._setup_page()
    
    def _setup_page(self) -> None:
        """Set up page configuration, logo, and authentication."""
        st.title("Discover Results")
        
        page_config_manager = PageConfigManager()
        page_config_manager.configure_page(page_number=7)
        
        logo_manager = SidebarLogoManager()
        logo_manager.add_logo()
        
        if "PYTEST_CURRENT_TEST" not in os.environ:
            authenticator = Authenticator()
            if not authenticator.require_authentication():
                st.stop()
    
    def run(self) -> None:
        """Run the main application page."""
        if not self._check_data_availability():
            self._show_data_requirement_message()
            return
        
        self._render_main_interface()
    
    def _check_data_availability(self) -> bool:
        """Check if required data is available in session state.
        
        Returns:
            True if data is available, False otherwise
        """
        return ("correction_output_summary" in self.sss or 
                "input_intensities" in self.sss)
    
    def _show_data_requirement_message(self) -> None:
        """Display message when no data is available."""
        st.markdown("Please upload (and process) a dataset in order to screen results.")
        st.page_link("pages/01_Data_IO.py", label=r"$\rightarrow  \textit{Data-IO}$  page")
        st.page_link("pages/04_Processing.py", label=r"$\rightarrow  \textit{Processing}$  page")
        st.stop()
    
    def _get_available_data_types(self) -> List[str]:
        """Get list of available data types for plotting.
        
        Returns:
            List of available data type strings
        """
        options = []
        if "correction_output_summary" in self.sss:
            options.extend([DataType.SAMPLE_MEAN.value, DataType.ALL_REPLICATES.value])
        if "input_intensities" in self.sss:
            options.append(DataType.RAW_DATA.value)
        return options
    
    def _render_main_interface(self) -> None:
        """Render the main user interface."""
        options_plot = self._get_available_data_types()
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_data_selection(options_plot)
            self._render_annotation_selection()
        
        self._render_filter_controls()
        
        # Get filtered data and column options
        df, columns = self._get_filtered_data_and_columns()
        if df is None:
            return
        
        with col2:
            self._render_axis_selection(columns)
        
        if self._can_create_plot():
            self._render_plot_section(df)
    
    def _render_data_selection(self, options: List[str]) -> None:
        """Render data type selection widget.
        
        Args:
            options: List of available data type options
        """
        self.sss["07_df_to_show_"] = st.selectbox(
            label="Please choose if you want to plot every replicate, or sample mean ±2SE!",
            options=options,
        )
    
    def _render_annotation_selection(self) -> None:
        """Render annotation selection widget."""
        if "replicates" in self.sss.get("07_df_to_show_", ""):
            annotation_options = [None, "Sample", "Timetag"]
        else:
            annotation_options = [None, "Sample"]
        
        self.sss["07_text_label"] = st.selectbox(
            label="Please select in-plot annotations!",
            options=annotation_options
        )
    
    def _render_filter_controls(self) -> None:
        """Render filter control widgets in the sidebar."""
        st.sidebar.text_input(
            "Sample name contains (KEEP):",
            help="Set multiple keywords by separating them through semicolons ;",
            key="07_sample_contains",
            value="",
        )
        
        st.sidebar.text_input(
            "Sample name contains (DROP):",
            help="Set multiple keywords by separating them through semicolons ;",
            key="07_sample_not_contains",
            value="",
        )
    
    def _get_filtered_data_and_columns(self) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        """Get filtered data and available columns.
        
        Returns:
            Tuple of (filtered_dataframe, column_list) or (None, None) if no data
        """
        if "07_df_to_show_" not in self.sss:
            return None, None
        
        # Get the appropriate dataset
        df = self._select_dataset()
        
        # Apply filters
        filter_config = FilterConfig(
            sample_contains=self.sss.get("07_sample_contains", ""),
            sample_not_contains=self.sss.get("07_sample_not_contains", "")
        )
        df = self.data_filter.apply_filters(df, "Sample", filter_config)
        
        if len(df) == 0:
            st.info('Please apply filters to see data. No data to display.')
            st.stop()
        
        return df, list(df.columns)
    
    def _select_dataset(self) -> pd.DataFrame:
        """Select the appropriate dataset based on user choice.
        
        Returns:
            Selected DataFrame
        """
        data_type = self.sss["07_df_to_show_"]
        
        if "replicate" in data_type:
            return self.sss.correction_output_full_dataset
        elif "Sample" in data_type:
            return self.sss.correction_output_summary
        else:  # Raw data
            return self.sss.input_intensities
    
    def _get_default_axis_columns(self) -> Tuple[str, str]:
        """Get default x and y axis columns based on data type.
        
        Returns:
            Tuple of (x_column, y_column)
        """
        data_type = self.sss.get("07_df_to_show_", "")
        
        if "replicate" in data_type:
            return "Timetag", "Sample"
        elif "Sample" in data_type:
            return "Sample", "N"
        else:  # Raw data
            return "Timetag", "Sample"
    
    def _render_axis_selection(self, columns: List[str]) -> None:
        """Render axis selection widgets.
        
        Args:
            columns: List of available column names
        """
        x_default, y_default = self._get_default_axis_columns()
        
        try:
            x_pre_idx = columns.index(x_default)
            y_pre_idx = columns.index(y_default)
        except ValueError:
            x_pre_idx = y_pre_idx = 0
        
        self.sss["07_col_X"] = st.selectbox(
            label="Please choose x-axis for plot!",
            options=columns,
            index=x_pre_idx
        )
        self.sss["07_col_Y"] = st.selectbox(
            label="Please choose y-axis for plot!",
            options=columns,
            index=y_pre_idx
        )
    
    def _can_create_plot(self) -> bool:
        """Check if plot can be created.
        
        Returns:
            True if both X and Y columns are selected
        """
        return ("07_col_X" in self.sss and "07_col_Y" in self.sss)
    
    def _render_plot_section(self, df: pd.DataFrame) -> None:
        """Render the plotting section with fit options and the actual plot.
        
        Args:
            df: DataFrame to plot
        """
        plot_config = self._get_plot_config()
        
        # Render fit controls
        st.checkbox('Fit data', value=False, key='07_fit')
        
        error_x = error_y = None
        ogls_object = None
        
        if self.sss.get('07_fit', False):
            plot_config, error_x, error_y, ogls_object = self._render_fit_controls(df, plot_config)
        
        # Create and display plot
        fig = self._create_plot(df, plot_config, error_x, error_y, ogls_object)
        st.plotly_chart(fig, config=PlotlyConfig.CONFIG)
    
    def _get_plot_config(self) -> PlotConfig:
        """Get the current plot configuration.
        
        Returns:
            PlotConfig object
        """
        return PlotConfig(
            x_column=self.sss["07_col_X"],
            y_column=self.sss["07_col_Y"],
            text_label=self.sss.get("07_text_label"),
            fit_enabled=self.sss.get('07_fit', False)
        )
    
    def _render_fit_controls(
        self, df: pd.DataFrame, plot_config: PlotConfig
    ) -> Tuple[PlotConfig, Optional[np.ndarray], Optional[np.ndarray], Optional[object]]:
        """Render fit control widgets and calculate fit parameters.
        
        Args:
            df: DataFrame being plotted
            plot_config: Current plot configuration
            
        Returns:
            Tuple of (updated_plot_config, error_x, error_y, ogls_object)
        """
        fitcol1, fitcol2 = st.columns(2)
        
        with fitcol1:
            plot_config.fit_order = st.selectbox('Order', options=[1, 2, 3, 4, 5], key='07_fit_order')
            error_type_str = st.selectbox(
                'Include error', 
                options=[e.value for e in FitErrorType], 
                key='07_fit_error'
            )
            plot_config.fit_error_type = FitErrorType(error_type_str)
        
        error_x, error_y = self._handle_error_configuration(df, plot_config, fitcol2)
        ogls_object = None
        
        # Perform fitting
        if plot_config.fit_error_type == FitErrorType.NONE:
            self._perform_simple_fit(df, plot_config)
        else:
            ogls_object = self._perform_error_weighted_fit(df, plot_config, error_x, error_y)
        
        return plot_config, error_x, error_y, ogls_object
    
    def _handle_error_configuration(
        self, df: pd.DataFrame, plot_config: PlotConfig, column_widget
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Handle error configuration based on the selected error type.
        
        Args:
            df: DataFrame being plotted
            plot_config: Plot configuration
            column_widget: Streamlit column for widgets
            
        Returns:
            Tuple of (error_x, error_y)
        """
        with column_widget:
            if plot_config.fit_error_type == FitErrorType.HARMONIC:
                return self._configure_harmonic_errors(df, plot_config)
            elif plot_config.fit_error_type == FitErrorType.COLUMN:
                return self._configure_column_errors(df, plot_config)
            else:  # None
                data_length = len(df[plot_config.x_column].values)
                return np.full(data_length, np.nan), np.full(data_length, np.nan)
    
    def _configure_harmonic_errors(
        self, df: pd.DataFrame, plot_config: PlotConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Configure harmonic (constant) errors.
        
        Args:
            df: DataFrame being plotted
            plot_config: Plot configuration
            
        Returns:
            Tuple of (error_x, error_y)
        """
        plot_config.error_x = st.number_input(
            'Error on x-axis',
            min_value=0.001,
            step=0.001,
            format='%.3f',
            key='07_fit_error_x'
        )
        plot_config.error_y = st.number_input(
            'Error on y-axis',
            min_value=0.001,
            format='%.3f',
            step=0.001,
            key='07_fit_error_y'
        )
        
        data_length = len(df[plot_config.x_column].values)
        return (np.full(data_length, plot_config.error_x),
                np.full(data_length, plot_config.error_y))
    
    def _configure_column_errors(
        self, df: pd.DataFrame, plot_config: PlotConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Configure column-based errors.
        
        Args:
            df: DataFrame being plotted
            plot_config: Plot configuration
            
        Returns:
            Tuple of (error_x, error_y)
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        plot_config.error_column_x = st.selectbox(
            'Error column (x-axis)', 
            options=numeric_columns, 
            key='07_fit_error_column_x'
        )
        plot_config.error_column_y = st.selectbox(
            'Error column (y-axis)', 
            options=numeric_columns, 
            key='07_fit_error_column_y'
        )
        
        # Filter out NaN values for the error columns
        df_clean = df.dropna(subset=[plot_config.error_column_x, plot_config.error_column_y])
        
        return df_clean[plot_config.error_column_x].values, df_clean[plot_config.error_column_y].values
    
    def _perform_simple_fit(self, df: pd.DataFrame, plot_config: PlotConfig) -> None:
        """Perform simple polynomial fitting without errors.
        
        Args:
            df: DataFrame being plotted
            plot_config: Plot configuration
        """
        x_data = df[plot_config.x_column].values
        y_data = df[plot_config.y_column].values
        
        self.sss['07_fit_results'] = self.data_fitter.fit_polynomial_simple(
            x_data, y_data, plot_config.fit_order
        )
    
    def _perform_error_weighted_fit(
        self, df: pd.DataFrame, plot_config: PlotConfig,
        error_x: np.ndarray, error_y: np.ndarray
    ) -> Optional[object]:
        """Perform error-weighted polynomial fitting.
        
        Args:
            df: DataFrame being plotted
            plot_config: Plot configuration
            error_x: X-axis errors
            error_y: Y-axis errors
            
        Returns:
            OGLS object or None if fitting failed
        """
        try:
            x_data = df[plot_config.x_column].values
            y_data = df[plot_config.y_column].values
            
            x_fit, y_fit, ci, equation, ogls = self.data_fitter.fit_polynomial_with_errors(
                x_data, y_data, error_x, error_y, plot_config.fit_order
            )
            
            self.sss['07_fit_results'] = (x_fit, y_fit, equation, ci)
            
            with st.expander('Fit parameters'):
                st.write(ogls)
            
            return ogls
            
        except ValueError:
            st.info('Error: Please choose numeric columns for x and y if you want to perform a fit.')
            st.stop()
    
    def _create_plot(
        self, df: pd.DataFrame, plot_config: PlotConfig,
        error_x: Optional[np.ndarray], error_y: Optional[np.ndarray],
        ogls_object: Optional[object]
    ) -> go.Figure:
        """Create the main plot with optional fit line.
        
        Args:
            df: DataFrame to plot
            plot_config: Plot configuration
            error_x: X-axis error bars
            error_y: Y-axis error bars
            ogls_object: OGLS object for error-weighted fits
            
        Returns:
            Plotly figure
        """
        # Determine if error bars should be shown
        show_errors = (plot_config.fit_enabled and 
                      plot_config.fit_error_type != FitErrorType.NONE)
        
        fig = self.plot_generator.create_scatter_plot(
            df, plot_config,
            error_x=error_x if show_errors else None,
            error_y=error_y if show_errors else None
        )
        
        # Add fit line if fitting is enabled
        if plot_config.fit_enabled and '07_fit_results' in self.sss:
            fit_results = self.sss['07_fit_results']
            
            if len(fit_results) == 3:  # Simple fit
                x_fit, y_fit, equation = fit_results
                fig = self.plot_generator.add_fit_line(fig, x_fit, y_fit, equation)
            elif len(fit_results) == 4:  # Error-weighted fit
                x_fit, y_fit, equation, ci = fit_results
                fig = self.plot_generator.add_fit_line(fig, x_fit, y_fit, equation, ci)
        
        return fig


if __name__ == "__main__":
    page = DiscoverResultsPage()
    page.run()