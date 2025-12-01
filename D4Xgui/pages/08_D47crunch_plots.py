#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Any

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from tools.authenticator import Authenticator
from tools.page_config import PageConfigManager
from tools.sidebar_logo import SidebarLogoManager
from tools.commons import modify_text_label_sizes, PlotlyConfig


class D47CrunchPlotsPage:
    """Manages the D47crunch plots page of the D4Xgui application."""

    def __init__(self):
        """Initialize the D47CrunchPlotsPage."""
        self.sss = st.session_state
        self._setup_page()
        self._validate_data()

    def _setup_page(self) -> None:
        """Set up page configuration, logo, and authentication."""
        st.title("D47crunch plots")
        
        page_config_manager = PageConfigManager()
        page_config_manager.configure_page(page_number=8)
        
        logo_manager = SidebarLogoManager()
        logo_manager.add_logo()
        
        if "PYTEST_CURRENT_TEST" not in os.environ:
            authenticator = Authenticator()
            if not authenticator.require_authentication():
                st.stop()

    def _validate_data(self) -> None:
        """Validate that required data is available in session state."""
        # Check for input data
        has_input_rep = "input_rep" in self.sss and len(self.sss.input_rep) > 0
        has_input_intensities = "input_intensities" in self.sss and len(self.sss.input_intensities) > 0
        
        if not has_input_rep and not has_input_intensities:
            st.markdown("Please provide input data to be processed!")
            st.page_link("pages/01_Data_IO.py", label=r"$\rightarrow  \textit{Data-IO}$  page")
            st.stop()

        # Check for processing results
        if "correction_output_summary" not in self.sss:
            st.markdown("Please process your dataset to view D47crunch plots!")
            st.page_link(
                "pages/04_Processing.py", label=r"$\rightarrow  \textit{Processing}$  page"
            )
            st.stop()

    def _get_available_isotopes(self) -> List[int]:
        """Get list of available isotopes based on processing parameters."""
        isotopes = []
        for isotope in [47, 48, 49]:
            if self.sss.params_last_run.get(f"process_D{isotope}", False):
                isotopes.append(isotope)
        return isotopes

    def _render_sidebar_controls(self) -> None:
        """Render sidebar controls for isotope selection and plot options."""
        available_isotopes = self._get_available_isotopes()
        
        if not available_isotopes:
            st.sidebar.error("No processed isotopes available!")
            return
        
        st.sidebar.radio(
            label="$∆_i$", 
            options=available_isotopes, 
            key="D47c_plots_mz"
        )
        st.sidebar.checkbox(
            label="Shape filling", 
            key="D47c_plots_shape", 
            value=True
        )

    def _get_hover_data(self, session: str, data_type: str, std_err_obj: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get hover data for sample names and timestamps.
        
        Args:
            session: Session identifier
            data_type: Either 'anchors' or 'unknowns'
            
        Returns:
            Dictionary containing sample names and timestamps for hover display
        """
        hover_data = {"sample_names": [], "timestamps": [], 'UID': []}
    
        df_session = self.sss.input_rep.loc[self.sss.input_rep['Session'] == session].sort_values(by='Timetag')    
        df_anchors_unknowns = df_session.loc[df_session['Sample'].isin(self.sss[f"D{self.sss.D47c_plots_mz}_standardization_error_{session}_{data_type}"])]
        hover_data["sample_names"] = df_anchors_unknowns['Sample'].values
        hover_data["UID"] = df_anchors_unknowns['UID'].values
        hover_data["timestamps"] = [str(_) for _ in df_anchors_unknowns['Timetag']]
       
        return hover_data

    def _get_standard_samples(self) -> List[str]:
        """Get list of standard sample names from session state."""
        standards = []
        if "standards_nominal" in self.sss:
            scale = self.sss.params_last_run.get("scale", "Fiebig2024 carb")
            isotope_key = str(self.sss.D47c_plots_mz)
            if scale in self.sss.standards_nominal and isotope_key in self.sss.standards_nominal[scale]:
                standards = list(self.sss.standards_nominal[scale][isotope_key].keys())
        return standards

    def _create_scatter_trace(
        self, 
        x_data: List[float], 
        y_data: List[float], 
        name: str, 
        color: str, 
        symbol: str = "circle",
        hover_data: Optional[Dict[str, List[str]]] = None
    ) -> go.Scatter:
        """Create a scatter trace with enhanced hover information.
        
        Args:
            x_data: X-axis data points
            y_data: Y-axis data points
            name: Trace name
            color: Marker color
            symbol: Marker symbol
            hover_data: Optional hover data containing sample names and timestamps
            
        Returns:
            Plotly Scatter trace
        """
        # Prepare hover text
        hover_text = []
        
        # Check if we have reliable hover data (both sample names and timestamps)
        if (hover_data and 
            hover_data.get("sample_names") and 
            hover_data.get("timestamps") and
            len(hover_data["sample_names"]) == len(x_data) and
            len(hover_data["timestamps"]) == len(x_data)):
            
            # We have matching data - create detailed hover text
            for i, (sample, timestamp, UID) in enumerate(zip(
                hover_data["sample_names"], 
                hover_data["timestamps"],
                hover_data["UID"]
            )):
                hover_text.append(
                    f"UID: {UID}<br>"
                    f"Sample: {sample}<br>"
                    f"Timestamp: {timestamp}<br>"
                    f"δ{self.sss.D47c_plots_mz}: {x_data[i]:.4f}‰<br>"
                    f"∆{self.sss.D47c_plots_mz}: {y_data[i]:.4f}‰"
                )
        else:
            # Use default hover with just coordinates and data type
            data_type_label = "Standard" if name == "anchors" else "Sample"
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                hover_text.append(
                    f"{data_type_label} {i+1}<br>"
                    f"δ{self.sss.D47c_plots_mz}: {x:.4f}‰<br>"
                    f"∆{self.sss.D47c_plots_mz}: {y:.4f}‰"
                )
        
        return go.Scatter(
            x=x_data,
            y=y_data,
            name=name,
            mode="markers",
            marker=dict(color=color, symbol=symbol),
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
            text=hover_text
        )

    def _create_contour_plot(self, contour_data: Any) -> go.Contour:
        """Create the contour plot for standardization error.
        
        Args:
            contour_data: Contour data from session state
            
        Returns:
            Plotly Contour trace
        """
        return go.Contour(
            x=contour_data[0][0, :],
            y=contour_data[1][:, 0],
            z=contour_data[2],
            colorscale=px.colors.sequential.YlGnBu[:-5:],
            line=dict(width=2, color="black"),
            contours=dict(
                coloring="heatmap" if self.sss.D47c_plots_shape else "lines",
                showlabels=True,
                labelfont=dict(size=12, color="grey"),
            ),
            colorbar=dict(title="Standardization error [‰]"),
        )

    def _create_plot(self, session: str, mz: int, std_err_obj: Dict[str, Any]) -> go.Figure:
        """Create the complete standardization error plot.
        
        Args:
            session: Session identifier
            mz: Mass number (47, 48, or 49)
            std_err_obj: Standardization error data object
            
        Returns:
            Complete Plotly figure
        """
        # Create layout
        layout = go.Layout(
            height=600,
            title="Standardization error including standards and sample data points",
            xaxis=dict(title=f"δ<sup>{mz}</sup> [‰]"),
            yaxis=dict(title=f"∆<sub>{mz}</sub> [‰]"),
            hoverlabel=dict(font=dict(family="sans-serif", size=18)),
        )
        
        fig = go.Figure(layout=layout)
        
        # Add contour plot
        contour_data = std_err_obj[f"D{mz}_standardization_error_{session}_contour"]
        contour_trace = self._create_contour_plot(contour_data)
        fig.add_trace(contour_trace)
        
        # Add anchors (standards) scatter plot
        anchors_x = std_err_obj[f"D{mz}_standardization_error_{session}_anchors_d"]
        anchors_y = std_err_obj[f"D{mz}_standardization_error_{session}_anchors_D"]
        anchors_hover = self._get_hover_data(session, "anchors", std_err_obj)
        
        
        anchors_trace = self._create_scatter_trace(
            anchors_x, anchors_y, "anchors", "red", "circle", anchors_hover
        )
        fig.add_trace(anchors_trace)
        
        # Add unknowns scatter plot
        unknowns_x = std_err_obj[f"D{mz}_standardization_error_{session}_unknowns_d"]
        unknowns_y = std_err_obj[f"D{mz}_standardization_error_{session}_unknowns_D"]
        unknowns_hover = self._get_hover_data(session, "unknowns", std_err_obj)
        
        unknowns_trace = self._create_scatter_trace(
            unknowns_x, unknowns_y, "unknowns", "black", "x", unknowns_hover
        )
        fig.add_trace(unknowns_trace)
        
        # Apply text size modifications
        fig = modify_text_label_sizes(fig)
        
        return fig

    def _get_standardization_error_data(self) -> Dict[str, Any]:
        """Get standardization error data from session state.
        
        Returns:
            Dictionary containing standardization error data
        """
        selected_mz = self.sss.D47c_plots_mz
        std_err_obj = {
            key: self.sss[key] 
            for key in self.sss.keys() 
            if f"D{selected_mz}_standardization_error_" in key
        }
        return std_err_obj

    def run(self) -> None:
        """Run the main application page."""
        st.title("D47crunch Plots")
        
        # Render sidebar controls
        self._render_sidebar_controls()
        
        # Get standardization error data
        std_err_obj = self._get_standardization_error_data()
        
        if not std_err_obj:
            st.warning("No standardization error data available for the selected isotope.")
            st.stop()
        
        # Get processing sessions
        sessions = self.sss.params_last_run.get("processing_sessions", [])
        selected_mz = self.sss.D47c_plots_mz
        
        # Display plots for each session
        for session in sessions:
            if self.sss.params_last_run.get(f"process_D{selected_mz}", False):
                # Filter data for this session
                session_err_obj = {
                    key: value for key, value in std_err_obj.items() 
                    if session in key
                }
                
                if session_err_obj:
                    with st.expander(f"Session {session}", expanded=True):
                        fig = self._create_plot(session, selected_mz, session_err_obj)
                        st.plotly_chart(fig, config=PlotlyConfig.CONFIG)


if __name__ == "__main__":
    page = D47CrunchPlotsPage()
    page.run()