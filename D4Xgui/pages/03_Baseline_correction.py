#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import difflib
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import linregress, t

from tools.Pysotope_fork import Pysotope
from tools.page_config import PageConfigManager
from tools.sidebar_logo import SidebarLogoManager
from tools.authenticator import Authenticator
from tools.database import DatabaseManager
from tools.commons import PlotParameters, modify_plot_text_sizes, PlotlyConfig

# st.session_state

class BaselineCorrectionPage:
    """Manages the Baseline Correction page of the D4Xgui application."""

    # Constants for baseline correction methods
    METHOD_MINIMIZE = "Minimize equilibrated gase slope"
    METHOD_ETH = "Correct baseline using ETH-1 & ETH-2"
    METHOD_NONE = "Without baseline correction"
    METHOD_CUSTOM = "Use custom standards..."

    # Processing constants
    LEVEL = "replicate"
    LEVEL_ETF = "replicate"
    OPTIMIZE = "leastSquares"
    WG_RATIOS = {"d18O": 25.260, "d13C": -4.20}

    # Standard values for different isotope systems
    STANDARD_D47 = {
        "1000C": 0.0266,
        "25C": 0.9196,
        "ETH-1": 0.2052,
        "ETH-2": 0.2085,
    }
    STANDARD_D48 = {
        "1000C": 0.0,
        "25C": 0.345,
        "ETH-1": 0.1286,
        "ETH-2": 0.1286,
    }
    STANDARD_D49 = {
        "1000C": 0.0,
        "25C": 2.228,
        "ETH-1": 0.562,
        "ETH-2": 0.707,
    }

    def __init__(self):
        """Initialize the BaselineCorrectionPage."""
        self.sss = st.session_state
        self.db_manager = DatabaseManager()
        self.symbols = PlotParameters.SYMBOLS
        self._setup_page()
        self._initialize_session_state()

    def _setup_page(self) -> None:
        """Set up page configuration, logo, and authentication."""
        page_config_manager = PageConfigManager()
        page_config_manager.configure_page(page_number=3)
        
        logo_manager = SidebarLogoManager()
        logo_manager.add_logo()
        
        if "PYTEST_CURRENT_TEST" not in os.environ:
            authenticator = Authenticator()
            if not authenticator.require_authentication():
                st.stop()

    def _initialize_session_state(self) -> None:
        """Initialize session state with default parameters if not present."""
        # Initialize standard values in session state
        if "STANDARD_D47" not in self.sss:
            self.sss.STANDARD_D47 = self.STANDARD_D47.copy()
        if "STANDARD_D48" not in self.sss:
            self.sss.STANDARD_D48 = self.STANDARD_D48.copy()
        if "STANDARD_D49" not in self.sss:
            self.sss.STANDARD_D49 = self.STANDARD_D49.copy()
        
        if "to_remove" not in self.sss:
            self.sss.to_remove = "not set"
        if "bg_success" not in self.sss:
            self.sss.bg_success = False

    def run(self) -> None:
        """Run the main application page."""
        st.title("Baseline Correction")
        
        if not self._validate_input_data():
            return
            
        self._render_sidebar_controls()
        
        if self.sss.get("bg_success", False):
            self._render_results_tabs()
        else:
            st.markdown("Please perform a baseline correction after choosing standards and method.")
            self.sss['03_pbl_log'] = ''

    def _validate_input_data(self) -> bool:
        """Validate that required input data is available."""
        if "input_intensities" not in self.sss or len(self.sss.input_intensities) == 0:
            st.markdown(
                "Please upload raw intensity data to perform a baseline correction "
                "(:violet[Upload m/z44-m/z49 intensities] tab)."
            )
            st.page_link("pages/01_Data_IO.py", label=r"$\rightarrow  \textit{Data-IO}$  page")
            
            if "input_rep" in self.sss and len(self.sss.input_rep) > 0:
                st.markdown(
                    r"You have uploaded replicate data and can standardize pre-processed "
                    r"$\delta^{45}$-$\delta^{49}$ values directly. (Baseline correction is recommended!)"
                )
                st.page_link("pages/04_Processing.py", label=r"$\rightarrow  \textit{Processing}$  page")
            
            st.stop()
            return False
        return True

    def _render_sidebar_controls(self) -> None:
        """Render the sidebar controls for baseline correction settings."""
        with st.sidebar:
            st.checkbox('Overwrite database with new d45-d49', key='03_overwrite_data', value=True)
            
            self._render_method_selection()
            self._render_standard_selection()
            
            if st.button("Run...", key="BUTTON1"):
                self._execute_baseline_correction()

    def _render_method_selection(self) -> None:
        """Render the baseline correction method selection."""
        has_half_mass = f"raw_r47.5" in self.sss.input_intensities
        
        if has_half_mass:
            methods = [self.METHOD_MINIMIZE, self.METHOD_ETH, self.METHOD_NONE, self.METHOD_CUSTOM]
            help_text = None
        else:
            methods = [self.METHOD_NONE]
            help_text = (
                "No half-mass cup data provided (`raw_s47.5` and `raw_r47.5`) within the intensity input. "
                "Therefore, the baseline correction method via optimized scaling factors is not available."
            )
        
        st.radio(
            label="Baseline correction method",
            options=methods,
            key="bg_method",
            help=help_text
        )

    def _render_standard_selection(self) -> None:
        """Render the standard sample selection based on the chosen method."""
        samples = sorted(self.sss.input_intensities["Sample"].unique())
        
        if self.sss.get("bg_method") == self.METHOD_MINIMIZE:
            self._render_gas_standard_selection(samples)
        elif self.sss.get("bg_method") == self.METHOD_ETH:
            self._render_eth_standard_selection(samples)
        elif self.sss.get("bg_method") == self.METHOD_CUSTOM:
            self._render_custom_standard_selection(samples)

    def _render_gas_standard_selection(self, samples: List[str]) -> None:
        """Render gas standard (25C/1000C) selection."""
        # 25C standard selection
        try:
            idx_25 = samples.index(difflib.get_close_matches("25C", samples, n=1)[0])
        except (IndexError, ValueError):
            idx_25 = 0
        
        self.sss.bg_25_name = st.selectbox(
            label="25C sample name", 
            options=samples, 
            index=idx_25
        )

        # 1000C standard selection
        try:
            idx_1000 = samples.index(difflib.get_close_matches("1000C", samples, n=1)[0])
        except (IndexError, ValueError):
            idx_1000 = 0
        
        self.sss.bg_1000_name = st.selectbox(
            label="1000C sample name", 
            options=samples, 
            index=idx_1000
        )

    def _render_eth_standard_selection(self, samples: List[str]) -> None:
        """Render ETH standard selection."""
        # ETH-1 standard selection
        try:
            idx_eth1 = samples.index(difflib.get_close_matches("ETH-1", samples, n=1)[0])
        except (IndexError, ValueError):
            idx_eth1 = 0
        
        self.sss.bg_ETH1_name = st.selectbox(
            label="ETH-1 sample name",
            options=samples,
            index=idx_eth1
        )

        # ETH-2 standard selection
        try:
            idx_eth2 = samples.index(difflib.get_close_matches("ETH-2", samples, n=1)[0])
        except (IndexError, ValueError):
            idx_eth2 = 0
        
        self.sss.bg_ETH2_name = st.selectbox(
            label="ETH-2 sample name",
            options=samples,
            index=idx_eth2
        )

    def _render_custom_standard_selection(self, samples: List[str]) -> None:
        """Render custom standard selection."""
        # Custom standard selection
        st.write("Custom samples")
        COLS = st.columns(3)
        

        
        for col, mz in zip(COLS, ["47", "48", "49"]):
            if f"bg_custom_{mz}_values" not in st.session_state:
                st.session_state[f"bg_custom_{mz}_values"] = '{\n"Sample1": 0.000,\n"Sample2": 1.000\n}'
            with col:
                st.checkbox(f'$\Delta_{{{mz}}}$', key=f'bg_custom_{mz}', value=True if mz == '47' else False)
            
            if st.session_state[f'bg_custom_{mz}']:
                st.text_area(
                    label=f"$\Delta_{{{mz}}}$ standards",
                    key=f"bg_custom_{mz}_values"
                )

    def _execute_baseline_correction(self) -> None:
        """Execute the baseline correction process."""
        method = self.sss.get("bg_method", self.METHOD_NONE)
        
        # Update standard names based on method
        if method == self.METHOD_MINIMIZE:
            self._update_gas_standard_names()
        elif method == self.METHOD_ETH:
            self._update_eth_standard_names()
       
        # Process the dataset
        df = self._process_dataset()
        
        # Add missing columns
        for col in ("Type", "Project"):
            if col not in df:
                df[col] = [np.nan] * len(df)

        # Create aggregated replicate data
        self._create_replicate_dataframe(df)
        
        # Update database if requested
        if self.sss.get('03_overwrite_data', False):
            self._update_database()
        
        self.sss.bg_success = True

    def _update_gas_standard_names(self) -> None:
        """Update gas standard names in session state."""
        for temp, name_key in [(25, "bg_25_name"), (1000, "bg_1000_name")]:
            name = self.sss.get(name_key)
            if name:
                old_key = f"{temp}C"
                if old_key in self.sss.STANDARD_D47 and name not in self.sss.STANDARD_D47:
                    for standard_dict in [self.sss.STANDARD_D47, self.sss.STANDARD_D48, self.sss.STANDARD_D49]:
                        standard_dict[name] = standard_dict[old_key]
                        del standard_dict[old_key]

    def _update_eth_standard_names(self) -> None:
        """Update ETH standard names in session state."""
        for nr, name_key in [(1, "bg_ETH1_name"), (2, "bg_ETH2_name")]:
            name = self.sss.get(name_key)
            if name:
                old_key = f"ETH-{nr}"
                if old_key in self.sss.STANDARD_D47 and name not in self.sss.STANDARD_D47:
                    for standard_dict in [self.sss.STANDARD_D47, self.sss.STANDARD_D48, self.sss.STANDARD_D49]:
                        standard_dict[name] = standard_dict[old_key]
                        del standard_dict[old_key]

    def _process_dataset(self) -> pd.DataFrame:
        """Process all sessions and return concatenated results."""
        pysotope = Pysotope()

        # Add data for all sessions
        for session, df_session in self.sss.input_intensities.groupby("Session", as_index=False):
            pysotope.add_data(session, df_session)

        # Configure pysotope
        pysotope.level = self.LEVEL
        pysotope.level_etf = self.LEVEL_ETF
        pysotope.optimize = self.OPTIMIZE
        pysotope.set_wg_ratios(self.WG_RATIOS)

        # Set standards based on method
        self._configure_pysotope_standards(pysotope)

        # Process all sessions
        all_data = pd.DataFrame()
        sessions = sorted(self.sss.input_intensities["Session"].unique())
        
        method = self.sss.get("bg_method", self.METHOD_NONE)
        title = (
            r"Correcting baseline and calculating $\delta^{45}$-$\delta^{49}$..."
            if method != self.METHOD_NONE
            else r"Calculating $\delta^{45}$-$\delta^{49}$ without baseline correction..."
        )
        self.sss['03_pbl_log'] = ''
        with st.status(title, expanded=True) as status:
            #cols = st.columns(len(sessions))
            for idx, session in enumerate(sessions):
                #with cols[idx]:
                    #st.write(f"## {session}...")
                    if method != self.METHOD_NONE:
                        self.sss['03_pbl_log'] = str(self.sss['03_pbl_log'])+ f"\n## Session {session}..."
                    pysotope = self._process_session(pysotope, session)
                    pysotope.analyses[session]["Session"] = session
                    all_data = pd.concat([all_data, pysotope.analyses[session]], ignore_index=True)
            #st.write(self.sss['03_pbl_log'])
            if method != self.METHOD_NONE:
                status.update(label="Baseline correction completed!", state="complete", expanded=False)
            else:
                status.update(label=r"Calculated $\delta^{45}$-$\delta^{49}$ without baseline correction!", state="complete", expanded=False)

        return all_data

    def _configure_pysotope_standards(self, pysotope: Pysotope) -> None:
        """Configure standards for pysotope based on the selected method."""
        method = self.sss.get("bg_method", self.METHOD_NONE)
        
        if method == self.METHOD_CUSTOM:
            #standards= {}
            for mz in 47,48,49:
                #st.write(json.loads(self.sss.get(f'bg_custom_{mz}_values', '{}')))
                # if self.sss[f'bg_custom_{mz}']:
                pysotope.set_standards(system=f"D{mz}", standards=json.loads(self.sss.get(f'bg_custom_{mz}_values', '{}')))
                # else:
                #     pysotope.set_standards(system=f"D{mz}", standards=dict())
                #st.write('ööö',pysotope.standards)
        else:
            if "ETH" in method:
                std_keys = ["ETH-1", "ETH-2"]
            else:
                std_keys = ["25C", "1000C"]
    
            # Get standards for each system
            for system, standard_dict in [("D47", self.sss.STANDARD_D47),
                                         ("D48", self.sss.STANDARD_D48),
                                         ("D49", self.sss.STANDARD_D49)]:
                standards = {key: standard_dict[key] for key in std_keys if key in standard_dict}
                pysotope.set_standards(system=system, standards=standards)

    def _process_session(self, pysotope: Pysotope, session: str) -> Pysotope:
        """Process a single session with the selected baseline correction method."""
        pysotope.calc_sample_ratios_1(session=session)

        method = self.sss.get("bg_method", self.METHOD_NONE)
        
        if method == self.METHOD_MINIMIZE:
            self._apply_minimize_method(pysotope, session)
        elif method == self.METHOD_ETH:
            self._apply_eth_method(pysotope, session)
        elif method == self.METHOD_CUSTOM:
            self._apply_custom_method(pysotope, session)
        else:
            pysotope.calc_sample_ratios_2(mode="raw", session=session)
        
        return pysotope

    def _apply_minimize_method(self, pysotope: Pysotope, session: str) -> None:
        """Apply the minimize equilibrated gas slope method."""
        pysotope.optimize = "leastSquares"
        pysotope.correctBaseline(
            scaling_mode="scale",
            session=session,
            scaling_factors=None,
            D47std={"25C": self.sss.STANDARD_D47["25C"], "1000C": self.sss.STANDARD_D47["1000C"]},
            D48std={"25C": self.sss.STANDARD_D48["25C"], "1000C": self.sss.STANDARD_D48["1000C"]},
            D49std={"25C": self.sss.STANDARD_D49["25C"], "1000C": self.sss.STANDARD_D49["1000C"]},
        )
        self.sss["scaling_factors"] = dict(pysotope.scaling_factors)
        pysotope.calc_sample_ratios_2(mode="bg", session=session)

    def _apply_eth_method(self, pysotope: Pysotope, session: str) -> None:
        """Apply the ETH standards method."""
        pysotope.optimize = "ETH"
        pysotope.correctBaseline(
            scaling_mode="scale",
            session=session,
            scaling_factors=None,
            D47std={"ETH-1": self.sss.STANDARD_D47["ETH-1"], "ETH-2": self.sss.STANDARD_D47["ETH-2"]},
            D48std={"ETH-1": self.sss.STANDARD_D48["ETH-1"], "ETH-2": self.sss.STANDARD_D48["ETH-2"]},
            D49std={"ETH-1": self.sss.STANDARD_D49["ETH-1"], "ETH-2": self.sss.STANDARD_D49["ETH-2"]},
        )
        self.sss["scaling_factors"] = dict(pysotope.scaling_factors)
        pysotope.calc_sample_ratios_2(mode="bg", session=session)
        
    def _apply_custom_method(self, pysotope: Pysotope, session: str) -> None:
        """Apply the ETH standards method."""
        pysotope.optimize = "customStds"
        # print(json.loads(self.sss.get(f'bg_custom_47_values', '{}')))
        # works
        st.write("json.loads(self.sss.get(f'bg_custom_47_values', '{}'))", json.loads(self.sss.get(f'bg_custom_47_values', '{}')))
        pysotope.correctBaseline(
            scaling_mode="scale",
            session=session,
            scaling_factors=None,
            D47std= json.loads(self.sss.get(f'bg_custom_47_values', '{}')),
            D48std = json.loads(self.sss.get(f'bg_custom_48_values', '{}')),
            D49std = json.loads(self.sss.get(f'bg_custom_49_values', '{}')),
            # D47std={"ETH-1": self.sss.STANDARD_D47["ETH-1"], "ETH-2": self.sss.STANDARD_D47["ETH-2"]},
            # D48std={"ETH-1": self.sss.STANDARD_D48["ETH-1"], "ETH-2": self.sss.STANDARD_D48["ETH-2"]},
            # D49std={"ETH-1": self.sss.STANDARD_D49["ETH-1"], "ETH-2": self.sss.STANDARD_D49["ETH-2"]},
        )
        self.sss["scaling_factors"] = dict(pysotope.scaling_factors)
        st.write(self.sss["scaling_factors"])
        pysotope.calc_sample_ratios_2(mode="bg", session=session)
        

    def _create_replicate_dataframe(self, df: pd.DataFrame) -> None:
        """Create aggregated replicate dataframe from cycle-level data."""
        agg_dict = self._get_aggregation_dict(df)
        
        # Create replicate-level data
        self.sss.input_intensities = df
        self.sss.input_rep = df.groupby("Replicate", as_index=False).agg(agg_dict)
        self.sss.input_rep.sort_values(by="UID", inplace=True)
        self.sss.input_rep["UID"] = list(range(len(self.sss.input_rep)))

    def _get_aggregation_dict(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get the aggregation dictionary for creating replicate-level data."""
        agg_dict = {
            "UID": "first",
            "Sample": "first",
            "Type": "first",
            "Project": "first",
            "Session": "first",
            "Replicate": "first",
            "Timetag": "first",
            "raw_s44": "mean",
            "raw_s45": "mean",
            "raw_s46": "mean",
            "raw_s47": "mean",
            "raw_s48": "mean",
            "raw_s49": "mean",
            "raw_r44": "mean",
            "raw_r45": "mean",
            "raw_r46": "mean",
            "raw_r47": "mean",
            "raw_r48": "mean",
            "raw_r49": "mean",
            "d45": "mean",
            "d46": "mean",
            "d47": "mean",
            "d48": "mean",
            "d49": "mean",
            "D47": "mean",
            "D48": "mean",
            "D49": "mean",
        }

        # Add half-mass columns if present
        for mz in (47, 48):
            if f"raw_r{mz}.5" in self.sss.input_intensities:
                agg_dict.update({
                    f"raw_s{mz}.5": "mean",
                    f"raw_r{mz}.5": "mean",
                })

        # Add baseline-corrected columns if present
        if "bg_s47" in df:
            agg_dict.update({
                "bg_s47": "mean",
                "bg_s48": "mean",
                "bg_s49": "mean",
                "bg_r47": "mean",
                "bg_r48": "mean",
                "bg_r49": "mean",
            })

        return agg_dict

    def _update_database(self) -> None:
        """Update the database with processed data."""
        if "input_rep" not in self.sss:
            return
            
        session_name = str(self.sss.input_rep['Session'].values[0]) if 'Session' in self.sss.input_rep else 'unknown'
        rows_affected = self.db_manager.upsert_dataframe(self.sss.input_rep, session_name)
        
        if rows_affected > 0:
            st.success(f"{rows_affected} rows inserted/updated in the database.")
        else:
            st.info("No data changes detected for the database.")

    def _render_results_tabs(self) -> None:
        """Render the results tabs after successful baseline correction."""
        if "scaling_factors" in self.sss:
            tabs = st.tabs([
                "Determined optimal scaling factors",
                r"Resulting $\delta^{45}-\delta^{49}$ dataframe",
                "Overview plot (m/z intensities)",
                r"Overview plot ($\delta^{i}/\Delta_{i}$)",
                "Heated gas lines",
                "Correlation matrix",
            ])
            


            with tabs[0]:
                st.json(self.sss["scaling_factors"])
                if '03_pbl_log' in self.sss:
                    self.sss['03_pbl_log']
            
            tab_offset = 1
        else:
            tabs = st.tabs([
                r"Resulting $\delta^{45}-\delta^{49}$ dataframe",
                "Overview plot (m/z intensities)",
                r"Overview plot ($\delta^{i}/\Delta_{i}$)",
                "Heated gas lines",
                "Correlation matrix",
            ])
            tab_offset = 0

        with tabs[tab_offset]:
            self._render_data_tables()
        
        with tabs[tab_offset + 1]:
            self._render_intensity_plots()
        
        with tabs[tab_offset + 2]:
            self._render_delta_plots()
        
        with tabs[tab_offset + 3]:
            self._render_heated_gas_lines()
        
        with tabs[tab_offset + 4]:
            self._render_correlation_matrix()

    def _render_data_tables(self) -> None:
        """Render the data tables for replicate and cycle data."""
        with st.expander("### Replicate data"):
            st.data_editor(self.sss.input_rep, num_rows="dynamic")
        with st.expander("### Cycle data"):
            st.data_editor(self.sss.input_intensities, num_rows="dynamic")

    def _render_intensity_plots(self) -> None:
        """Render the intensity overview plots."""
        col1, col2 = st.columns(2)
        with col1:
            self.sss.rep_cycle = st.radio(
                label="Choose level",
                options=("Replicates :grey[(Cycles are deactivated in the cloud version for performance reasons...)]",),
            )
        with col2:
            st.checkbox(label="Normalize data", key="03_normalize_int")

        df = self.sss.input_rep if "Rep" in self.sss.rep_cycle else self.sss.input_intensities
        
        self.sss.filter_intensities = st.multiselect(
            label="Select samples to plot",
            options=sorted(df["Sample"].unique()),
            default=sorted(df["Sample"].unique()),
        )

        fig = self._create_intensity_plot()
        st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)

    def _render_delta_plots(self) -> None:
        """Render the delta/Delta overview plots."""
        col1, col2 = st.columns(2)
        with col1:
            self.sss.mz_03 = st.selectbox("Select m/z", options=[47, 48, 49], index=0)
        with col2:
            pass  # Reserved for future controls

        fig = self._create_delta_overview_plot(self.sss.mz_03)
        st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)

    def _render_heated_gas_lines(self) -> None:
        """Render the heated gas line plots."""
        col1, col2 = st.columns(2)
        with col1:
            self.sss.mz_03 = st.selectbox("Select m/z", options=[47, 48, 49], index=0, key="hgl_mz")
        
        # Get available gas standards (only 25C and 1000C, not ETH carbonates)
        available_gases = []
        for gas in ["25C", "1000C"]:
            if gas in self.sss.input_rep["Sample"].values:
                available_gases.append(gas)
        
        if available_gases:
            with col2:
                selected_gas = st.selectbox("Select gas standard", options=available_gases)
            
            fig = self._create_heated_gas_line_plot(selected_gas, self.sss.mz_03)
            st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)
        else:
            st.info("No heated gas standards (25C/1000C) found in the data.")

    def _render_correlation_matrix(self) -> None:
        """Render the correlation matrix."""
        if "input_rep" in self.sss:
            numeric_cols = self.sss.input_rep.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = self.sss.input_rep[numeric_cols].corr()
                fig = self._create_correlation_matrix_plot(corr_matrix)
                st.plotly_chart(fig, config=PlotlyConfig.CONFIG)
            else:
                st.info("Not enough numeric columns for correlation matrix.")

    def _create_intensity_plot(self) -> go.Figure:
        """Create the intensity overview plot."""
        filtered = self.sss.input_rep if "Rep" in self.sss.rep_cycle else self.sss.input_intensities
        
        # Filter by selected samples
        if hasattr(self.sss, 'filter_intensities') and self.sss.filter_intensities:
            filtered = filtered[filtered["Sample"].isin(self.sss.filter_intensities)]

        layout = go.Layout(
            xaxis=dict(title="Datetime"),
            yaxis=dict(title="Intensity [mV]"),
            hoverlabel=dict(font_size=16),
            legend=dict(font_size=15, title="m/z signal"),
        )
        fig = go.Figure(layout=layout)

        # Define columns to plot
        cols = [
            "raw_s44", "raw_s45", "raw_s46", "raw_s47", "raw_s48", "raw_s49",
            "raw_r44", "raw_r45", "raw_r46", "raw_r47", "raw_r48", "raw_r49",
        ]
        
        # Add baseline-corrected columns if available
        if "bg_s47" in filtered:
            cols.extend(["bg_s47", "bg_s48", "bg_s49", "bg_r47", "bg_r48", "bg_r49"])
        
        # Add half-mass columns if available
        for mz in (47, 48):
            if f"raw_r{mz}.5" in self.sss.input_intensities:
                cols.extend([f"raw_s{mz}.5", f"raw_r{mz}.5"])

        # Create traces for each column
        for idx, col in enumerate(cols):
            if col in filtered.columns:
                y_data = (
                    self._normalize_series(filtered[col]) + idx * 1.1
                    if self.sss.get("03_normalize_int", False)
                    else filtered[col]
                )
                
                scatter_trace = go.Scatter(
                    x=filtered["Timetag"],
                    y=y_data,
                    mode="markers",
                    opacity=0.75,
                    name=col,
                    marker=dict(size=12, line=dict(width=0.5)),
                    text=[
                        f"Sample={sample}<br>Timetag={timetag}<br>UID={uid}"
                        for sample, timetag, uid in zip(
                            filtered["Sample"], filtered["Timetag"], filtered["UID"]
                        )
                    ],
                )
                fig.add_trace(scatter_trace)

        fig.update_layout(height=500)
        fig.update_traces(textfont_size=15)
        fig.update_xaxes(showline=True, linewidth=2, linecolor="white", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="white", mirror=True)
        
        return fig

    def _create_delta_overview_plot(self, mz: int) -> go.Figure:
        """Create the delta/Delta overview plot."""
        layout = go.Layout(
            xaxis=dict(title=f"δ<sup>{mz}</sup> [‰]"),
            yaxis=dict(title=f"∆<sub>{mz}, raw</sub> [‰]"),
            hoverlabel=dict(font_size=16),
            legend=dict(font_size=15),
        )
        fig = go.Figure(layout=layout)

        df = self.sss.input_rep
        
        for idx, sample in enumerate(df["Sample"].unique()):
            df_sample = df[df["Sample"] == sample]
            scatter_trace = go.Scatter(
                x=df_sample[f"d{mz}"],
                y=df_sample[f"D{mz}"],
                mode="markers",
                opacity=0.75,
                name=sample,
                marker=dict(size=12, line=dict(width=0.5), symbol=self.symbols[idx % len(self.symbols)]),
                text=[
                    f"{sample}<br>Datetime={timetag}"
                    for timetag in df_sample["Timetag"]
                ],
            )
            fig.add_trace(scatter_trace)

        fig.update_traces(textfont_size=15)
        fig.update_xaxes(showline=True, linewidth=2, linecolor="white", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="white", mirror=True)
        
        return fig

    def _create_heated_gas_line_plot(self, gas: str, mz: int) -> go.Figure:
        """Create the heated gas line plot."""
        layout = go.Layout(
            xaxis=dict(title=f"δ<sup>{mz}</sup> [‰]"),
            yaxis=dict(title=f"∆<sub>{mz}</sub> [‰]"),
            hoverlabel=dict(font_size=16),
            legend=dict(font_size=15),
        )
        fig = go.Figure(layout=layout)

        df_std = self.sss.input_rep[self.sss.input_rep["Sample"] == gas]
        
        if df_std.empty:
            st.warning(f"No data found for gas standard: {gas}")
            return fig

        # Scatter plot
        scatter_trace = go.Scatter(
            x=df_std[f"d{mz}"],
            y=df_std[f"D{mz}"],
            mode="markers",
            opacity=0.75,
            name=gas,
            marker=dict(size=12, line=dict(width=0.5), color="red" if gas == "1000C" else "blue"),
            text=[
                f"{gas}<br>UID={uid}<br>d<sub>{mz}</sub>={round(x, 3)} ‰<br>∆<sub>{mz}</sub>={round(y, 3)} ‰"
                for uid, x, y in zip(df_std["UID"], df_std[f"d{mz}"], df_std[f"D{mz}"])
            ],
        )
        fig.add_trace(scatter_trace)

        # Linear regression
        if len(df_std) > 1:
            res = linregress(df_std[f"d{mz}"], df_std[f"D{mz}"])
            
            # Regression line
            reg_line = go.Scatter(
                x=df_std[f"d{mz}"],
                y=res.intercept + res.slope * df_std[f"d{mz}"],
                mode="lines",
                line_color="red" if gas == "1000C" else "blue",
                opacity=0.75,
                name=None,
                showlegend=False,
            )
            fig.add_trace(reg_line)

            # Add regression equation
            tinv = lambda p, df: abs(t.ppf(p / 2, df))
            ts = tinv(0.05, len(df_std) - 2)
            
            equation = (
                f"f(x)=({res.slope:.2e} ± {res.stderr:.2e})x"
                f"{'+' if res.intercept >= 0 else ''}{res.intercept:.2e} ± {res.intercept_stderr:.2e}"
            )
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.01, y=0.94,
                showarrow=False,
                text=f"<b>{equation}</b>",
                font=dict(size=18, color="grey"),
            )

        return fig

    def _create_correlation_matrix_plot(self, corr_matrix: pd.DataFrame) -> go.Figure:
        """Create a correlation matrix heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            xaxis_title="Variables",
            yaxis_title="Variables",
            height=600
        )
        
        return fig

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        """Normalize a numeric pandas Series to [0, 1]."""
        denom = series.max() - series.min()
        return (series - series.min()) / denom if denom != 0 else series - series.min()


if __name__ == "__main__":
    page = BaselineCorrectionPage()
    page.run()