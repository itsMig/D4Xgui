#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import difflib
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import linregress, t
import re

from tools.Pysotope_fork import Pysotope
from tools.base_page import BasePage
from tools import config as user_cfg
from tools.constants import ISOTOPIC_CONSTANTS
from tools.database import DatabaseManager
from tools.commons import PlotParameters, modify_plot_text_sizes, PlotlyConfig


class BaselineCorrectionPage(BasePage):
    """Manages the Baseline Correction page of the D4Xgui application."""

    PAGE_NUMBER = 3

    # Constants for baseline correction methods
    METHOD_MINIMIZE = "Minimize equilibrated gase slope"
    METHOD_ETH = "Correct baseline using ETH-1 & ETH-2"
    METHOD_NONE = "Without baseline correction"
    METHOD_CUSTOM = "Use custom standards..."

    # Processing constants
    LEVEL = "replicate"
    LEVEL_ETF = "replicate"
    OPTIMIZE = "leastSquares"
    WG_RATIOS = user_cfg.get("working_gas_ratios")

    GAS_LABEL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*C\s*$", re.IGNORECASE)

    def __init__(self):
        """Initialize the BaselineCorrectionPage."""
        self.db_manager = DatabaseManager()
        self.symbols = PlotParameters.SYMBOLS
        super().__init__()

    def _initialize_session_state(self) -> None:
        """Initialize session state with default parameters if not present."""
        self.sss.STANDARD_D47 = user_cfg.get("standard_d47")
        self.sss.STANDARD_D48 = user_cfg.get("standard_d48")
        self.sss.STANDARD_D49 = user_cfg.get("standard_d49")
        
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
            default_idx = 0
            cfg_method = user_cfg.get("baseline_correction_method")
            if cfg_method in methods:
                default_idx = methods.index(cfg_method)
            help_text = None
        else:
            methods = [self.METHOD_NONE]
            default_idx = 0
            help_text = (
                "No half-mass cup data provided (`raw_s47.5` and `raw_r47.5`) within the intensity input. "
                "Therefore, the baseline correction method via optimized scaling factors is not available."
            )
        
        st.radio(
            label="Baseline correction method",
            options=methods,
            index=default_idx,
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
        """Render equilibrated gas standard selection (2+ sets supported)."""
        gas_candidates = self._infer_equilibrated_gas_candidates(samples)
        default_candidates = gas_candidates if gas_candidates else samples
        
        # Prefer 25C/1000C if present, otherwise select first two candidates.
        default_selection: List[str] = []
        for preferred in ("25C", "1000C"):
            if preferred in default_candidates:
                default_selection.append(preferred)
        if len(default_selection) < 2:
            for s in default_candidates:
                if s not in default_selection:
                    default_selection.append(s)
                if len(default_selection) >= 2:
                    break
        
        selected = st.multiselect(
            label="Equilibrated gas samples (choose 2+)",
            options=default_candidates,
            default=default_selection,
            key="bg_gas_samples",
            help="These samples are used to determine optimal baseline scaling factors. Select 2 or more equilibrated gas sets (e.g., 25C, 50C, 1000C).",
        )
        
        if len(selected) < 2:
            st.warning("Select at least two equilibrated gas samples.")
            self.sss.bg_gas_table = pd.DataFrame()
            return
        
        table = self._build_gas_table(selected, None)

        st.caption("Standard values from **Settings**.")
        st.dataframe(table, width='stretch', hide_index=True)

        missing = table[["D47", "D48", "D49"]].isna().any(axis=1)
        if missing.any():
            names = table.loc[missing, "Sample"].tolist()
            st.warning(
                f"Missing standard values for: {', '.join(names)}. "
                "Configure them on the **Settings** page."
            )

        self.sss.bg_gas_table = table
        self.sss.bg_gas_standards = list(table["Sample"].astype(str).values)
    
    def _infer_equilibrated_gas_candidates(self, samples: List[str]) -> List[str]:
        """Infer equilibrated gas sample candidates from names (e.g., '25C', '1000C')."""
        out: List[str] = []
        for s in samples:
            if not isinstance(s, str):
                continue
            if self.GAS_LABEL_RE.match(s) and "ETH" not in s.upper():
                out.append(s)
        return out
    
    def _parse_temp_from_label(self, label: str) -> Optional[float]:
        """Parse a temperature (°C) from a label like '25C'."""
        if not isinstance(label, str):
            return None
        m = self.GAS_LABEL_RE.match(label)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None
    
    def _build_gas_table(self, selected_samples: List[str], existing: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Create an editable table for equilibrated gas definitions."""
        # Preserve existing edits where possible.
        existing_map: Dict[str, Dict[str, Any]] = {}
        if isinstance(existing, pd.DataFrame) and not existing.empty and "Sample" in existing.columns:
            for _, row in existing.iterrows():
                existing_map[str(row.get("Sample"))] = {
                    "Temp_C": row.get("Temp_C"),
                    "D47": row.get("D47"),
                    "D48": row.get("D48"),
                    "D49": row.get("D49"),
                }
        
        rows: List[Dict[str, Any]] = []
        for s in selected_samples:
            s_str = str(s)
            prev = existing_map.get(s_str, {})
            temp_c = prev.get("Temp_C")
            if pd.isna(temp_c) or temp_c is None:
                temp_c = self._parse_temp_from_label(s_str)
            
            # Default expected values:
            # - Prefer any values stored under the sample name in session state (from previous renames),
            # - else fall back to canonical "25C"/"1000C" if the name matches those,
            # - else leave as NaN so user can fill in.
            def _default_val(std_dict: Dict[str, float], key: str) -> Any:
                if s_str in std_dict:
                    return std_dict[s_str]
                if key in std_dict and s_str == key:
                    return std_dict[key]
                return np.nan
            
            rows.append({
                "Sample": s_str,
                "Temp_C": temp_c if temp_c is not None else np.nan,
                "D47": prev.get("D47", _default_val(self.sss.STANDARD_D47, s_str)),
                "D48": prev.get("D48", _default_val(self.sss.STANDARD_D48, s_str)),
                "D49": prev.get("D49", _default_val(self.sss.STANDARD_D49, s_str)),
            })
        
        df = pd.DataFrame(rows)
        # Ensure numeric types where possible.
        for col in ("Temp_C", "D47", "D48", "D49"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

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

        st.caption("Expected standard values (from **Settings**):")
        rows = []
        for std_key in ("ETH-1", "ETH-2"):
            rows.append({
                "Standard": std_key,
                "Δ₄₇": self.sss.STANDARD_D47.get(std_key, float("nan")),
                "Δ₄₈": self.sss.STANDARD_D48.get(std_key, float("nan")),
                "Δ₄₉": self.sss.STANDARD_D49.get(std_key, float("nan")),
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    _KNOWN_BL_KEYS = {"25C", "1000C", "ETH-1", "ETH-2"}

    def _render_custom_standard_selection(self, samples: List[str]) -> None:
        """Render custom standard selection (read-only, configured in Settings)."""
        st.caption("Custom standard values (from **Settings**):")

        st.markdown("**Samples in dataset:** " + ", ".join(f"`{s}`" for s in samples))

        cols = st.columns(3)

        std_dicts = {"47": self.sss.STANDARD_D47,
                     "48": self.sss.STANDARD_D48,
                     "49": self.sss.STANDARD_D49}

        for col, mz in zip(cols, ["47", "48", "49"]):
            with col:
                st.checkbox(
                    rf"$\Delta_{{{mz}}}$",
                    key=f"bg_custom_{mz}",
                    value=(mz == "47"),
                )

            if st.session_state.get(f"bg_custom_{mz}", False):
                custom_entries = {
                    k: v for k, v in std_dicts[mz].items()
                    if k not in self._KNOWN_BL_KEYS
                }
                table_key = f"bg_custom_{mz}_table"
                if custom_entries:
                    df = pd.DataFrame(
                        [{"Standard": k, "Value": v}
                         for k, v in custom_entries.items()]
                    )
                    st.dataframe(df, width='stretch', hide_index=True)
                    st.session_state[table_key] = df
                else:
                    st.info(f"No custom Δ{mz} standards configured. Add them on the **Settings** page.")
                    st.session_state[table_key] = pd.DataFrame(
                        columns=["Standard", "Value"],
                    ).astype({"Standard": str, "Value": float})

    def _collect_selected_standard_names(self, method: str) -> List[str]:
        """Return the list of standard sample names the user selected for *method*."""
        if method == self.METHOD_MINIMIZE:
            gas_table = getattr(self.sss, "bg_gas_table", None)
            if isinstance(gas_table, pd.DataFrame) and not gas_table.empty:
                return list(gas_table["Sample"].astype(str))
            return list(getattr(self.sss, "bg_gas_standards", ["25C", "1000C"]))

        if method == self.METHOD_ETH:
            names: List[str] = []
            for key in ("bg_ETH1_name", "bg_ETH2_name"):
                v = self.sss.get(key)
                if v:
                    names.append(str(v))
            return names if names else ["ETH-1", "ETH-2"]

        if method == self.METHOD_CUSTOM:
            all_names: set[str] = set()
            for mz in ("47", "48", "49"):
                if self.sss.get(f"bg_custom_{mz}", False):
                    stds = self._extract_custom_standards(mz)
                    all_names.update(stds.keys())
            return sorted(all_names)

        return []

    def _execute_baseline_correction(self) -> None:
        """Execute the baseline correction process."""
        method = self.sss.get("bg_method", self.METHOD_NONE)

        if method != self.METHOD_NONE:
            selected_stds = self._collect_selected_standard_names(method)
            dataset_samples = set(self.sss.input_intensities["Sample"].unique())
            found = [s for s in selected_stds if s in dataset_samples]

            if len(found) < 2:
                found_str = ", ".join(f"`{s}`" for s in found) if found else "none"
                selected_str = ", ".join(f"`{s}`" for s in selected_stds)
                st.warning(
                    f"**At least 2 selected standards must be present in the dataset.**\n\n"
                    f"Selected standards: {selected_str}\n\n"
                    f"Found in dataset ({len(found)}): {found_str}"
                )
                return

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
        """Update gas standard definitions in session state based on selected gas table."""
        # Backward-compatibility: if legacy keys exist, keep them as-is.
        gas_table = getattr(self.sss, "bg_gas_table", None)
        if not isinstance(gas_table, pd.DataFrame) or gas_table.empty:
            return
        
        # Persist the selected gases and their expected values by sample name.
        for _, row in gas_table.iterrows():
            sample = str(row.get("Sample"))
            if not sample:
                continue
            for system_key, std_dict in [("D47", self.sss.STANDARD_D47),
                                         ("D48", self.sss.STANDARD_D48),
                                         ("D49", self.sss.STANDARD_D49)]:
                val = row.get(system_key)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                std_dict[sample] = float(val)

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
        pysotope = Pysotope(isotopic_constants=ISOTOPIC_CONSTANTS)

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
            st.write(self.sss['03_pbl_log'])
            if method != self.METHOD_NONE:
                status.update(label="Baseline correction completed!", state="complete", expanded=False)
            else:
                status.update(label=r"Calculated $\delta^{45}$-$\delta^{49}$ without baseline correction!", state="complete", expanded=False)

        return all_data

    def _extract_custom_standards(self, mz: str) -> dict:
        """Extract a {Standard: Value} dict from the custom data_editor table."""
        table_key = f"bg_custom_{mz}_table"
        df = self.sss.get(table_key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        result = {}
        for _, row in df.iterrows():
            name = row.get("Standard", "")
            if isinstance(name, (list, tuple)):
                name = name[0] if name else ""
            name = str(name).strip()
            if name:
                try:
                    result[name] = float(row["Value"])
                except (ValueError, TypeError):
                    pass
        return result

    def _configure_pysotope_standards(self, pysotope: Pysotope) -> None:
        """Configure standards for pysotope based on the selected method."""
        method = self.sss.get("bg_method", self.METHOD_NONE)
        
        if method == self.METHOD_CUSTOM:
            for mz in ("47", "48", "49"):
                if self.sss.get(f"bg_custom_{mz}", False):
                    stds = self._extract_custom_standards(mz)
                else:
                    stds = {}
                pysotope.set_standards(system=f"D{mz}", standards=stds)
        else:
            if "ETH" in method:
                std_keys = ["ETH-1", "ETH-2"]
            elif method == self.METHOD_MINIMIZE and isinstance(getattr(self.sss, "bg_gas_table", None), pd.DataFrame):
                # Use all selected equilibrated gas sets (2+ supported)
                std_keys = list(getattr(self.sss, "bg_gas_standards", []))
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
        gas_table = getattr(self.sss, "bg_gas_table", None)
        if isinstance(gas_table, pd.DataFrame) and not gas_table.empty:
            d47std = {str(r["Sample"]): float(r["D47"]) for _, r in gas_table.dropna(subset=["D47"]).iterrows()}
            d48std = {str(r["Sample"]): float(r["D48"]) for _, r in gas_table.dropna(subset=["D48"]).iterrows()}
            d49std = {str(r["Sample"]): float(r["D49"]) for _, r in gas_table.dropna(subset=["D49"]).iterrows()}
        else:
            d47std = {"25C": self.sss.STANDARD_D47["25C"], "1000C": self.sss.STANDARD_D47["1000C"]}
            d48std = {"25C": self.sss.STANDARD_D48["25C"], "1000C": self.sss.STANDARD_D48["1000C"]}
            d49std = {"25C": self.sss.STANDARD_D49["25C"], "1000C": self.sss.STANDARD_D49["1000C"]}
        
        pysotope.correctBaseline(
            scaling_mode="scale",
            session=session,
            scaling_factors=None,
            D47std=d47std,
            D48std=d48std,
            D49std=d49std,
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
        """Apply the custom carbonate standards method."""
        pysotope.optimize = "customStds"
        pysotope.correctBaseline(
            scaling_mode="scale",
            session=session,
            scaling_factors=None,
            D47std=self._extract_custom_standards("47"),
            D48std=self._extract_custom_standards("48"),
            D49std=self._extract_custom_standards("49"),
        )
        self.sss["scaling_factors"] = dict(pysotope.scaling_factors)
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
        method = self.sss.get("bg_method", self.METHOD_NONE)
        show_hgl = method not in (self.METHOD_ETH, self.METHOD_CUSTOM)

        tab_labels: list[str] = []
        if "scaling_factors" in self.sss:
            tab_labels.append("Determined optimal scaling factors")
        tab_labels += [
            r"Resulting $\delta^{45}-\delta^{49}$ dataframe",
            "Overview plot (m/z intensities)",
            r"Overview plot ($\delta^{i}/\Delta_{i}$)",
        ]
        if show_hgl:
            tab_labels.append("Heated gas lines")
        tab_labels.append("Correlation matrix")

        tabs = st.tabs(tab_labels)
        ti = 0

        if "scaling_factors" in self.sss:
            with tabs[ti]:
                st.json(self.sss["scaling_factors"])
                if '03_pbl_log' in self.sss:
                    self.sss['03_pbl_log']
            ti += 1

        with tabs[ti]:
            self._render_data_tables()
        ti += 1

        with tabs[ti]:
            self._render_intensity_plots()
        ti += 1

        with tabs[ti]:
            self._render_delta_plots()
        ti += 1

        if show_hgl:
            with tabs[ti]:
                self._render_heated_gas_lines()
            ti += 1

        with tabs[ti]:
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
        
        # Use the same set of gases used for baseline determination (if available),
        # otherwise fall back to any 'xxC' samples found.
        preferred = list(getattr(self.sss, "bg_gas_standards", []))
        if not preferred:
            preferred = self._infer_equilibrated_gas_candidates(sorted(self.sss.input_rep["Sample"].unique()))
        
        available_gases = [g for g in preferred if g in self.sss.input_rep["Sample"].values]
        
        if not available_gases:
            st.info("No equilibrated gas standards found in the data.")
            return
        
        with col2:
            selected_gases = st.multiselect(
                "Select gas standards to plot",
                options=available_gases,
                default=available_gases,
                key="hgl_selected_gases",
            )
        
        if not selected_gases:
            st.info("Select at least one gas standard to plot.")
            return
        
        fig = self._create_heated_gas_lines_plot(selected_gases, self.sss.mz_03)
        st.plotly_chart(modify_plot_text_sizes(fig), config=PlotlyConfig.CONFIG)

    def _create_heated_gas_lines_plot(self, gases: List[str], mz: int) -> go.Figure:
        """Create a combined heated gas line plot for multiple gas sets."""
        layout = go.Layout(
            xaxis=dict(title=f"δ<sup>{mz}</sup> [‰]"),
            yaxis=dict(title=f"∆<sub>{mz}</sub> [‰]"),
            hoverlabel=dict(font_size=16),
            legend=dict(font_size=15),
        )
        fig = go.Figure(layout=layout)
        
        palette = [
            "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        
        annotations = []
        for idx, gas in enumerate(gases):
            df_std = self.sss.input_rep[self.sss.input_rep["Sample"] == gas]
            if df_std.empty:
                continue
            
            color = palette[idx % len(palette)]
            
            fig.add_trace(go.Scatter(
                x=df_std[f"d{mz}"],
                y=df_std[f"D{mz}"],
                mode="markers",
                opacity=0.75,
                name=gas,
                marker=dict(size=12, line=dict(width=0.5), color=color),
                text=[
                    f"{gas}<br>UID={uid}<br>d<sub>{mz}</sub>={round(x, 3)} ‰<br>∆<sub>{mz}</sub>={round(y, 3)} ‰"
                    for uid, x, y in zip(df_std["UID"], df_std[f"d{mz}"], df_std[f"D{mz}"])
                ],
            ))
            
            if len(df_std) > 1:
                res = linregress(df_std[f"d{mz}"], df_std[f"D{mz}"])
                x_line = np.linspace(df_std[f"d{mz}"].min(), df_std[f"d{mz}"].max(), 50)
                y_line = res.intercept + res.slope * x_line
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color=color, width=2),
                    opacity=0.75,
                    showlegend=False,
                    name=None,
                ))
                equation = (
                    f"<b>{gas}:</b> slope={res.slope:.2e} ± {res.stderr:.2e}, "
                    f"intercept={res.intercept:.2e} ± {res.intercept_stderr:.2e}"
                )
                annotations.append(dict(
                    xref="paper", yref="paper",
                    x=0.01, y=0.97 - idx * 0.06,
                    showarrow=False,
                    text=equation,
                    font=dict(size=14, color=color),
                ))
        
        fig.update_layout(annotations=annotations)
        fig.update_traces(textfont_size=15)
        fig.update_xaxes(showline=True, linewidth=2, linecolor="white", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="white", mirror=True)
        return fig

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
        
        for idx, sample in enumerate(sorted(df["Sample"].unique())):
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