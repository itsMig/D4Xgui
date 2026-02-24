#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from dateutil import parser as dateutil_parser

import D47crunch as D47c
from scipy.stats import t
from tools.base_page import BasePage
from tools.constants import (
    VPDB_FACTOR, VPDB_OFFSET, KELVIN_OFFSET, DEFAULT_ACID_TEMPERATURE,
    ISOTOPIC_CONSTANTS, D18O_VPDB_VSMOW, DEFAULT_WG_RATIOS,
    DEFAULT_WG_VIA_STANDARDS, DEFAULT_CO2_STANDARDS,
)
from tools.init_params import IsotopeStandards
from tools.commons import clear_session_cache
from tools.calc_temperature import TemperatureCalculator as _BaseTemperatureCalculator
from scipy import optimize as so

@dataclass
class ProcessingConfig:
    """Configuration class for processing parameters."""
    process_D47: bool = False
    process_D48: bool = False
    process_D49: bool = False
    scale: str = "CDES"
    correction_method: str = "pooled"
    processing_sessions: List[str] = None
    drifting_sessions: List[str] = None
    selected_calibrations: List[str] = None
    
    def __post_init__(self):
        if self.processing_sessions is None:
            self.processing_sessions = []
        if self.drifting_sessions is None:
            self.drifting_sessions = []
        if self.selected_calibrations is None:
            self.selected_calibrations = ["Fiebig24 (original)"]


class IsotopeProcessor:
    """Handles isotope data processing using D47crunch."""
    
    #ACID_TEMPERATURE = 90  # Celsius
    
    
    def __init__(self, session_state: st.session_state):
        self.sss = session_state
        

    @staticmethod
    def find_key_value(obj, key):
            """Recursively search for key in nested dict/list and return its value."""
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                for v in obj.values():
                    result = IsotopeProcessor.find_key_value(v, key)
                    if result is not None:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = IsotopeProcessor.find_key_value(item, key)
                    if result is not None:
                        return result
            return None
        
    def _initialize_d47crunch_object(self, d47crunch_obj: Any) -> None:
        """Initialize D47crunch object with reference samples and constants."""
        raw_data = self.sss.raw_data
        
        # Set reference sample for Levene test
        if "1000C" in raw_data["Sample"].values:
            d47crunch_obj.LEVENE_REF_SAMPLE = "1000C"
        elif "ETH-1" in raw_data["Sample"].values:
            d47crunch_obj.LEVENE_REF_SAMPLE = "ETH-1"
        else:
            d47crunch_obj.LEVENE_REF_SAMPLE = raw_data["Sample"].iloc[0]
        
        if self.sss.get('working_gas_co2_stds', False):
            d47crunch_obj.ALPHA_18O_ACID_REACTION = 1
        else:
            d47crunch_obj.ALPHA_18O_ACID_REACTION = np.exp(
                3.59 / (self.sss.temp_acid + KELVIN_OFFSET) - 1.79e-3
            )

        ic = ISOTOPIC_CONSTANTS
        d47crunch_obj.R13_VPDB = ic["R13_VPDB"]
        d47crunch_obj.R17_VSMOW = ic["R17_VSMOW"]
        d47crunch_obj.R18_VSMOW = ic["R18_VSMOW"]
        d47crunch_obj.LAMBDA_17 = ic.get("lambda_17", ic.get("lambda_", 0.528))
        d47crunch_obj.R18_VPDB = d47crunch_obj.R18_VSMOW * (1 + D18O_VPDB_VSMOW / 1000)
        d47crunch_obj.R17_VPDB = d47crunch_obj.R17_VSMOW * (
            d47crunch_obj.R18_VPDB / d47crunch_obj.R18_VSMOW
        ) ** d47crunch_obj.LAMBDA_17

        if self.sss.working_gas:  # "Working gas composition via standards"
            # 🔑 KEY: Use session state values (preserves user edits)
            d47crunch_obj.Nominal_d18O_VPDB = self.sss['standards_bulk'][18].copy()
            d47crunch_obj.Nominal_d13C_VPDB = self.sss['standards_bulk'][13].copy()
            
            # Determine standardization method based on number of anchors
            d47crunch_obj.d18O_standardization_method = '1pt' if len(self.sss['standards_bulk'][18]) == 1 else '2pt'
            d47crunch_obj.d13C_standardization_method = '1pt' if len(self.sss['standards_bulk'][13]) == 1 else '2pt'
            
            # Calculate working gas composition
            d47crunch_obj.wg()
            
            # Extract and store working gas values
            self.sss['d13Cwg_VPDB'] = self.find_key_value(d47crunch_obj.__dict__, "d13Cwg_VPDB")
            self.sss['d18Owg_VSMOW'] = self.find_key_value(d47crunch_obj.__dict__, "d18Owg_VSMOW")
            
            # 🔑 KEY: Store what was actually used for this processing run
            self.sss['bulk_anchors_d13C_used'] = self.sss['standards_bulk'][13].copy()
            self.sss['bulk_anchors_d18O_used'] = self.sss['standards_bulk'][18].copy()
        
        
        else:
            d47crunch_obj.d18O_standardization_method = False
            d47crunch_obj.d13C_standardization_method = False
            # Set nominal isotope values
            self.sss['d18Owg_VSMOW'] = self.sss.d18O_wg
            self.sss['d13Cwg_VPDB'] = self.sss.d13C_wg
            for record in d47crunch_obj:
                record["d13Cwg_VPDB"] = self.sss.d13C_wg
                record["d18Owg_VSMOW"] = self.sss.d18O_wg
                
            for s in d47crunch_obj.sessions:
                d47crunch_obj.sessions[s]['d13C_standardization_method'] = False
                d47crunch_obj.sessions[s]["d18O_standardization_method"] = False
                

    def _activate_drift_corrections(self, d47crunch_obj: Any) -> None:
        """Activate drift corrections for all sessions."""
        for session in d47crunch_obj.sessions:
            d47crunch_obj.sessions[session]["scrambling_drift"] = True
            d47crunch_obj.sessions[session]["wg_drift"] = True
            d47crunch_obj.sessions[session]["slope_drift"] = True
    
    def _process_isotope_data(self, isotope_type: str) -> Optional[Any]:
        """Process isotope data for specified type (D47, D48, or D49)."""
        isotope_classes = {
            "D47": D47c.D47data,
            "D48": D47c.D48data,
            "D49": D47c.D49data
        }
        mz_keyditc = {
            'D47': r"$\Delta_{47}$",
             'D48': r"$\Delta_{48}$",
              'D49': r"$\Delta_{49}$",
        }
        
        if isotope_type not in isotope_classes:
            raise ValueError(f"Invalid isotope type: {isotope_type}")
        
        st.toast(f"Processing {mz_keyditc[isotope_type]} data...")
        
        # Create and initialize processor
        processor = isotope_classes[isotope_type](verbose=False)
        processor.input(self.sss.csv_text)
        
        # Set nominal values
        nominal_key = f"Nominal_{isotope_type}"
        setattr(processor, nominal_key, 
                self.sss["standards_nominal"][self.sss.scale][isotope_type[-2:]])
        
        self._initialize_d47crunch_object(processor)
        processor.crunch()
        
        if isotope_type == "D47":
            st.toast("Bulk isotope processing finished!")
        
        self._activate_drift_corrections(processor)
        
        # Standardize data
        if self.sss.correction_method == "pooled":
            processor.standardize(method=self.sss.correction_method, verbose=False)
        else:
            processor.split_samples(grouping="by_session")
            processor.standardize(method="indep_sessions", verbose=False)
            processor.unsplit_samples()
            
        # Store session results
        for session in self.sss.processing_sessions:
            results = processor.plot_single_session(session, fig=None)
            for key, value in results.items():
                self.sss[f"{isotope_type}_standardization_error_{session}_{key}"] = value
        
        
        st.toast(f"{mz_keyditc[isotope_type]} processing finished!")
        return processor
    
    def process_all_isotopes(self, config: ProcessingConfig) -> Dict[str, Any]:
        """Process all selected isotope types."""
        clear_session_cache()
        processors = {}
        
        isotope_types = ["D47", "D48", "D49"]
        process_flags = [config.process_D47, config.process_D48, config.process_D49]
        
        for isotope_type, should_process in zip(isotope_types, process_flags):
            if should_process:
                processors[isotope_type] = self._process_isotope_data(isotope_type)
                # Store with correct attribute name for merging
                setattr(self.sss, f"D47c_{isotope_type[-2:]}", processors[isotope_type])
            else:
                setattr(self.sss, f"D47c_{isotope_type[-2:]}", None)
        
        st.success("Standardization finished!", icon="✅")
        return processors


class TemperatureCalculator(_BaseTemperatureCalculator):
    """Extends base TemperatureCalculator with session-state-dependent methods."""

    def __init__(self, session_state: st.session_state = st.session_state):
        self.sss = session_state

    def calc_temp(self, summary):
        calibs = self.sss["04_selected_calibs"]

        self.sss['04_used_calibs'] = [_ for _ in calibs]

        if "Fiebig24 (original)" in calibs:
            for (label, key, sign) in zip (['min, 2SE', 'min, 1SE', 'mean', 'max, 1SE', 'max, 2SE'],
                                            ['2SE_D47', 'SE_D47', 'D47', 'SE_D47', '2SE_D47'],
                                            [+1, +1, 0, -1, -1]
                                     ):
                summary[f"T({label}), Fiebig24 (original)"] = [
                    round(so.minimize_scalar(self.get_temperature_difference_d47_fiebig2024,
                                             args=(t,)).x, 2)
                    for t in (summary["D47"] + (summary[key]) * sign)
                ]

        if "Anderson21 (original)" in calibs:
            for (label, key, sign) in zip(['min, 2SE', 'min, 1SE', 'mean', 'max, 1SE', 'max, 2SE'],
                                          ['2SE_D47', 'SE_D47', 'D47', 'SE_D47', '2SE_D47'],
                                          [+1, +1, 0, -1, -1]
                                          ):
                summary[f"T({label}), Anderson21 (original)"] = [
                            round(
                                - KELVIN_OFFSET + so.minimize_scalar(
                                    self.get_temperature_difference_d47_anderson2021, args=(t,), bounds=(0.000000001, 1000)
                                ).x,
                                2,
                            )
                            for t in (summary["D47"] + (summary[key]) * sign)
                        ]

        if "Swart21 (original)" in calibs:
            for (label, key, sign) in zip(['min, 2SE', 'min, 1SE', 'mean', 'max, 1SE', 'max, 2SE'],
                                          ['2SE_D47', 'SE_D47', 'D47', 'SE_D47', '2SE_D47'],
                                          [+1, +1, 0, -1, -1]
                                          ):

                summary[f"T({label}), Swart21 (original)"] = [
                    round(self.direct_temperature_swart2021(t), 2)
                    for t in (summary["D47"] + (summary[key]) * sign)
                ]

        for calib in calibs:
            if calib in [
                "Fiebig24 (original)",
                "Fiebig21 (original)",
                "Swart21 (original)",
                "Anderson21 (original)"
            ]:
                continue
            for idx in range(len(summary)):
                try:

                    summary.loc[summary.index == idx, f"T(min, 2SE), {calib}"] = round(
                        self.sss["04_calibs"][calib].__dict__["_T_from_D47"](
                            summary.loc[summary.index == idx, 'D47'] + summary.loc[summary.index == idx, '2SE_D47'])[0],
                        2)

                    summary.loc[summary.index == idx, f"T(min, 1SE), {calib}"] = round(
                        self.sss["04_calibs"][calib].__dict__["_T_from_D47"](
                            summary.loc[summary.index == idx, 'D47'] + summary.loc[summary.index == idx, 'SE_D47'])[0],
                        2)

                    summary.loc[summary.index == idx, f"T(mean), {calib}"] = round(
                        self.sss["04_calibs"][calib].__dict__["_T_from_D47"](summary.loc[summary.index == idx, 'D47'])[0], 2)

                    summary.loc[summary.index == idx, f"T(max, 1SE), {calib}"] = round(
                        self.sss["04_calibs"][calib].__dict__["_T_from_D47"](
                            summary.loc[summary.index == idx, 'D47'] - summary.loc[summary.index == idx, 'SE_D47'])[0],
                        2)

                    summary.loc[summary.index == idx, f"T(max, 2SE), {calib}"] = round(
                        self.sss["04_calibs"][calib].__dict__["_T_from_D47"](
                            summary.loc[summary.index == idx, 'D47'] - summary.loc[summary.index == idx, '2SE_D47'])[0],
                        2)

                except Exception as e:
                    summary.loc[summary.index == idx, f"T(mean), {calib}"] = str(e)

        return summary


class DataProcessor:
    """Handles data merging and processing operations."""
    
    def __init__(self, session_state: st.session_state):
        self.sss = session_state
        self.DATETIME_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ]
    
    @staticmethod
    def smart_numeric_conversion(series):
        """Convert series to numeric only if it contains numeric data."""
        # Check if the series contains any numeric-like values
        
        col_name = series.name
        if col_name in ['Sample', 'Session']:
            return series
        
        numeric_count = 0
        total_non_null = 0
        CHECK_ARR = []
        for val in series:
            if pd.notna(val):
                total_non_null += 1
                try:
                    # Try to convert to float
                    float(str(val))
                    CHECK_ARR.append(True)
                    numeric_count += 1
                except (ValueError, TypeError):
                    CHECK_ARR.append(False)
                    pass
            else:
                CHECK_ARR.append(False)
        
        new_series = []
        for idx, _ in enumerate(series):
            new_series.append(float(_) if CHECK_ARR[idx] else (np.nan if len(str(_)) == 0 else _))
        
        return new_series
    
    @staticmethod
    def apply_smart_numeric_conversion(df):
        """Apply smart numeric conversion to all columns in a DataFrame."""
        return df.apply(DataProcessor.smart_numeric_conversion)
    
    def merge_datasets(self) -> None:
        """Merge D47crunch outputs into unified datasets."""
        full_dataset = None
        summary = None
        
        # Process each isotope type
        isotope_objects = [
            getattr(self.sss, f"D47c_47", None),
            getattr(self.sss, f"D47c_48", None),
            getattr(self.sss, f"D47c_49", None),
        ]
        
        # Check which objects are available
        available_isotopes = []
        for i, obj in enumerate(isotope_objects):
            isotope_num = ["47", "48", "49"][i]
            if obj is not None:
                available_isotopes.append(isotope_num)
        
        if not available_isotopes:
            st.warning("No isotope data available for merging")
            return
        
        # Merge full datasets
        for i, obj in enumerate(isotope_objects):
            if obj is None:
                continue
                
            temp_full = self._extract_analysis_table(obj)
            if full_dataset is None:
                # First dataset - take all columns
                full_dataset = temp_full.copy()
            else:
                # For subsequent datasets, merge on UID and add new columns
                # Get columns that are not already in full_dataset (except UID)
                existing_cols = set(full_dataset.columns)
                new_cols = [col for col in temp_full.columns if col not in existing_cols or col == "UID"]
                
                if len(new_cols) > 1:  # More than just UID
                    full_dataset = pd.merge(
                        full_dataset, 
                        temp_full[new_cols], 
                        on=["UID"],
                        how="outer"
                    )
                else:
                    # If no new columns, still try to merge to ensure all UIDs are included
                    full_dataset = pd.merge(
                        full_dataset, 
                        temp_full[["UID"]], 
                        on=["UID"],
                        how="outer"
                    )
        
        # Add metadata from input replicates if we have a dataset
        if full_dataset is not None:
            full_dataset = self._add_metadata_to_dataset(full_dataset)
        
        # Process summary data
        summary = self._create_summary_dataset(isotope_objects, full_dataset)
        
        # Store results and session data
        return_dict = {"full_dataset": full_dataset, "summary": summary}
        self._process_session_data(isotope_objects, return_dict)
        
        # Store all results in session state
        for key, value in return_dict.items():
            self.sss[f"correction_output_{key}"] = value
    
    def _extract_analysis_table(self, obj: Any) -> pd.DataFrame:
        """Extract and format analysis table from D47crunch object."""
        temp_full = D47c.table_of_analyses(
            obj, save_to_file=False, print_out=False, output="raw"
        )
        df = pd.DataFrame(temp_full[1:], columns=temp_full[0])
        df = self.apply_smart_numeric_conversion(df)
        return df.sort_values("Sample")
    
    def _add_metadata_to_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add metadata from input replicates to the dataset."""
        if dataset is None or dataset.empty:
            return dataset
            
        # Ensure required columns exist
        for col in ['Project', 'Type']:
            if col not in self.sss.input_rep.columns:
                self.sss.input_rep[col] = ''
        
        # Merge with input replicate metadata
        dataset = pd.merge(
            dataset,
            self.sss.input_rep[["UID", "Timetag", "Project", "Type"]],
            on=["UID"],
            how="left"
        )
        
        # Process timestamps
        dataset = self._process_timestamps(dataset)
        
        # Add d18O_VPDB column if d18O_VSMOW exists
        if "d18O_VSMOW" in dataset.columns:
            dataset.insert(
                12, "d18O_VPDB", 
                (dataset["d18O_VSMOW"] * VPDB_FACTOR) - VPDB_OFFSET
            )
        
        return dataset
    

    
    def _is_datetime_column(dtype) -> bool:
        """Check if column is already datetime type."""
        return np.issubdtype(dtype, np.datetime64)
    
    def _try_standard_formats(self, timestamp_str: str) -> Optional[datetime]:
        """Try parsing with standard datetime formats."""
        for fmt in self.DATETIME_FORMATS:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        return None
    
    def _clean_timestamp(timestamp_str: str) -> str:
        """
        Clean timestamp string by removing problematic characters.

        Preserves: digits, spaces, colons, hyphens, periods, T separator
        Removes: timezone abbreviations (PDT, EST, etc.), extra symbols

        Args:
            timestamp_str: Raw timestamp string

        Returns:
            Cleaned timestamp string
        """
        # Remove timezone abbreviations (3-4 letter codes at end)
        # e.g., "2020-07-29 01:00 PDT" -> "2020-07-29 01:00"
        cleaned = re.sub(r'\s+[A-Z]{2,4}$', '', timestamp_str)
        
        # Keep only useful characters: digits, space, :, -, ., T, +, Z
        cleaned = "".join(
            char for char in cleaned
            if char.isdigit() or char in (" ", ":", "-", ".", "T", "+", "Z")
        )
        
        return cleaned.rstrip()
    
    def _parse_single_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse a single timestamp string with multiple fallback strategies.

        Strategies in order:
        1. Standard datetime formats (fastest)
        2. ISO format parsing
        3. dateutil flexible parser (handles timezone abbreviations like PDT, EST)
        4. Last resort: attempt to clean and reparse

        Args:
            timestamp_str: Timestamp string to parse

        Returns:
            Parsed datetime object or None if parsing fails
        """
        if pd.isna(timestamp_str) or timestamp_str == "":
            return None
        
        timestamp_str = str(timestamp_str).strip()
        
        # Strategy 1: Try standard formats (fast)
        result = self._try_standard_formats(timestamp_str)
        if result is not None:
            return result
        
        # Strategy 2: Try ISO format (handles T separator)
        try:
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            pass
        
        # Strategy 3: Use dateutil parser for flexibility
        # Handles timezone abbreviations (PDT, EST, etc.), various formats
        try:
            return dateutil_parser.parse(timestamp_str, fuzzy=False)
        except (ValueError, TypeError):
            pass
        
        # Strategy 4: Clean special characters and retry
        try:
            cleaned = self._clean_timestamp(timestamp_str)
            # Try standard formats on cleaned string
            result = self._try_standard_formats(cleaned)
            if result is not None:
                return result
            # Final attempt with dateutil
            return dateutil_parser.parse(cleaned, fuzzy=False)
        except (ValueError, TypeError):
            return None
    
    def _process_timestamps(
            self,
            dataset: pd.DataFrame,
            column: str = "Timetag",
            errors: str = "coerce",
            infer_datetime_format: bool = True,
    ) -> pd.DataFrame:
        """
        Process and standardize timestamp column in dataframe.

        Args:
            dataset: Input dataframe
            column: Name of timestamp column (default: "Timetag")
            errors: How to handle parsing errors
                - "coerce": Convert unparseable values to NaT (default)
                - "raise": Raise exception on first error
                - "warn": Log warnings but continue
            infer_datetime_format: Let pandas infer datetime format

        Returns:
            Dataframe with standardized datetime column

        Raises:
            ValueError: If column not found or errors='raise'
            KeyError: If column doesn't exist
        """
        if column not in dataset.columns:
            raise KeyError(f"Column '{column}' not found in dataframe")
        
        if isinstance(dataset[column].dtype, np.datetime64):
            return dataset
        
        dataset = dataset.copy()
        
        # Try pandas to_datetime first (fastest for standard formats)
        try:
            dataset[column] = pd.to_datetime(
                dataset[column],
                errors="coerce" if errors != "raise" else "raise",
                infer_datetime_format=infer_datetime_format,
            )
            # Check if all values parsed successfully
            if dataset[column].isna().sum() == 0:
                return dataset
        except Exception as e:
            if errors == "raise":
                raise ValueError(
                    f"Failed to parse timestamps in column '{column}': {e}"
                )


        parsed_timestamps = []
        failed_rows = []
        
        for idx, ts in enumerate(dataset[column]):
            parsed_ts = self._parse_single_timestamp(ts)
            
            if parsed_ts is None:
                failed_rows.append((idx, ts))
                if errors == "raise":
                    raise ValueError(
                        f"Cannot parse timestamp at row {idx}: {ts!r}"
                    )
            
            parsed_timestamps.append(parsed_ts)
        
        dataset[column] = pd.to_datetime(parsed_timestamps, errors="coerce")
        
        return dataset
    
    def _create_summary_dataset(self, isotope_objects: List[Any], 
                               full_dataset: pd.DataFrame) -> pd.DataFrame:
        """Create summary dataset from isotope processing results."""
        summary = None
        
        # Handle case where no full dataset exists
        if full_dataset is None or full_dataset.empty:
            return pd.DataFrame()
        
        # Create aggregated dataset for d-values
        header_cols = ["Sample", "d45", "d46"]
        for col in ("d47", "d48", "d49"):
            if col in self.sss.input_rep.columns:
                header_cols.append(col)
        
        # Filter header_cols to only include columns that exist in full_dataset
        available_header_cols = [col for col in header_cols if col in full_dataset.columns]
        
        if not available_header_cols:
            # If no header columns are available, create a basic summary
            return pd.DataFrame({"Sample": full_dataset.get("Sample", []).unique() if "Sample" in full_dataset.columns else []})
        
        agg_dataset = full_dataset[available_header_cols].groupby("Sample", as_index=False).agg("mean")
        
        # Process each isotope type
        processed_isotopes = []
        for i, obj in enumerate(isotope_objects):
            if obj is None:
                continue
                
            isotope_num = ["47", "48", "49"][i]
            processed_isotopes.append(isotope_num)
            
            try:
                temp_summary = self._extract_sample_table(obj, agg_dataset, isotope_num)
                
                if summary is None:
                    summary = self._rename_summary_columns(temp_summary, isotope_num)
                else:
                    summary = self._merge_summary_data(summary, temp_summary, isotope_num)
                    
            except Exception as e:
                st.error(f"Failed to process isotope D{isotope_num}: {str(e)}")
                continue
        
        if summary is None:
            # Create empty summary if no isotopes were processed
            summary = pd.DataFrame({"Sample": agg_dataset["Sample"]})
        
        # Add project and type information
        summary = self._add_project_type_info(summary)
        
        # Process confidence intervals
        summary = self._process_confidence_intervals(summary)
        
        # Add isotope ratio columns
        summary = self._add_isotope_ratio_columns(summary, full_dataset)
        
        return summary
    
    def _extract_sample_table(self, obj: Any, agg_dataset: pd.DataFrame, isotope_num: str = None) -> pd.DataFrame:
        """Extract sample table from D47crunch object."""
        temp_summary = D47c.table_of_samples(
            obj, save_to_file=False, print_out=False, output="raw"
        )
        df = pd.DataFrame(temp_summary[1:], columns=temp_summary[0])
        df = df.sort_values(by="Sample")
        
        # Use provided isotope_num or fall back to obj._4x
        isotope_suffix = isotope_num if isotope_num is not None else obj._4x
        
        # Check if the required column exists in agg_dataset
        d_col = f"d{isotope_suffix}"
        if d_col in agg_dataset.columns:
            return df.merge(
                agg_dataset[["Sample", d_col]], 
                on="Sample",
                how="left"
            )
        else:
            # If the d-column doesn't exist, just return the df
            return df
    
    def _rename_summary_columns(self, summary: pd.DataFrame, isotope_num: str) -> pd.DataFrame:
        """Rename summary columns for specific isotope."""
        return summary.rename(columns={
            "SD": f"SD_D{isotope_num}",
            "SE": f"SE_D{isotope_num}",
            "95% CL": f"95% CL_D{isotope_num}",
        })
    
    def _merge_summary_data(self, summary: pd.DataFrame, 
                           temp_summary: pd.DataFrame, isotope_num: str) -> pd.DataFrame:
        """Merge summary data for additional isotope."""
        # Define the columns we want to merge
        base_cols = ["Sample"]
        isotope_cols = []
        
        
        # for col in [f"d{isotope_num}", f"D{isotope_num}", "SD", "SE", "95% CL"]:
        #     if col in summary.columns:
        #         st.write(summary[col])
        #         summary[col] = summary[col].astype(float, errors='ignore')
        #         st.write(summary[col])
                
        # Check which columns exist in temp_summary
        for col in [f"d{isotope_num}", f"D{isotope_num}", "SD", "SE", "95% CL"]:
            if col in temp_summary.columns:
                isotope_cols.append(col)
        
        if not isotope_cols:
            # If no isotope columns found, return original summary
            return summary
        
        merge_cols = base_cols + isotope_cols
        
        # Create renamed columns for the isotope-specific data
        rename_dict = {}
        for col in isotope_cols:
            if col in ["SD", "SE", "95% CL"]:
                rename_dict[col] = f"{col}_D{isotope_num}"
        
        temp_summary_renamed = temp_summary[merge_cols].copy()
        if rename_dict:
            temp_summary_renamed = temp_summary_renamed.rename(columns=rename_dict)
        
        # Perform the merge
        try:
            return pd.merge(summary, temp_summary_renamed, on=["Sample"], how="outer")
        except Exception as e:
            st.warning(f"Failed to merge {isotope_num} data: {str(e)}")
            return summary
    
    def _add_project_type_info(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Add project and type information to summary."""
        project_type_info = (
            self.sss.input_rep[["Type", "Project", "Sample"]]
            .groupby("Sample", as_index=False)
            .agg("first")
        )
        return pd.merge(summary, project_type_info, on=["Sample"])
    
    def _process_confidence_intervals(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Process confidence interval columns."""
        for mz in [47, 48, 49]:
            cl_col = f"95% CL_D{mz}"
            se_col = f"2SE_D{mz}"
            
            if cl_col in summary.columns:
                summary = summary.rename(columns={cl_col: se_col})
                summary[se_col] = summary[se_col].str.extract(r"(\d*\.\d*)")
        
        return self.apply_smart_numeric_conversion(summary)
    
    def _add_isotope_ratio_columns(self, summary: pd.DataFrame, 
                                  full_dataset: pd.DataFrame) -> pd.DataFrame:
        """Add isotope ratio columns to summary."""
        if summary.empty or full_dataset is None or full_dataset.empty:
            return summary
            
        # Add d18O columns if they exist
        if "d18O_VSMOW" in summary.columns:
            summary.insert(3, "d18O_CO2_VSMOW", summary["d18O_VSMOW"])
            summary.drop(columns=["d18O_VSMOW"], inplace=True)
            summary.insert(4, "d18O_CO2_VPDB", 
                          (summary["d18O_CO2_VSMOW"] * VPDB_FACTOR) - VPDB_OFFSET)
        
        # Add standard deviation columns
        sd_configs = [
            ("SD_d18O", "d18O_VPDB", 5),
            ("SD_d13C", "d13C_VPDB", 3)
        ]
        
        for sd_col, data_col, position in sd_configs:
            # Check if we can insert at the specified position
            insert_pos = min(position, len(summary.columns))
            summary.insert(insert_pos, sd_col, [np.nan] * len(summary))
            
            # Only calculate if the data column exists in full_dataset
            if data_col in full_dataset.columns:
                for sample in summary["Sample"]:
                    sample_data = full_dataset[full_dataset["Sample"] == sample]
                    if not sample_data.empty:
                        std_dev = sample_data[data_col].std()
                        summary.loc[summary["Sample"] == sample, sd_col] = std_dev
        
        return summary
    
    def _process_session_data(self, isotope_objects: List[Any], 
                             return_dict: Dict[str, Any]) -> None:
        """Process session data for each isotope type."""
        isotope_nums = ["47", "48", "49"]
        process_flags = [
            getattr(self.sss, f"process_D{num}", False) 
            for num in isotope_nums
        ]
        
        for obj, num, should_process in zip(isotope_objects, isotope_nums, process_flags):
            if not should_process or obj is None:
                continue
            
            # Extract session data
            sessions_data = obj.table_of_sessions(
                save_to_file=False, print_out=False, output="raw"
            )
            sessions_df = pd.DataFrame(sessions_data[1:], columns=sessions_data[0])
            #sessions_df = sessions_df.apply(pd.to_numeric, errors="coerce")
            
            # Extract repeatability data
            repeatability = obj.repeatability
            
            return_dict.update({
                f"sessions{num}": sessions_df,
                f"r{num}Anchors": repeatability[f"r_D{num}a"],
                f"r{num}Sample": repeatability[f"r_D{num}u"],
                f"r{num}All": repeatability[f"r_D{num}"],
            })
            
            # Clear the object from session state
            setattr(self.sss, f"D47c_{num}", None)


class ExcelExporter:
    """Handles Excel file creation and export functionality."""

    @staticmethod
    def create_excel_download(
        dataframe: pd.DataFrame,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create base64 encoded Excel file with a FAIR metadata sheet."""
        from tools.commons import build_fair_metadata

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            build_fair_metadata(extra=extra_meta).to_excel(
                writer, index=False, sheet_name="FAIR_metadata",
            )
            dataframe.to_excel(
                writer,
                index="Value" in dataframe.columns,
                header=True,
                sheet_name="data",
            )
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()

    @staticmethod
    def create_multi_sheet_excel(
        dataframes: Dict[str, pd.DataFrame],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create base64 encoded Excel file with a leading FAIR metadata sheet."""
        from tools.commons import build_fair_metadata

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            build_fair_metadata(extra=extra_meta).to_excel(
                writer, index=False, sheet_name="Metadata",
            )
            for sheet_name, df in dataframes.items():
                df.to_excel(
                    writer,
                    index="Value" in df.columns,
                    header=True,
                    sheet_name=sheet_name,
                )
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()


class ProcessingPage(BasePage):
    """Main class for the clumped isotope data processing page."""
    
    PAGE_NUMBER = 4
    PAGE_TITLE = "Processing"
    
    # Standard samples for filtering
    STANDARD_SAMPLES = [
        *[f"ETH-{i}" for i in (1, 2, 3, 4)], 
        "GU1", "Carrara", "25C", "1000C"
    ]
    
    # State parameters to track
    STATE_PARAMS = [
        "process_D47", "process_D48", "process_D49", "scale",
        "correction_method", "processing_sessions", "drifting_sessions", 'selected_calibrations'
    ]
    
    def __init__(self):
        """Initialize the processing page."""
        super().__init__()
        self._initialize_calibrations()
        
        # Initialize processors
        self.isotope_processor = IsotopeProcessor(self.sss)
        self.data_processor = DataProcessor(self.sss)
        self.temp_calculator = TemperatureCalculator()
        self.excel_exporter = ExcelExporter()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        self.sss["standards_nominal"] = IsotopeStandards.get_standards()
        self.sss["standards_bulk"] = IsotopeStandards.get_bulk()

        self.sss.temp_acid = DEFAULT_ACID_TEMPERATURE

        wg_ratios = DEFAULT_WG_RATIOS or {"d18O": 25.260, "d13C": -4.20}
        self.sss.d18O_wg = wg_ratios["d18O"]
        self.sss.d13C_wg = wg_ratios["d13C"]

        self.sss.working_gas = DEFAULT_WG_VIA_STANDARDS
        self.sss.working_gas_co2_stds = DEFAULT_CO2_STANDARDS

        if "params_last_run" not in self.sss:
            self.sss.params_last_run = {param: None for param in self.STATE_PARAMS}
        
        session_defaults = {
            "plots_D47crunch": [],
            "show_confirmation": False,
            "submitted_stds": False,
        }
        
        for key, default_value in session_defaults.items():
            if key not in self.sss:
                setattr(self.sss, key, default_value)
    
    def _initialize_calibrations(self) -> None:
        """Initialize temperature calibrations."""
        if "04_calibs" not in self.sss:
            try:
                import inspect
                import D47calib
                
                self.sss["04_calibs"] = {
                    "Fiebig24 (original)": "hardcoded",
                    "Fiebig21 (original)": "hardcoded",
                    "Anderson21 (original)": "hardcoded",
                    "Swart21 (original)": "hardcoded",
                }
                
                # Add D47calib calibrations
                self.sss["04_calibs"].update({
                    f"{name} (D47calib)": obj
                    for name, obj in inspect.getmembers(D47calib)
                    if isinstance(obj, D47calib.D47calib)
                })
            except ImportError:
                st.warning("D47calib module not available. Using built-in calibrations only.")
                self.sss["04_calibs"] = {
                    "Fiebig24 (original)": "hardcoded",
                    "Fiebig21 (original)": "hardcoded",
                    "Anderson21 (original)": "hardcoded",
                    "Swart21 (original)": "hardcoded",
                }
    

    
    
    def _validate_input_data(self) -> bool:
        """Validate that required input data is available."""
        if "input_rep" not in self.sss or len(self.sss.input_rep) == 0:
            st.markdown(
                r"Please upload δ⁴⁵-δ⁴⁹ replicate data to be processed "
                r"(:violet[Upload δ⁴⁵-δ⁴⁹ replicates] tab)."
            )
            st.page_link("pages/01_Data_IO.py", label=r"$\rightarrow  \textit{Data-IO}$  page")
            
            return False
        return True
    
    def _render_input_data_editor(self) -> None:
        """Render the input data editor for outlier removal."""
        with st.expander("Input replicates (temporarily rename/delete outliers here!)"):
            self.sss.input_rep = st.data_editor(
                self.sss.input_rep,
                num_rows="dynamic",
                key="outlier_inputReps",
            )
    
    def _render_sidebar_controls(self) -> ProcessingConfig:
        """Render sidebar controls and return processing configuration."""
        # Isotope selection checkboxes
        config = ProcessingConfig()
        
        # Session selection
        all_sessions = list(self.sss.input_rep["Session"].unique())
        config.processing_sessions = st.sidebar.multiselect(
            "Sessions to process:",
            all_sessions,
            default=all_sessions,
        )
        config.drifting_sessions = all_sessions
        
        wg_mode = "via standards" if self.sss.working_gas else "known composition"
        st.sidebar.caption(f"Working gas mode: **{wg_mode}** (change in Settings)")
        
        # Calibration selection
        default_calibs = (
            self.sss.get('04_selected_calibs', ["Fiebig24 (original)"])
        )
        if "params_last_run" in self.sss and "selected_calibrations" in self.sss.params_last_run:
            default_calibs = self.sss.params_last_run["selected_calibrations"]
        selected_calibs = st.sidebar.multiselect(
            "Choose calibration(s) for temperature estimates",
            list(self.sss["04_calibs"].keys()),
            key="04_selected_calibs",
            default='Fiebig24 (original)',
        )
        
        # Reference frame selection
        all_standards = list(self.sss["standards_nominal"].keys())
        last_std_idx = 0
        
        if "params_last_run" in self.sss and "scale" in self.sss.params_last_run and self.sss.params_last_run.get('scale', None):
            last_std_idx = all_standards.index(self.sss.params_last_run["scale"])
        else:
            try:
                last_std_idx = all_standards.index('CDES')
            except ValueError:
                last_std_idx = 0
        scale = st.sidebar.selectbox(
            "**Reference frame:**",
            all_standards,
            index=last_std_idx,
            key="scale",
        )
        
        # Display reference frame info
        if "#info" in self.sss.standards_nominal[scale]:
            st.sidebar.text(self.sss.standards_nominal[scale]["#info"])
        
        
        config.scale = scale
        config.selected_calibrations = selected_calibs
        
        mz_cols = st.sidebar.columns(3)
        isotope_types = [("47", "D47"), ("48", "D48"), ("49", "D49")]
        
        for i, (mz, isotope) in enumerate(isotope_types):
            if mz in self.sss.standards_nominal[scale]:
                with mz_cols[i]:
                    process_key = f"process_{isotope}"
                    value = st.checkbox(
                        rf"$\Delta_{{{mz}}}$",
                        value=True if "47" in process_key else self.sss.params_last_run.get(f"process_D{mz}", False),
                        key=process_key,
                    )
                    #
                    setattr(config, process_key, value) 
            else:
                setattr(config, f"process_{isotope}", False)
                if f"process_{isotope}" in self.sss:
                    del self.sss[f"process_{isotope}"]

        # Read-only display of anchors present in dataset for selected frame
        available_samples = set(self.sss.input_rep['Sample'].unique())
        all_anchor_names = set()
        for mz in ("47", "48", "49"):
            if mz in self.sss.standards_nominal[scale]:
                all_anchor_names.update(
                    k for k in self.sss.standards_nominal[scale][mz] if k in available_samples
                )
        if all_anchor_names:
            rows = []
            for std_name in sorted(all_anchor_names):
                row = {"Standard": std_name}
                for mz in ("47", "48", "49"):
                    if mz in self.sss.standards_nominal[scale] and std_name in self.sss.standards_nominal[scale][mz]:
                        row[f"Δ{mz}"] = self.sss.standards_nominal[scale][mz][std_name]
                rows.append(row)
            st.sidebar.caption("Anchors (in dataset)")
            st.sidebar.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

        all_defined_anchors = set()
        for mz in ("47", "48", "49"):
            if mz in self.sss.standards_nominal[scale]:
                all_defined_anchors.update(self.sss.standards_nominal[scale][mz].keys())
        missing_anchor_names = all_defined_anchors - available_samples
        if missing_anchor_names:
            rows_missing = []
            for std_name in sorted(missing_anchor_names):
                row = {"Standard": std_name}
                for mz in ("47", "48", "49"):
                    if mz in self.sss.standards_nominal[scale] and std_name in self.sss.standards_nominal[scale][mz]:
                        row[f"Δ{mz}"] = self.sss.standards_nominal[scale][mz][std_name]
                rows_missing.append(row)
            st.sidebar.caption("Anchors (not in dataset)")
            st.sidebar.dataframe(pd.DataFrame(rows_missing), hide_index=True, width='stretch')

        # Session treatment method
        method_radio = st.sidebar.radio(
            "Treat sessions:",
            ("Pooled",),  # Independent mode disabled
            help="Independent mode not available, see open issue for D47crunch: "
                 "https://github.com/mdaeron/D47crunch/issues/19",
            key="correction_method_radio",
        )
        
        config.correction_method = "pooled" if method_radio == "Pooled" else "indep_sessions"
        
     
        
        return config
    
    
    def _prepare_processing_data(self, config: ProcessingConfig) -> None:
        """Prepare data for processing."""
        # Filter data by selected sessions
        raw_data = self.sss.input_rep[
            self.sss.input_rep["Session"].isin(config.processing_sessions)
        ]
        
        # Rename datetime column if present
        if "datetime" in raw_data.columns:
            raw_data = raw_data.rename(columns={"datetime": "Timetag"})
        
        COLS = ['d45', 'd46', 'd47', 'd48', 'd49', 'Timetag', 'Sample', 'UID', 'Session', ]
        raw_data = raw_data[[_ for _ in COLS if _ in raw_data.columns]]
        
        # Sort by timestamp and prepare CSV
        raw_data = raw_data.sort_values(by="Timetag")
        self.sss.raw_data = raw_data
        self.sss.csv_text = raw_data.to_csv(sep=";")
        
        # Update session state with current config (only non-widget bound values)
        self.sss.correction_method = config.correction_method
        self.sss.processing_sessions = config.processing_sessions
        self.sss.drifting_sessions = config.processing_sessions
    
    def _calculate_longterm_error(self, summary_name: str) -> pd.DataFrame:
        """Calculate long-term error estimates."""
        summary = self.sss[f"correction_output_{summary_name}"]
        
        isotope_types = ["D47", "D48", "D49"]
        process_flags = [
            getattr(self.sss, f"process_{isotope}", False) 
            for isotope in isotope_types
        ]
        
        for isotope, should_process in zip(isotope_types, process_flags):
            if should_process:
                isotope_num = isotope[-2:]
                repeatability_key = f"correction_output_r{isotope_num}All"
                
                if repeatability_key in self.sss:
                    summary[f"{isotope} 2SE (longterm)"] = (
                        t.ppf(1 - 0.025, summary["N"].sum() - 1)
                        * self.sss[repeatability_key]
                        / np.sqrt(summary["N"])
                    )
        
        return summary



    def _create_processing_params_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with processing parameters."""

        IP = IsotopeProcessor
        
        
        #if self.sss.scale in self.sss.standards_nominal:
        STDS = {}
        for mz in '47', '48', '49':
            if self.sss.params_last_run.get(f"process_D{mz}", False):
                if self.sss.params_last_run.get('scale', None):
                    if mz in self.sss["standards_nominal"][self.sss.params_last_run['scale']]:
                        last_scale = self.sss.params_last_run['scale']
                        STDS[mz] = str({key: self.sss["standards_nominal"][last_scale][mz][key] for key in self.sss.standards_nominal[last_scale].get(mz, {}) if key in self.sss.standards['Sample'].values})
                        
                else:
                    STDS[mz] = 'N/A'
            else:
                STDS[mz] = 'N/A'
                
        params = {
            "Parameter": [
                "Processing sessions",
                "Reference frame", 
                "Correction method",
                "Standards D47",
                "Standards D48",
                "Standards D49",
                "D47 processed",
                "D48 processed", 
                "D49 processed",
                "D47 long-term repeatability (1sd)",
                "D48 long-term repeatability (1sd)",
                "D49 long-term repeatability (1sd)",
                "Selected calibrations",
                "Acid temperature",
                "Working gas d13C",
                "Working gas d18O",
                "Standards d13C",
                "Standards d18O",
            ],
            
            
            "Value": [
                ", ".join(str(s) for s in self.sss.params_last_run.get("processing_sessions", [])),
                self.sss.params_last_run.get("scale", ""),
                "Independent sessions" if "indep" in self.sss.params_last_run.get("correction_method", "") else "Pooled sessions",
                STDS['47'],
                STDS['48'],
                STDS['49'],
                "Yes" if self.sss.params_last_run.get("process_D47", False) else "No",
                "Yes" if self.sss.params_last_run.get("process_D48", False) else "No", 
                "Yes" if self.sss.params_last_run.get("process_D49", False) else "No",
                str(self.sss.get("correction_output_r47All", "N/A")) if self.sss.params_last_run.get("process_D47", False) else "N/A",
                str(self.sss.get("correction_output_r48All", "N/A")) if self.sss.params_last_run.get("process_D48", False) else "N/A",
                str(self.sss.get("correction_output_r49All", "N/A")) if self.sss.params_last_run.get("process_D49", False) else "N/A",
                ", ".join(self.sss.get("04_used_calibs", [])),
                str(self.sss.temp_acid),
                str(round(self.sss['d13Cwg_VPDB'],3)),
                str(round(self.sss['d18Owg_VSMOW'],3)),                
                str({key:self.sss["standards_bulk"][18][key] for key in self.sss["standards_bulk"][18] if key in self.sss.input_rep['Sample'].values}),
                str({key:self.sss["standards_bulk"][13][key] for key in self.sss["standards_bulk"][13] if key in self.sss.input_rep['Sample'].values}),                
            ]
        }
        

        if self.sss.get('scaling_factors', False):
            params['Parameter'].append('Scaling factors')
            params['Value'].append(str(self.sss.scaling_factors))


        return pd.DataFrame(params)

    
    
    def _display_processing_summary(self) -> None:
        """Display processing summary information."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Display processing parameters
            sessions_str = ", ".join(str(s) for s in self.sss.params_last_run["processing_sessions"])
            st.markdown(f"Processed sessions: {sessions_str}")
            st.markdown(f"Reference frame: {self.sss.params_last_run['scale']}")
            
            method_display = (
                "Independent sessions" if "indep" in self.sss.params_last_run["correction_method"]
                else "Pooled sessions"
            )
            st.markdown(f"Correction method: {method_display}")
            
            # Create download link for full results
            self._create_full_results_download()
        
        with col2:
            # Display repeatability metrics
            self._display_repeatability_metrics()
    
    def _create_full_results_download(self) -> None:
        """Create download link for full processing results."""
        dataframes = {
            "proc_params": self._create_processing_params_dataframe(),
            "summary": self.sss["correction_output_summary"],
            "replicates": self.sss["correction_output_full_dataset"],
        }
        
        # Add session-specific data
        for mz in ["47", "48", "49"]:
            process_key = f"process_D{mz}"
            session_key = f"correction_output_sessions{mz}"
            
            if (self.sss.params_last_run.get(process_key) and session_key in self.sss):
                dataframes[f"session{mz}"] = self.sss[session_key]
        
        # Generate filename
        sessions = self.sss.params_last_run["processing_sessions"]
        session_parts = []
        for session in sessions:
            if "-" in str(session):
                session_parts.extend(str(session).split("-"))
            else:
                session_parts.append(str(session))
        
        session_parts = sorted(set(session_parts))
        filename = (
            f"{session_parts[0]}-{session_parts[-1]}_{len(sessions)}sessions_"
            f"{self.sss.params_last_run['scale']}_{self.sss.params_last_run['correction_method']}.xlsx"
        )
        
        run_meta = {
            "Processing sessions": ", ".join(str(s) for s in sessions),
            "Reference frame": self.sss.params_last_run.get("scale", ""),
            "Correction method": self.sss.params_last_run.get("correction_method", ""),
            "Δ47 processed": self.sss.params_last_run.get("process_D47", False),
            "Δ48 processed": self.sss.params_last_run.get("process_D48", False),
            "Δ49 processed": self.sss.params_last_run.get("process_D49", False),
            "Calibrations used": ", ".join(self.sss.get("04_used_calibs", [])),
            "Acid temperature (run)": self.sss.get("temp_acid", ""),
            "Working gas δ¹³C (run)": round(self.sss.get("d13Cwg_VPDB", 0), 5),
            "Working gas δ¹⁸O (run)": round(self.sss.get("d18Owg_VSMOW", 0), 5),
        }
        if self.sss.get("scaling_factors"):
            run_meta["Scaling factors"] = self.sss.scaling_factors

        excel_data = self.excel_exporter.create_multi_sheet_excel(
            dataframes, extra_meta=run_meta,
        )
        download_link = (
            f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;'
            f'base64,{excel_data}" download="{filename}">📥 download full results!</a>'
        )
        st.markdown(download_link, unsafe_allow_html=True)
    
    def _display_repeatability_metrics(self) -> None:
        """Display repeatability metrics."""
        md_pieces = []
        
        isotope_configs = [
            ("D47", "process_D47", "correction_output_r47All"),
            ("D48", "process_D48", "correction_output_r48All"),
            ("D49", "process_D49", "correction_output_r49All"),
        ]
        
        for i, (isotope, process_key, repeatability_key) in enumerate(isotope_configs):
            if getattr(self.sss, process_key, False) and repeatability_key in self.sss:
                if i == 0:  # First isotope gets the header
                    md_pieces.extend([
                        "Long-term repeatability (1sd)", "<br>",
                        f'<font size="5">Δ<sub>{isotope[-2:]}</sub></font>',
                        "&nbsp;&nbsp;",
                        f'<font size="5">{round(self.sss[repeatability_key] * 1000, 2)} ppm</font>',
                        "<br>",
                    ])
                else:
                    precision = 2 if isotope != "D49" else 0
                    md_pieces.extend([
                        f'<font size="5">Δ<sub>{isotope[-2:]}</sub></font>',
                        "&nbsp;&nbsp;",
                        f'<font size="5">{round(self.sss[repeatability_key] * 1000, precision)} ppm</font>',
                        "<br>" if i < len(isotope_configs) - 1 else "",
                    ])
        
        if md_pieces:
            st.markdown("".join(md_pieces), unsafe_allow_html=True)
    
    def _display_results_expanders(self) -> None:
        """Display detailed results in expandable sections."""
        if "correction_output_summary" not in self.sss:
            return
        
        summary = self.sss["correction_output_summary"]
        
        # Summary section
        with st.expander("Summary"):
            excel_data = self.excel_exporter.create_excel_download(summary)
            download_link = (
                f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;'
                f'base64,{excel_data}" download="summary.xlsx">📥 download!</a>'
            )
            st.markdown(download_link, unsafe_allow_html=True)
            st.dataframe(summary.set_index("Sample"))
        
        # Processing parameters section
        with st.expander("Processing parameters"):
            params_df = self._create_processing_params_dataframe()
            excel_data = self.excel_exporter.create_excel_download(params_df)
            download_link = (
                f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;'
                f'base64,{excel_data}" download="proc_params.xlsx">📥 download!</a>'
            )
            st.markdown(download_link, unsafe_allow_html=True)
            st.dataframe(params_df)
        
        # Session parameters sections
        self._display_session_parameters()
        
        # Full dataset section
        if "correction_output_full_dataset" in self.sss:
            with st.expander("All replicates"):
                full_data = self.sss["correction_output_full_dataset"]
                excel_data = self.excel_exporter.create_excel_download(full_data)
                download_link = (
                    f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;'
                    f'base64,{excel_data}" download="replicates.xlsx">📥 download!</a>'
                )
                st.markdown(download_link, unsafe_allow_html=True)
                st.dataframe(full_data.set_index("Sample"))
    
    def _display_session_parameters(self) -> None:
        """Display session parameters for each processed isotope."""
        isotope_configs = [
            ("47", "process_D47", "correction_output_sessions47", "47params.xlsx"),
            ("48", "process_D48", "correction_output_sessions48", "48params.xlsx"),
            ("49", "process_D49", "correction_output_sessions49", "49params.xlsx"),
        ]
        
        for mz, process_key, session_key, filename in isotope_configs:
            if (self.sss.params_last_run.get(process_key) and session_key in self.sss):
                session_data = self.sss[session_key]
                
                with st.expander(rf"$\Delta_{{{mz}}}$ session parameters"):
                    excel_data = self.excel_exporter.create_excel_download(session_data)
                    download_link = (
                        f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;'
                        f'base64,{excel_data}" download="{filename}">📥 download!</a>'
                    )
                    st.markdown(download_link, unsafe_allow_html=True)
                    st.dataframe(session_data.set_index("Session"))
    
    def _run_processing(self, config: ProcessingConfig) -> None:
        """Execute the main processing workflow."""
        # Validate configuration
        if config.scale in ("ICDES", "compile") and config.process_D49:
            st.warning(
                "There are no I-CDES standard values for Δ49! "
                "Please choose CDES if you want to process Δ49 data using gas standards."
            )
            return
        
        with st.spinner("Processing in progress..."):
            # Prepare data
            self._prepare_processing_data(config)
            
            # Process isotopes
            self.isotope_processor.process_all_isotopes(config)
            
            # Merge datasets
            self.data_processor.merge_datasets()
            
            # Clean up CSV text
            self.sss.csv_text = None
        
        # Calculate long-term errors
        summary = self._calculate_longterm_error("summary")
        
        # Calculate temperatures
        summary = self.temp_calculator.calc_temp(summary)
        
        # Store final summary
        self.sss["correction_output_summary"] = summary
        
        # Extract standards data
        standards_mask = self.sss["correction_output_full_dataset"]["Sample"].isin(
            self.STANDARD_SAMPLES
        )
        self.sss.standards = self.sss["correction_output_full_dataset"][standards_mask]
        
        # Update parameters tracking
        for param in self.STATE_PARAMS:
            if hasattr(config, param):
                self.sss.params_last_run[param] = getattr(config, param)
        
        # Reset run button state
        self.sss.show_run_button = False
    
    def run(self) -> None:
        """Main function to run the processing page."""
        # Validate input data
        if not self._validate_input_data():
            st.stop()

                    
        # Main processing button
        run_button = st.sidebar.button("Run...", key="BUTTON1")
        
        # Render input data editor
        self._render_input_data_editor()
        
        # Render sidebar controls and get configuration
        config = self._render_sidebar_controls()

        
        # Handle processing execution
        if run_button:
            if not(
            self.sss.get("process_D47", False) or 
            self.sss.get("process_D48", False) or 
            self.sss.get("process_D49", False)
            ):
                st.markdown(
                    r"## Please choose at least one of $\Delta_{47}$, $\Delta_{48}$, or $\Delta_{49}$!"                    
                )
                st.stop()
            self._run_processing(config)
            self._display_processing_summary()
            self._display_results_expanders()
        else:
            # Display existing results if available
            if "correction_output_summary" in self.sss:
                self._display_processing_summary()
                self._display_results_expanders()
            else:
                st.markdown(
                    "Please choose processing parameters on the sidebar and "
                    "click the :violet[Run...] button to process the dataset."
                )


def RUN():
    page = ProcessingPage()
    page.run()
# Main execution
if __name__ == "__main__":
    RUN()
