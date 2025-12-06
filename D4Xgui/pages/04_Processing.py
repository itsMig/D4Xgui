#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from doctest import DocFileCase
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import base64
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

import D47crunch as D47c
from scipy.stats import t
from streamlit_extras.stylable_container import stylable_container

from tools.page_config import PageConfigManager
from tools.sidebar_logo import SidebarLogoManager
from tools.authenticator import Authenticator
from tools.init_params import IsotopeStandards
from tools.commons import clear_session_cache
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
        
        # Set acid reaction constant
        # d47crunch_obj.ALPHA_18O_ACID_REACTION = np.exp(
        #     3.59 / (self.ACID_TEMPERATURE + 273.15) - 1.79e-3
        # )
        if self.sss.get('working_gas_co2_stds', False):
            d47crunch_obj.ALPHA_18O_ACID_REACTION = 1
        else:
            d47crunch_obj.ALPHA_18O_ACID_REACTION = np.exp(
                3.59 / (self.sss.temp_acid + 273.15) - 1.79e-3
            )
        
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
            #d47crunch_obj.Nominal_d18O_VPDB = self.sss.d18O_wg
            self.sss['d18Owg_VSMOW'] = self.sss.d18O_wg
            #d47crunch_obj.Nominal_d13C_VPDB = self.sss.d13C_wg
            self.sss['d13Cwg_VPDB'] = self.sss.d13C_wg
            for record in d47crunch_obj:
                record["d13Cwg_VPDB"] = self.sss.d13C_wg
                record["d18Owg_VSMOW"] = self.sss.d18O_wg
                
            for s in d47crunch_obj.sessions:
                d47crunch_obj.sessions[s]['d13C_standardization_method'] = False
                d47crunch_obj.sessions[s]["d18O_standardization_method"] = False
                
            # d47crunch_obj.wg()
        

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
            #st.toast(f"Processing Δ₄₇ data...")
        
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


class TemperatureCalculator:
    """Handles temperature calculations from D47 values using various calibrations."""
    POLY_63_COEFFS = (-5.896755e00, -3.520888e03, 2.391274e07, -3.540693e09)
    POLY_64_COEFFS = (6.001624e00, -1.298978e04, 8.995634e06, -7.422972e08)
    POLY_65_COEFFS = (-6.741e00, -1.950e04, 5.845e07, -8.093e09)
    #  Scaling and offset parameters for Fiebig et al. (2021)
    FIEBIG2021_D47_SCALING = 1.0381881
    FIEBIG2021_D47_OFFSET = 0.1855537
    FIEBIG2021_D48_SCALING = 1.0280693
    FIEBIG2021_D48_OFFSET = 0.1244564
    # Scaling and offset parameters for Fiebig et al. (2024)
    FIEBIG2024_D47_SCALING = 1.038
    FIEBIG2024_D47_OFFSET = 0.1848
    FIEBIG2024_D48_SCALING = 1.038
    FIEBIG2024_D48_OFFSET = 0.1214
    
    def __init__(self, session_state: st.session_state = st.session_state):
        self.sss = session_state
        # Polynomial coefficients for Hill et al. (2014)
      
    
    @staticmethod
    def _evaluate_polynomial_4th_order(coeffs: Tuple[float, ...], x: Union[float, np.ndarray]) -> Union[
        float, np.ndarray]:
        """Evaluate a 4th-degree polynomial.

        Args:
            coeffs: Polynomial coefficients (a, b, c, d) for ax + bx² + cx³ + dx⁴.
            x: Input value(s).

        Returns:
            Polynomial evaluation result.
        """
        a, b, c, d = coeffs
        return a * x + b * x ** 2 + c * x ** 3 + d * x ** 4
    
    @classmethod
    def calculate_d63_hill2014(cls, inverse_temp_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate D63 values using Hill et al. (2014) calibration.

        Args:
            inverse_temp_k: 1/T in Kelvin.

        Returns:
            D63 values.
        """
        return cls._evaluate_polynomial_4th_order(cls.POLY_63_COEFFS, inverse_temp_k)
    
    @classmethod
    def calculate_d64_hill2014(cls, inverse_temp_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate D64 values using Hill et al. (2014) calibration.

        Args:
            inverse_temp_k: 1/T in Kelvin.

        Returns:
            D64 values.
        """
        return cls._evaluate_polynomial_4th_order(cls.POLY_64_COEFFS, inverse_temp_k)
    
    @classmethod
    def calculate_d65_hill2014(cls, inverse_temp_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate D65 values using Hill et al. (2014) calibration.

        Args:
            inverse_temp_k: 1/T in Kelvin.

        Returns:
            D65 values.
        """
        return cls._evaluate_polynomial_4th_order(cls.POLY_65_COEFFS, inverse_temp_k)
    
    @classmethod
    def calculate_d47_fiebig2021(cls, inverse_temp_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate D47 values using Fiebig et al. (2021) calibration.

        Args:
            inverse_temp_k: 1/T in Kelvin.

        Returns:
            D47 values.
        """
        d63 = cls.calculate_d63_hill2014(inverse_temp_k)
        return (d63 * cls.FIEBIG2021_D47_SCALING) + cls.FIEBIG2021_D47_OFFSET
    
    def calculate_d47_fiebig2024(cls, inverse_temp_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate D47 values using Fiebig et al. (2024) calibration.

		Args:
			inverse_temp_k: 1/T in Kelvin.

		Returns:
			D47 values.
		"""
        d63 = cls.calculate_d63_hill2014(inverse_temp_k)
        return (d63 * cls.FIEBIG2024_D47_SCALING) + cls.FIEBIG2024_D47_OFFSET
    
    @classmethod
    def get_temperature_difference_d47_fiebig2021(cls, temp_celsius: float, target_d47: float) -> float:
        """Calculate absolute difference between target D47 and calculated D47 (Fiebig 2021).

		Args:
			temp_celsius: Temperature in Celsius.
			target_d47: Target D47 value.

		Returns:
			Absolute difference between target and calculated D47.
		"""
        inverse_temp_k = 1 / (temp_celsius + 273.15)
        calculated_d47 = cls.calculate_d47_fiebig2021(inverse_temp_k)
        return abs(target_d47 - calculated_d47)
    
    def get_temperature_difference_d47_fiebig2024(cls, temp_celsius: float, target_d47: float) -> float:
        """Calculate absolute difference between target D47 and calculated D47 (Fiebig 2024).

		Args:
			temp_celsius: Temperature in Celsius.
			target_d47: Target D47 value.

		Returns:
			Absolute difference between target and calculated D47.
		"""
        inverse_temp_k = 1 / (temp_celsius + 273.15)
        calculated_d47 = cls.calculate_d47_fiebig2024(inverse_temp_k)
        return abs(target_d47 - calculated_d47)
    
    def get_temperature_difference_d47_anderson2021(cls, temp_celsius: float, target_d47: float) -> float:
        return abs(target_d47 - ((0.0391 * 1e6 / temp_celsius ** 2) + 0.154))
    
    def _direct_temperature_swart2021(cls, D47):
        """
        Calculate temperature using the Swart21 equation: Δ47(CDES90) = 0.039 * 10^6/T^2 + 0.158

        :param D47: The D47 value
        :return: Temperature in °C
        """
        # Rearranged equation: T = sqrt(10^6 * 0.039 / (D47 - 0.158))
        # Convert from Kelvin to Celsius by subtracting 273.15
        try:
            temp_K = np.sqrt(1e6 * 0.039 / (D47 - 0.158))
            return temp_K - 273.15
        except Exception as e:
            # Handle cases where D47 <= 0.158 which would result in negative or zero denominator
            return e
    
    def calc_temp(self, summary):
        calibs = self.sss["04_selected_calibs"]
        
        self.sss['04_used_calibs'] = [_ for _ in calibs]
        

        _ = """
        if not "04_calibs" in self.sss:
        import inspect
        import D47calib

        self.sss["04_calibs"] = {
            name: obj
            for name, obj in inspect.getmembers(D47calib)
            if isinstance(obj, D47calib.D47calib)
        }

        """
        
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
                                - 273.15 + so.minimize_scalar(
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
                    round(self._direct_temperature_swart2021(t), 2)
                    for t in (summary["D47"] + (summary[key]) * sign)
                ]

        for calib in calibs:
            if calib in [
                "Fiebig24 (original)",
                "Swart21 (original)",
                "Anderson21 (original)"
            ]:
                continue
            # summary[f"T(mean), {calib}"] = [None] * len(summary)
            for idx in range(len(summary)):
                # st.write(idx)
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
    
    @staticmethod
    def calculate_swart21_temperature(d47_value: float) -> float:
        """Calculate temperature using Swart21 calibration."""
        try:
            temp_k = np.sqrt(1e6 * 0.039 / (d47_value - 0.158))
            return temp_k - 273.15
        except (ValueError, ZeroDivisionError):
            return float('nan')
    
    @staticmethod
    def calculate_temperature_range(d47_values: pd.Series, se_values: pd.Series, 
                                  calibration_func, calibration_name: str) -> Dict[str, List[float]]:
        """Calculate temperature ranges for a given calibration."""
        results = {}
        
        # Define error ranges
        error_ranges = {
            "T(min, 2SE)": d47_values + 2 * se_values,
            "T(min, 1SE)": d47_values + se_values,
            "T(mean)": d47_values,
            "T(max, 1SE)": d47_values - se_values,
            "T(max, 2SE)": d47_values - 2 * se_values,
        }
        
        for temp_type, d47_range in error_ranges.items():
            column_name = f"{temp_type}, {calibration_name}"
            if calibration_name in ["Fiebig24 (original)", "Anderson21 (original)"]:
                results[column_name] = [
                    round(calibration_func(d47), 2) for d47 in d47_range
                ]
            else:
                results[column_name] = [
                    round(calibration_func(d47), 2) for d47 in d47_range
                ]
        
        return results
    
    def _add_fiebig21_temperatures(self, summary: pd.DataFrame, d47_values: pd.Series, se_values: pd.Series,
                                   _2se_values: pd.Series) -> None:
        """Add Fiebig21 temperature calculations to summary."""
        try:
            from scipy.optimize import minimize_scalar
            
            
            # Define error ranges for temperature calculation
            error_ranges = {
                "T(min, 2SE)": d47_values + _2se_values,
                "T(min, 1SE)": d47_values + se_values,
                "T(mean)": d47_values,
                "T(max, 1SE)": d47_values - se_values,
                "T(max, 2SE)": d47_values - _2se_values,
            }
            
            # Calculate temperatures for each error range
            for temp_type, d47_range in error_ranges.items():
                temperatures = []
                
                for d47 in d47_range:
                    if pd.notna(d47):
                        try:
                            # Calculate temperature using optimization
                            result = minimize_scalar(
                                TemperatureCalculator.get_temperature_difference_d47_fiebig2021,
                                args=(d47,),
                                # bounds=(-100, 1000),
                                # method='bounded'
                            )
                            temp = result.x
                            temperatures.append(round(temp, 2))
                        except Exception as e:
                            temperatures.append(e)
                    else:
                        temperatures.append(np.nan)
                
                summary[f"{temp_type}, Fiebig21 (original)"] = temperatures
        
        except Exception as e:
            summary[f"T (°C), Fiebig21 (original)"] = f"Error: {str(e)}"
    
    def _add_fiebig24_temperatures(self, summary: pd.DataFrame, d47_values: pd.Series, se_values: pd.Series,
                                   _2se_values: pd.Series) -> None:
        """Add Fiebig24 temperature calculations to summary."""
        try:
            from scipy.optimize import minimize_scalar
            #from tools.calc_temperature import TemperatureCalculator
            st.write(d47_values, se_values, _2se_values)
            # Define error ranges for temperature calculation
            error_ranges = {
                "T(min, 2SE)": d47_values + _2se_values,
                "T(min, 1SE)": d47_values + se_values,
                "T(mean)": d47_values,
                "T(max, 1SE)": d47_values - se_values,
                "T(max, 2SE)": d47_values - _2se_values,
            }
            
            # Calculate temperatures for each error range
            for temp_type, d47_range in error_ranges.items():
                temperatures = []
                
                for d47 in d47_range:
                    if pd.notna(d47):
                        try:
                            # Calculate temperature using optimization
                            result = minimize_scalar(
                                TemperatureCalculator.get_temperature_difference_d47_fiebig2024,
                                args=(d47,),
                                # bounds=(0, 1000),
                                # method='bounded'
                            )
                            temp = result.x
                            temperatures.append(round(temp, 2))
                        except Exception as e:
                            temperatures.append(e)
                    else:
                        temperatures.append(np.nan)
                
                summary[f"{temp_type}, Fiebig24 (original)"] = temperatures
        
        except Exception as e:
            summary[f"T (°C), Fiebig24 (original)"] = f"Error: {str(e)}"
    
    def _add_anderson21_temperatures(self, summary: pd.DataFrame, d47_values: pd.Series, se_values: pd.Series,
                                     _2se_values: pd.Series) -> None:
        """Add Anderson21 temperature calculations to summary."""
        try:
            # Anderson21 calibration: T = 0.0449 * 10^6 / D47^2 - 273.15
            
            # Define error ranges for temperature calculation
            error_ranges = {
                "T(min, 2SE)": d47_values + _2se_values,
                "T(min, 1SE)": d47_values + se_values,
                "T(mean)": d47_values,
                "T(max, 1SE)": d47_values - se_values,
                "T(max, 2SE)": d47_values - _2se_values,
            }
            
            # Calculate temperatures for each error range
            for temp_type, d47_range in error_ranges.items():
                temperatures = []
                
                for d47 in d47_range:
                    if pd.notna(d47) and d47 > 0:
                        try:
                            temp = np.sqrt((0.0391 * 1e6) / (d47 - 0.154)) - 273.15
                            temperatures.append(round(temp, 2))
                        except Exception:
                            temperatures.append(np.nan)
                    else:
                        temperatures.append(np.nan)
                
                summary[f"{temp_type}, Anderson21 (original)"] = temperatures
        
        except Exception as e:
            summary[f"T (°C), Anderson21 (original)"] = f"Error: {str(e)}"
    
    def _add_swart21_temperatures(self, summary: pd.DataFrame, d47_values: pd.Series, se_values: pd.Series,
                                  _2se_values: pd.Series) -> None:
        """Add Swart21 temperature calculations to summary using the Swart21 (2021) CDES90 calibration."""
        try:
            error_ranges = {
                "T(min, 2SE)": d47_values + _2se_values,
                "T(min, 1SE)": d47_values + se_values,
                "T(mean)": d47_values,
                "T(max, 1SE)": d47_values - se_values,
                "T(max, 2SE)": d47_values - _2se_values,
            }
            
            coef = 1e6 * 0.039  # from
            
            # Calculate temperatures for each error range
            for temp_type, d47_range in error_ranges.items():
                temperatures = []
                
                for d47 in d47_range:
                    denom = d47 - 0.158
                    if denom == 0:
                        temperatures.append(np.nan)
                        continue
                    
                    try:
                        temp_K = np.sqrt(coef / denom)
                        temp_C = temp_K - 273.15
                        # Only round/append if finite; otherwise append NaN
                        temperatures.append(round(float(temp_C), 2) if np.isfinite(temp_C) else np.nan)
                    except (ValueError, ZeroDivisionError, FloatingPointError, TypeError):
                        temperatures.append(np.nan)
                
                summary[f"{temp_type}, Swart21 (original)"] = temperatures
        
        except Exception as e:
            # In case of an unexpected error, record it in the summary
            summary[f"T (°C), Swart21 (original)"] = f"Error: {str(e)}"
    
    def _add_d47calib_temperatures(self, summary: pd.DataFrame, d47_values: pd.Series, se_values: pd.Series,
                                   _2se_values: pd.Series, calib_name: str) -> None:
        """Add D47calib temperature calculations to summary."""
        try:
            # Use the D47calib library for other calibrations
            import D47calib
            
            # Define error ranges for temperature calculation
            error_ranges = {
                "T(min, 2SE)": d47_values + _2se_values,
                "T(min, 1SE)": d47_values + se_values,
                "T(mean)": d47_values,
                "T(max, 1SE)": d47_values - se_values,
                "T(max, 2SE)": d47_values - _2se_values,
            }
            
            # Calculate temperatures for each error range
            for temp_type, d47_range in error_ranges.items():
                temperatures = []
                
                for d47 in d47_range:
                    if pd.notna(d47):
                        try:
                            # Calculate temperature using D47calib with the calibration object
                            temp = self.sss["04_calibs"][calib_name].__dict__["_T_from_D47"](d47)
                            # temp = D47calib.temperature(d47, calib_obj)
                            temperatures.append(round(temp, 2))
                        except Exception as e:
                            # Log the specific error for debugging
                            temperatures.append(e)
                    else:
                        temperatures.append(np.nan)
                
                summary[f"{temp_type}, {calib_name}"] = temperatures
        
        except Exception as e:
            summary[f"T (°C), {calib_name}"] = f"Error: {str(e)}"
    
    def _calculate_temperatures(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Calculate temperatures using selected calibrations."""
        calibrations = self.sss.get("04_selected_calibs", [])
        self.sss['04_used_calibs'] = list(calibrations)
        
        TC = TemperatureCalculator(self.sss)
        TC._calc_temp(summary)
        if "D47" not in summary.columns:
            return summary
        
        d47_values = summary["D47"]
        se_values = summary[
            "SE_D47"]  # [float(_) for _ in summary["SE_D47"]]# if len(str(_)) != 0]#isinstance(float(_), float)]
        _2se_values = summary[
            "2SE_D47"]  # [float(_) for _ in summary["2SE_D47"]]# if len(str(_)) != 0]  # isinstance(float(_), float)]
        st.write(d47_values.dtypes)
        # se_values = summary.get("SE_D47", pd.Series([0] * len(summary)))
        # _2se_values = summary.get("2SE_D47", pd.Series([0] * len(summary)))
        # se_values[np.isnan(se_values)] = 0
        # Process each calibration
        for calib_name in calibrations:
            if calib_name == "Fiebig24 (original)":
                self._add_fiebig24_temperatures(summary, d47_values, se_values, _2se_values)
            elif calib_name == "Fiebig21 (original)":
                self._add_fiebig21_temperatures(summary, d47_values, se_values, _2se_values)
            elif calib_name == "Anderson21 (original)":
                self._add_anderson21_temperatures(summary, d47_values, se_values, _2se_values)
            elif calib_name == "Swart21 (original)":
                self._add_swart21_temperatures(summary, d47_values, se_values, _2se_values)
            else:
                self._add_d47calib_temperatures(summary, d47_values, se_values, _2se_values, calib_name)
        
        return summary


class DataProcessor:
    """Handles data merging and processing operations."""
    
    def __init__(self, session_state: st.session_state):
        self.sss = session_state
    
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
        # if numeric_count == len(series):
        #     return pd.to_numeric(series)#, errors='coerce')
        # else:
        #     # Keep as is (string or mixed type)
        #     return series
    
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
                (dataset["d18O_VSMOW"] * 0.97001) - 29.99
            )
        
        return dataset
    
    def _process_timestamps(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize timestamp formats."""
        if np.issubdtype(dataset["Timetag"].dtype, np.datetime64):
            return dataset
        
        def clean_timestamp(timestamp_str):
            """Clean timestamp string for parsing."""
            return "".join(
                char for char in str(timestamp_str)
                if not char.isalpha() or char.isspace() or char in (":", "-", ".", "T")
            ).rstrip(" ")
        
        try:
            dataset["Timetag"] = [
                datetime.fromisoformat(clean_timestamp(ts))
                for ts in dataset["Timetag"]
            ]
        except Exception:
            # Fallback processing without 'T'
            dataset["Timetag"] = [
                datetime.fromisoformat(
                    clean_timestamp(ts).replace("T", " ")
                )
                for ts in dataset["Timetag"]
            ]
        
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
                # st.write(temp_summary[col])
                # temp_summary[col] = temp_summary[col].astype(float)
                # st.write(temp_summary[col])
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
                          (summary["d18O_CO2_VSMOW"] * 0.97001) - 29.99)
        
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
    def create_excel_download(dataframe: pd.DataFrame) -> str:
        """Create base64 encoded Excel file from DataFrame."""
        buffer = io.BytesIO()
        dataframe.to_excel(
            buffer,
            index="Value" in dataframe.columns,
            header=True,
        )
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
    
    @staticmethod
    def create_multi_sheet_excel(dataframes: Dict[str, pd.DataFrame]) -> str:
        """Create base64 encoded Excel file with multiple sheets."""
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(
                    writer,
                    index="Value" in df.columns,
                    header=True,
                    sheet_name=sheet_name,
                )
        
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()


class ProcessingPage:
    """Main class for the clumped isotope data processing page."""
    
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
        self.sss = st.session_state
        self._setup_page()
        self._initialize_session_state()
        self._initialize_calibrations()
        
        # Initialize processors
        self.isotope_processor = IsotopeProcessor(self.sss)
        self.data_processor = DataProcessor(self.sss)
        self.temp_calculator = TemperatureCalculator()
        self.excel_exporter = ExcelExporter()
    
    def _setup_page(self) -> None:
        """Set up page configuration and authentication."""
        PageConfigManager().configure_page(4)
        SidebarLogoManager().add_logo()
        
        st.title("Processing")
        
        # Require authentication (skip during testing)
        if "PYTEST_CURRENT_TEST" not in os.environ:
            if not Authenticator().require_authentication():
                st.stop()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        # Initialize standards if not present
        if "standards_nominal" not in self.sss or "standards_bulk" not in self.sss:
            self._load_initial_standards()

        
        # Initialize parameters tracking
        if "params_last_run" not in self.sss:
            self.sss.params_last_run = {param: None for param in self.STATE_PARAMS}
        
        # Initialize other session state variables
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
    

    def _load_initial_standards(self, force=False) -> None:
        """Load initial isotope standards and bulk values."""
        self.sss["standards_nominal"] = IsotopeStandards.get_standards()
        
        # 🔑 KEY: Only load if not present - preserves user edits
        if 'standards_bulk' not in self.sss or force==True:
            self.sss['standards_bulk'] = IsotopeStandards.get_bulk()
        
        
    
        self.sss.show_confirmation = False
        
        # Clear bulk isotope text area caches (δ13C, δ18O)
        for bulk in ["13", "18"]:
            key = f"update_{bulk}_text"
            if key in self.sss:
                del self.sss[key]

        # Clear any cached edited text areas for standards
        for mz in ["47", "48", "49"]:
            key = f"update_{mz}_text"
            if key in self.sss:
                del self.sss[key]
        
        # Clear last updated standards key to avoid stale state
        if "last_updated_stds" in self.sss:
            del self.sss["last_updated_stds"]
        
        # Clear any JSON error messages
        if "error_json" in self.sss:
            del self.sss["error_json"]

        st.rerun()
    
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
    
    def _off_on(self):
        pass
        #self.sss.process_D47 = None#.set(False)
    
    
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
        
        # Acid temperature
        st.sidebar.number_input("Acid temperature (°C)", key="temp_acid", value=90)
        
        cols_stds = st.sidebar.columns(2)
        with cols_stds[0]:
            st.toggle("Working gas via standards", key="working_gas", value=True, help='Bulk isotopic standardization is either calculated via 1pt ETF or multi-point ETF, when calculated via standards. Otherwise, you can enter the known working gas ocmposition.')
            
        if self.sss.working_gas:
            with cols_stds[1]:
                st.checkbox("CO$_{2}$ standards", key="working_gas_co2_stds",
                            help='Activating this option will allow users to work solely with CO2 gases, by setting the acid fractionation factor to 1.')
        else:
            # Initialize session state variables if they don't exist
            if 'd18O_wg' not in self.sss:
                self.sss['d18O_wg'] = 0.0  # or your default value
            if 'd13C_wg' not in self.sss:
                self.sss['d13C_wg'] = 0.0
            
            # Use in your inputs:
            cols_wg = st.sidebar.columns(2)
            with cols_wg[0]:
                self.sss['d18O_wg'] = st.number_input(
                    '$\delta^{18}O$ VPDB',
                    value=self.sss['d18O_wg']
                )
            with cols_wg[1]:
                self.sss['d13C_wg'] = st.number_input(
                    '$\delta^{13}C$ VPDB',
                    value=self.sss['d13C_wg']
                )
            
            # cols_wg = st.sidebar.columns(2)
            # with cols_wg[0]:
            #     st.number_input('$\delta^{18}O$ VPDB', key='d18O_wg', value=self.sss.get('d18O_wg', 0.0))
            # with cols_wg[1]:
            #     st.number_input('$\delta^{13}C$ VPDB', key='d13C_wg', value=self.sss.get('d13C_wg', 0.0))
        
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
            except:
                last_std_idx =  0
        # if "last_updated_stds" in self.sss:
        #     try:
        #         last_std_idx = all_standards.index(self.sss["last_updated_stds"])
        #     except ValueError:
        #         last_std_idx = 0
        # elif "params_last_run" in self.sss and "scale" in self.sss.params_last_run:
        #     last_std_idx = all_standards.index(self.sss.params_last_run["scale"])
        #
        scale = st.sidebar.selectbox(
            "**Reference frame:**",
            all_standards,
            index=last_std_idx,
            key="scale",
            on_change=self._off_on(),
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
                setattr(config, f"process_{isotope.lower()}", False)
        


        # Standards editing
        self._render_standards_editor(scale)
        
        if self.sss.working_gas: #via standards
            self._render_standards_editor_bulk()
        
        # Reset standards button
        self._render_reset_standards_button()
        
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
    
    def _render_standards_editor(self, scale: str) -> None:
        """Render standards editing interface."""
        for mz in ["47", "48", "49"]:
            process_key = f"process_D{mz}"
            
            # Check if this isotope's checkbox should be shown
            if mz in self.sss.standards_nominal[scale] and self.sss.get(f"process_D{mz}", False):
                with st.sidebar.expander(rf"$\Delta_{{{mz}}}$ anchors"):
                    current_standards = self.sss.standards_nominal[scale][mz]
                    
                    # Use a key that includes the scale to make it responsive
                    text_key = f"update_{mz}_text_{scale}"  # ← Include scale in key
                    
                    updated_text = st.text_area(
                        f"anchors {mz}",
                        json.dumps(current_standards, indent=4),
                        key=text_key,
                        label_visibility="collapsed",
                    )
                    
                    if st.button(rf"Update $\Delta_{{{mz}}}$ values!", key=f"update_{mz}_{scale}"):
                        self._update_standards_dict(mz, updated_text, scale)
    
    def _render_standards_editor_bulk(self) -> None:
        """Render bulk isotope standards editor with persistent state."""
        for bulk in [13, 18]:
            key_str = rf"$\delta^{{{bulk}}}${'O' if bulk == 18 else 'C'}"
            
            with st.sidebar.expander(f"{key_str} anchors", expanded=False):
                # 🔑 KEY: Always read current values from session state
                current_standards = self.sss['standards_bulk'][bulk]
                
                # Show summary
                num_anchors = len(current_standards)
                st.caption(f"Current: {num_anchors} anchor{'s' if num_anchors != 1 else ''}")
                
                # Text area with current values
                updated_text = st.text_area(
                    f"Edit {key_str} anchors (JSON format)",
                    value=json.dumps(current_standards, indent=4),
                    key=f"update_{bulk}_text",
                    height=150,
                    label_visibility="collapsed",
                    help=f"Edit {key_str} anchor values. Format: {{\"ETH-1\": -2.19}}"
                )
                
                if st.button(f"Update {key_str} values", key=f"update_{bulk}", type="primary"):
                    self._update_standards_bulk_dict(str(bulk), updated_text)
        
        # Display JSON errors if any
        if "error_json" in self.sss:
            for error, formatted_json in self.sss["error_json"]:
                st.sidebar.error(f"Invalid JSON: {error}", icon="🚨")
                st.sidebar.code(formatted_json)
    
    
    def _render_standards_editor_old(self, scale: str) -> None:
        """Render standards editing interface."""
        for mz in ["47", "48", "49"]:
            process_key = f"process_D{mz}"
            if (getattr(self.sss, process_key, False) and 
                mz in self.sss.standards_nominal[scale]):
                
                with st.sidebar.expander(rf"$\Delta_{{{mz}}}$ anchors"):
                    current_standards = self.sss.standards_nominal[scale][mz]
                    
                    updated_text = st.text_area(
                        f"anchors {mz}",
                        json.dumps(current_standards, indent=4),
                        key=f"update_{mz}_text",
                        label_visibility="collapsed",
                    )
                    
                    if st.button(rf"Update $\Delta_{{{mz}}}$ values!", key=f"update_{mz}"):
                        self._update_standards_dict(mz, updated_text, scale)
        
        for bulk in [13, 18]:
            # Check if the isotope anchors exist in the standards dictionary for the scale
            #if iso in self.sss.standards_nominal[scale]:
            key_str = rf"$\delta^{{{bulk}}}${'O' if bulk == 18 else 'C'}"
            with st.sidebar.expander(f"{key_str} anchors"):
                current_standards = self.sss.standards_bulk[bulk]
                
                updated_text = st.text_area(
                    f"anchors {bulk}",
                    json.dumps(current_standards, indent=4),
                    key=f"update_{bulk}_text",
                    label_visibility="collapsed",
                )
                
                if st.button(f"Update {key_str} values!", key=f"update_{bulk}"):
                    self._update_standards_dict(bulk, updated_text, scale)


        # Display JSON errors if any
        if "error_json" in self.sss:
            for error, formatted_json in self.sss["error_json"]:
                st.error(f"Invalid JSON format: {error}", icon="🚨")
                st.code(formatted_json)
    
    def _update_standards_bulk_dict(self, param: str, json_text: str) -> None:
        """Update standards_bulk dictionary with new values and persist in session state."""
        try:
            new_dict = json.loads(json_text)
            param_int = int(param)
            
            if not isinstance(new_dict, dict):
                raise ValueError("Input must be a dictionary")
            
            # 🔑 KEY: Direct update to session state
            self.sss.standards_bulk[param_int] = new_dict
            
            st.success(f"δ{'¹⁸O' if param_int == 18 else '¹³C'} anchors updated!")
            st.info(f"Updated values will persist until app restart")
            
            if "error_json" in self.sss:
                del self.sss.error_json
            
            # 🔑 KEY: Clear cached text area to show updated values
            cache_key = f"update_{param}_text"
            if cache_key in self.sss:
                del self.sss[cache_key]
            
            st.rerun()
        
        except json.JSONDecodeError as e:
            # Format error message with highlighted error line
            lines = json_text.split("\n")
            if e.lineno <= len(lines):
                lines[e.lineno - 1] = f":red[{lines[e.lineno - 1]}]"
            
            # Add indentation to non-first/last lines
            for idx in range(1, len(lines) - 1):
                lines[idx] = f"    {lines[idx]}"
            
            formatted_json = "\n".join(lines)
            
            if "error_json" not in self.sss:
                self.sss["error_json"] = []
            self.sss["error_json"].append((e, formatted_json))
    
    
    def _update_standards_dict(self, mz: str, json_text: str, scale: str) -> None:
        """Update standards dictionary with new values."""
        self.sss["last_updated_stds"] = scale
        
        try:
            new_dict = json.loads(json_text)
            self.sss.standards_nominal[scale][mz] = new_dict
            st.success("Dictionary updated successfully!")
            
            # Clear any previous errors
            if "error_json" in self.sss:
                del self.sss.error_json
            
            st.rerun()
            
        except json.JSONDecodeError as e:
            # Format error message with highlighted error line
            lines = json_text.split("\n")
            if e.lineno <= len(lines):
                lines[e.lineno - 1] = f":red[{lines[e.lineno - 1]}]"
            
            # Add indentation to non-first/last lines
            for idx in range(1, len(lines) - 1):
                lines[idx] = f"    {lines[idx]}"
            
            formatted_json = "\n".join(lines)
            
            if "error_json" not in self.sss:
                self.sss["error_json"] = []
            self.sss["error_json"].append((e, formatted_json))
    
    def _render_reset_standards_button(self) -> None:
        """Render reset standards button with confirmation dialog."""
        if st.sidebar.button("Reset standards!", key="reset_standards"):
            self.sss.show_confirmation = True
        
        if self.sss.show_confirmation:
            st.sidebar.warning("Reset all standard values?")
            confirm_col, cancel_col = st.sidebar.columns(2)
            
            with confirm_col:
                with stylable_container(
                    key="confirm_reset_button",
                    css_styles="""
                    [data-testid="baseButton-secondary"] {
                        background-color: red;
                    }
                    """,
                ):
                    if st.button("Yes"):
                        self._load_initial_standards(force=True)
            
            with cancel_col:
                with stylable_container(
                    key="cancel_reset_button",
                    css_styles="""
                    [data-testid="baseButton-secondary"] {
                        background-color: grey;
                    }
                    """,
                ):
                    if st.button("No"):
                        self.sss.show_confirmation = False
                        st.rerun()
    
    def _prepare_processing_data(self, config: ProcessingConfig) -> None:
        """Prepare data for processing."""
        # Filter data by selected sessions
        raw_data = self.sss.input_rep[
            self.sss.input_rep["Session"].isin(config.processing_sessions)
        ]
        
        # Rename datetime column if present
        if "datetime" in raw_data.columns:
            raw_data = raw_data.rename(columns={"datetime": "Timetag"})
        
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
                        STDS[mz] = str({key: self.sss["standards_nominal"][self.sss.scale][mz][key] for key in self.sss.standards_nominal[self.sss.scale].get(mz, False) if key in self.sss.standards['Sample'].values})
                        
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
        
        # Create download link
        excel_data = self.excel_exporter.create_multi_sheet_excel(dataframes)
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
        run_button = st.sidebar.button("Run...", key="BUTTON1",
        # disabled= not(
        #    self.sss.get("process_D47", False) or
        #    self.sss.get("process_D48", False) or
        #    self.sss.get("process_D49", False)
        #             )
        )
        
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
