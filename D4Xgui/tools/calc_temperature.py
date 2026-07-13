"""Temperature calculation module for clumped isotope analysis.

This module provides functions to calculate clumped isotope values (D47, D48, D63, D64, D65)
as a function of temperature using polynomial equations from published calibrations
(Hill et al., 2014; Fiebig et al. 2021, 2024).
"""

from typing import Union, Tuple

import numpy as np
from scipy.optimize import minimize_scalar



class TemperatureCalculator:
    """Calculator for clumped isotope temperature relationships."""
    
    # Polynomial coefficients for Hill et al. (2014)
    POLY_63_COEFFS = (-5.896755e00, -3.520888e03, 2.391274e07, -3.540693e09)
    POLY_64_COEFFS = (6.001624e00, -1.298978e04, 8.995634e06, -7.422972e08)
    POLY_65_COEFFS = (-6.741e00, -1.950e04, 5.845e07, -8.093e09)
    
    # Scaling and offset parameters for Fiebig et al. (2021)
    FIEBIG2021_D47_SCALING = 1.0381881
    FIEBIG2021_D47_OFFSET = 0.1855537
    FIEBIG2021_D48_SCALING = 1.0280693
    FIEBIG2021_D48_OFFSET = 0.1244564
    

    # Fiebig et al. (2024) — published rounded Hill×affine parameters (CDES90).
    FIEBIG2024_D47_SCALING = 1.038
    FIEBIG2024_D47_OFFSET = 0.1848
    FIEBIG2024_D48_SCALING = 1.038
    FIEBIG2024_D48_OFFSET = 0.1214

    # Lab-reprocessed full-precision values (optional comparison overlay).
    FIEBIG2024_D47_SCALING_REPROCESSED = 1.0383343176557291
    FIEBIG2024_D47_OFFSET_REPROCESSED = 0.18477481336587162
    FIEBIG2024_D48_SCALING_REPROCESSED = 1.0379702801201238
    FIEBIG2024_D48_OFFSET_REPROCESSED = 0.12135157920099604

    @classmethod
    def hill_affine_to_d4x_coefs(
        cls,
        hill_coeffs: Tuple[float, ...],
        scaling: float,
        offset: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        Convert Hill(2014) × scaling + offset to D95eq ``D4x_calib_function`` coefficients.

        D95eq evaluates Δ4x = c₀ + c₁/T + c₂/T² + c₃/T³ + c₄/T⁴ (T in °C).
        Hill uses Δ6x = a₁/T + a₂/T² + a₃/T³ + a₄/T⁴, so
        Δ4x = offset + scaling × Hill(1/T).
        """
        a1, a2, a3, a4 = hill_coeffs
        return (offset, a1 * scaling, a2 * scaling, a3 * scaling, a4 * scaling)

    @staticmethod
    def _evaluate_polynomial_4th_order(coeffs: Tuple[float, ...], x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate a 4th-degree polynomial.
        
        Args:
            coeffs: Polynomial coefficients (a, b, c, d) for ax + bx² + cx³ + dx⁴.
            x: Input value(s).
            
        Returns:
            Polynomial evaluation result.
        """
        a, b, c, d = coeffs
        return a * x + b * x**2 + c * x**3 + d * x**4
    
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
    
    @classmethod
    def calculate_d48_fiebig2021(cls, inverse_temp_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate D48 values using Fiebig et al. (2021) calibration.
        
        Args:
            inverse_temp_k: 1/T in Kelvin.
            
        Returns:
            D48 values.
        """
        d64 = cls.calculate_d64_hill2014(inverse_temp_k)
        return (d64 * cls.FIEBIG2021_D48_SCALING) + cls.FIEBIG2021_D48_OFFSET
    
    @classmethod
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
    def calculate_d48_fiebig2024(cls, inverse_temp_k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate D48 values using Fiebig et al. (2024) calibration.
        
        Args:
            inverse_temp_k: 1/T in Kelvin.
            
        Returns:
            D48 values.
        """
        d64 = cls.calculate_d64_hill2014(inverse_temp_k)
        return (d64 * cls.FIEBIG2024_D48_SCALING) + cls.FIEBIG2024_D48_OFFSET

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
    
    @classmethod
    def get_temperature_difference_d48_fiebig2021(cls, temp_celsius: float, target_d48: float) -> float:
        """Calculate absolute difference between target D48 and calculated D48 (Fiebig 2021).
        
        Args:
            temp_celsius: Temperature in Celsius.
            target_d48: Target D48 value.
            
        Returns:
            Absolute difference between target and calculated D48.
        """
        inverse_temp_k = 1 / (temp_celsius + 273.15)
        calculated_d48 = cls.calculate_d48_fiebig2021(inverse_temp_k)
        return abs(target_d48 - calculated_d48)
    
    @classmethod
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
    
    @classmethod
    def get_temperature_difference_d48_fiebig2024(cls, temp_celsius: float, target_d48: float) -> float:
        """Calculate absolute difference between target D48 and calculated D48 (Fiebig 2024).
        
        Args:
            temp_celsius: Temperature in Celsius.
            target_d48: Target D48 value.
            
        Returns:
            Absolute difference between target and calculated D48.
        """
        inverse_temp_k = 1 / (temp_celsius + 273.15)
        calculated_d48 = cls.calculate_d48_fiebig2024(inverse_temp_k)
        return abs(target_d48 - calculated_d48)

    @classmethod
    def get_temperature_difference_d47_anderson2021(cls, temp_celsius: float, target_d47: float) -> float:
        """Calculate absolute difference between target D47 and Anderson (2021) D47.

        Args:
            temp_celsius: Temperature in Celsius.
            target_d47: Target D47 value.

        Returns:
            Absolute difference.
        """
        return abs(target_d47 - ((0.0391 * 1e6 / temp_celsius ** 2) + 0.154))

    @classmethod
    def direct_temperature_swart2021(cls, d47: float) -> float:
        """Calculate temperature using Swart (2021): D47(CDES90) = 0.039 * 10^6/T^2 + 0.158.

        Args:
            d47: The D47 value.

        Returns:
            Temperature in degrees Celsius.
        """
        try:
            temp_k = np.sqrt(1e6 * 0.039 / (d47 - 0.158))
            return temp_k - 273.15
        except (ValueError, ZeroDivisionError):
            return np.nan

