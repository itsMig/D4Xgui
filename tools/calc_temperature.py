from scipy.optimize import  minimize_scalar
"""Temperature calculation module for clumped isotope analysis.

This module provides functions to calculate clumped isotope values (D47, D48, D63, D64, D65)
as a function of temperature using polynomial equations from published calibrations
(Hill et al., 2014; Fiebig et al. 2021, 2024).
"""

from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt


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
    
    # Scaling and offset parameters for Fiebig et al. (2024)
    FIEBIG2024_D47_SCALING = 1.038
    FIEBIG2024_D47_OFFSET = 0.1848
    FIEBIG2024_D48_SCALING = 1.038
    FIEBIG2024_D48_OFFSET = 0.1214
    
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


class ClumpedIsotopePlotter:
    """Plotting utilities for clumped isotope calibration curves."""
    
    DEFAULT_TEMP_RANGE = [8, 15, 25, 35, 50, 75, 100, 200, 300, 500, 800, 1100]
    
    @classmethod
    def _get_inverse_temp_array(cls, temp_range_celsius: list = None) -> np.ndarray:
        """Convert temperature range to inverse temperature in Kelvin.
        
        Args:
            temp_range_celsius: Temperature range in Celsius. Uses default if None.
            
        Returns:
            Array of 1/T values in Kelvin.
        """
        if temp_range_celsius is None:
            temp_range_celsius = cls.DEFAULT_TEMP_RANGE
        return np.array([1 / (273.15 + temp) for temp in temp_range_celsius])
    
    @classmethod
    def plot_dual_clumped_fiebig2021(cls, temp_range_celsius: list = None) -> None:
        """Plot dual clumped isotope calibration curve (Fiebig et al., 2021).
        
        Args:
            temp_range_celsius: Temperature range in Celsius. Uses default if None.
        """
        inverse_temp_k = cls._get_inverse_temp_array(temp_range_celsius)
        d47_values = TemperatureCalculator.calculate_d47_fiebig2021(inverse_temp_k)
        d48_values = TemperatureCalculator.calculate_d48_fiebig2021(inverse_temp_k)
        
        plt.plot(
            d48_values, d47_values,
            label='$∆_{48}/∆_{47}$ calibration\n(Fiebig et al., 2021)',
            linestyle='-.',
            color='black'
        )
    
    @classmethod
    def plot_d64_d63_hill2014(cls, temp_range_celsius: list = None) -> None:
        """Plot theoretical D64/D63 relationship (Hill et al., 2014).
        
        Args:
            temp_range_celsius: Temperature range in Celsius. Uses default if None.
        """
        inverse_temp_k = cls._get_inverse_temp_array(temp_range_celsius)
        d63_values = TemperatureCalculator.calculate_d63_hill2014(inverse_temp_k)
        d64_values = TemperatureCalculator.calculate_d64_hill2014(inverse_temp_k)
        
        plt.plot(
            d64_values, d63_values,
            label='Theoretical $∆_{64}/∆_{63}$ polynom\n(Hill et al., 2014)',
            linestyle='-',
            color='black'
        )
    
    @classmethod
    def plot_d65_d63_hill2014(cls, temp_range_celsius: list = None) -> None:
        """Plot theoretical D65/D63 relationship (Hill et al., 2014).
        
        Args:
            temp_range_celsius: Temperature range in Celsius. Uses default if None.
        """
        inverse_temp_k = cls._get_inverse_temp_array(temp_range_celsius)
        d63_values = TemperatureCalculator.calculate_d63_hill2014(inverse_temp_k)
        d65_values = TemperatureCalculator.calculate_d65_hill2014(inverse_temp_k)
        
        plt.plot(
            d65_values, d63_values,
            label='Theoretical $∆_{65}/∆_{63}$ polynom\n(Hill et al., 2014)',
            linestyle='-',
            color='black'
        )
    
    @classmethod
    def plot_d64_d65_hill2014(cls, temp_range_celsius: list = None) -> None:
        """Plot theoretical D64/D65 relationship (Hill et al., 2014).
        
        Args:
            temp_range_celsius: Temperature range in Celsius. Uses default if None.
        """
        inverse_temp_k = cls._get_inverse_temp_array(temp_range_celsius)
        d64_values = TemperatureCalculator.calculate_d64_hill2014(inverse_temp_k)
        d65_values = TemperatureCalculator.calculate_d65_hill2014(inverse_temp_k)
        
        plt.plot(
            d64_values, d65_values,
            label='Theoretical $∆_{64}/∆_{65}$ polynom\n(Hill et al., 2014)',
            linestyle='-',
            color='black'
        )


# Legacy function aliases for backward compatibility
def f_poly4(poly, i):
    """Legacy function for 4th-order polynomial evaluation."""
    return TemperatureCalculator._evaluate_polynomial_4th_order(poly, i)

def D63_Hill2014(x_arr):
    """Legacy function for D63 calculation."""
    return TemperatureCalculator.calculate_d63_hill2014(x_arr)

def D64_Hill2014(x_arr):
    """Legacy function for D64 calculation."""
    return TemperatureCalculator.calculate_d64_hill2014(x_arr)

def D65_Hill2014(x_arr):
    """Legacy function for D65 calculation."""
    return TemperatureCalculator.calculate_d65_hill2014(x_arr)

def D47_Fiebig2021(x_arr):
    """Legacy function for D47 calculation (Fiebig 2021)."""
    return TemperatureCalculator.calculate_d47_fiebig2021(x_arr)

def D48_Fiebig2021(x_arr):
    """Legacy function for D48 calculation (Fiebig 2021)."""
    return TemperatureCalculator.calculate_d48_fiebig2021(x_arr)

def D47_Fiebig2024(x_arr):
    """Legacy function for D47 calculation (Fiebig 2024)."""
    return TemperatureCalculator.calculate_d47_fiebig2024(x_arr)

def D48_Fiebig2024(x_arr):
    """Legacy function for D48 calculation (Fiebig 2024)."""
    return TemperatureCalculator.calculate_d48_fiebig2024(x_arr)

def getTemp_D47_Fiebig2021(temp, args):
    """Legacy function for temperature difference calculation (D47, Fiebig 2021)."""
    return TemperatureCalculator.get_temperature_difference_d47_fiebig2021(temp, args["D47"])

def getTemp_D48_Fiebig2021(temp, args):
    """Legacy function for temperature difference calculation (D48, Fiebig 2021)."""
    return TemperatureCalculator.get_temperature_difference_d48_fiebig2021(temp, args["D48"])

def getTemp_D47_Fiebig2024(temp, args):
    """Legacy function for temperature difference calculation (D47, Fiebig 2024)."""
    return TemperatureCalculator.get_temperature_difference_d47_fiebig2024(temp, args["D47"])

def getTemp_D48_Fiebig2024(temp, args):
    """Legacy function for temperature difference calculation (D48, Fiebig 2024)."""
    return TemperatureCalculator.get_temperature_difference_d48_fiebig2024(temp, args["D48"])

def plotDualClumped():
    """Legacy function for plotting dual clumped calibration."""
    ClumpedIsotopePlotter.plot_dual_clumped_fiebig2021()

def plotD64D63():
    """Legacy function for plotting D64/D63 relationship."""
    ClumpedIsotopePlotter.plot_d64_d63_hill2014()

def plotD65D63():
    """Legacy function for plotting D65/D63 relationship."""
    ClumpedIsotopePlotter.plot_d65_d63_hill2014()

def plotD64D65():
    """Legacy function for plotting D64/D65 relationship."""
    ClumpedIsotopePlotter.plot_d64_d65_hill2014()


