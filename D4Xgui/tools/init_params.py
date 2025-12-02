"""
Initial parameters and standards for D4Xgui application.
"""

from typing import Dict, Any


class IsotopeStandards:
    """Container for isotopic standard values and working gas parameters."""

    # Standard reference frames and their isotopic compositions
    STANDARDS_NOMINAL = {
        "CDES": {
            "#info": "Wang_2004, revised by Petersen_2019",
            "47": {
                "1000C": 0.0266,
                "25C": 0.9196,
                "60C": 0.77062,
                "4C": 1.0402,
            },
            "48": {
                "1000C": 0.000,
                "25C": 0.345
            },
            "49": {
                "1000C": 0.000,
                "25C": 2.228
            },
        },
        
        "ICDES": {
            "#info": "InterCarb, Bernasconi et al. (2021)",
            "47": {
                "ETH-1": 0.2052,
                "ETH-2": 0.2085,
                "ETH-3": 0.6132,
            },
        },
        
        "mixed Fiebig24+CDES": {
            "#info": "Longterm GU, Fiebig et al. (2024)",
            "47": {
                "1000C": 0.0266,
                "25C": 0.9196,
                "ETH-1": 0.2052,
                "ETH-2": 0.2085,
                "ETH3OXI": 0.6132,
                "ETH3B": 0.6132,
                "ETH3oxi": 0.6132,
                "ETH-3": 0.6132,
                "GU1": 0.2254,
            },
            "48": {
                "1000C": 0.000,
                "25C": 0.345,
                "ETH-1": 0.1277,
                "ETH-2": 0.1299,
                "ETH3OXI": 0.2481,
                "ETH3oxi": 0.2481,
                "ETH3B": 0.2481,
                "ETH-3": 0.2481,
                "GU1": -0.3998,
            },
        },
        
        "Fiebig2024 carb": {
            "#info": "Longterm GU, Fiebig et al. (2024)",
            "47": {
                "ETH-1": 0.2052,
                "ETH-2": 0.2085,
                "ETH3OXI": 0.6132,
                "ETH3oxi": 0.6132,
                "ETH-3": 0.6132,
                "GU1": 0.2254,
            },
            "48": {
                "ETH-1": 0.1277,
                "ETH-2": 0.1299,
                "ETH3OXI": 0.2481,
                "ETH3oxi": 0.2481,
                "ETH-3": 0.2481,
                "GU1": -0.3998,
            },
        },
        
        # "Bernecker2023 carb": {
        #     "#info": "Longterm GU, Bernecker et al. (2023)",
        #     "47": {
        #         "ETH-1": 0.2061,
        #         "ETH-2": 0.2085,
        #         "ETH-3": 0.6032,
        #         "GU1": 0.2244,
        #     },
        #     "48": {
        #         "ETH-1": 0.1286,
        #         "ETH-2": 0.1286,
        #         "ETH-3": 0.3039,
        #         "GU1": -0.4015,
        #     },
        # },
        
        "mixed47": {
            "#info": "ETH1+2 (47), 25+1000C (47+48)",
            "47": {
                "ETH-1": 0.2052,
                "ETH-2": 0.2085,
                "1000C": 0.0266,
                "25C": 0.9196,
            },
            "48": {
                "1000C": 0.000,
                "25C": 0.345
            },
        },
        
        "mixed4748": {
            "#info": "ETH1+2, 25+1000C (47+48)",
            "47": {
                "ETH-1": 0.2052,
                "ETH-2": 0.2085,
                "1000C": 0.0266,
                "25C": 0.9196,
            },
            "48": {
                "ETH-1": 0.1277,
                "ETH-2": 0.1299,
                "1000C": 0.000,
                "25C": 0.345,
            },
        },
    }

    STANDARDS_BULK = {18:
         {
        "ETH-1": -2.19, "ETH-2": -18.69, "ETH-3": -1.78,
             "ETH3oxi": -1.78,
        # "ETH-1-110C": -2.19, "ETH-2-110C": -18.69,
        "ETH-1_110C": -2.19, "ETH-2_110C": -18.69,
    },
    
    13:{
        "ETH-1": 2.02, "ETH-2": -10.17, "ETH-3": 1.71,
        'ETH3oxi': 1.71,
        # "ETH-1-110C": 2.02, "ETH-2-110C": -10.17,
        "ETH-1_110C": 2.02, "ETH-2_110C": -10.17,
    }}
    
    # @classmethod
    # def get_working_gas(cls) -> Dict[str, float]:
    #     """Get working gas isotopic composition.
    #
    #     Returns:
    #         Dictionary containing working gas d13C and d18O values.
    #     """
    #     return cls.WORKING_GAS.copy()
    
    @classmethod
    def get_standards(cls) -> Dict[str, Any]:
        """Get all isotopic standards.
        
        Returns:
            Dictionary containing all standard reference frames.
        """
        return cls.STANDARDS_NOMINAL.copy()
    
    @classmethod
    def get_bulk(cls) -> Dict[str, Any]:
        """Get all isotopic standards.
        
        Returns:
            Dictionary containing all standard reference frames.
        """
        return cls.STANDARDS_BULK.copy()

    @classmethod
    def get_standard(cls, standard_name: str) -> Dict[str, Any]:
        """Get a specific isotopic standard.
        
        Args:
            standard_name: Name of the standard to retrieve.
            
        Returns:
            Dictionary containing the standard's isotopic values.
            
        Raises:
            KeyError: If the standard name is not found.
        """
        if standard_name not in cls.STANDARDS_NOMINAL:
            available = list(cls.STANDARDS_NOMINAL.keys())
            raise KeyError(f"Standard '{standard_name}' not found. Available: {available}")
        
        return cls.STANDARDS_NOMINAL[standard_name].copy()


# Legacy dictionary for backward compatibility
INIT_PARAMS = {
    #"working_gas": IsotopeStandards.get_working_gas(),
    "standards_nominal": IsotopeStandards.get_standards(),
    'standards_bulk' : IsotopeStandards.get_bulk(),
}
