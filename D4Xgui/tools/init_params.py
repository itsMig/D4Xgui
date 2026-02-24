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
                "60C": 0.77062,
                "50C": 0.805,
                "25C": 0.9196,
                "4C": 1.0402,
            },
            "48": {
                "1000C": 0.000,
                "50C": 0.2607,
                "25C": 0.345
            },
            "49": {
                "1000C": 0.000,
                "50C": 2.00,   # approx
                "25C": 2.228,  # ibid.; 25°C value per Wang 2004
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
        
    }

    STANDARDS_BULK = {18:
         {
        "ETH-1": -2.19, "ETH-2": -18.69, "ETH-3": -1.78,
             "ETH3oxi": -1.78,
        # "ETH-1-110C": -2.19, "ETH-2-110C": -18.69,
        "ETH-1_110C": -2.19, "ETH-2_110C": -18.69,
             "IAEA-C1": (-2.31-2.32)/2, "IAEA-C2" : (-8.94-9.00)/2, #Bernasconi2018, mixed MIT+ETH
             
    },
    
    13:{
        "ETH-1": 2.02, "ETH-2": -10.17, "ETH-3": 1.71,
        'ETH3oxi': 1.71,
        # "ETH-1-110C": 2.02, "ETH-2-110C": -10.17,
        "ETH-1_110C": 2.02, "ETH-2_110C": -10.17,
        "IAEA-C1": (2.43 +2.47) / 2, "IAEA-C2": (-8.26 - 8.25) / 2,  # Bernasconi2018, mixed MIT+ETH
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
        """Get all isotopic standards, including user-defined custom frames.
        
        Returns:
            Dictionary containing all standard reference frames.
        """
        from tools import config as _cfg
        merged = cls.STANDARDS_NOMINAL.copy()
        custom = _cfg.get("custom_reference_frames") or {}
        merged.update(custom)
        return merged
    
    @classmethod
    def get_bulk(cls) -> Dict[str, Any]:
        """Get bulk isotopic standards, merging any user overrides from config.
        
        Returns:
            Dictionary containing bulk standard values (keyed by isotope
            number, e.g. 13, 18).
        """
        from tools import config as _cfg
        base = {k: dict(v) for k, v in cls.STANDARDS_BULK.items()}
        overrides = _cfg.get("standards_bulk_overrides") or {}
        for iso_key, values in overrides.items():
            int_key = int(iso_key)
            base[int_key] = dict(values)
        return base

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
