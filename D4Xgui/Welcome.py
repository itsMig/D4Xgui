#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
from pathlib import Path
from typing import Dict, Any
import toml

import streamlit as st
import D47crunch

from tools.page_config import set_page_config
from tools import sidebar_logo
from tools.database import init_db
from tools.init_params import INIT_PARAMS



class WelcomePageManager:
    """Manages the welcome page content and initialization."""
    
    def __init__(self):
        """Initialize the welcome page manager."""
        self.app_dir = Path(__file__).parent.absolute()
        self.session_state = st.session_state
        self._setup_page()
        self._initialize_database()
        self._initialize_session_state()
    
    def _setup_page(self) -> None:
        """Set up the page configuration and sidebar."""
        set_page_config(0)
        sidebar_logo.add_logo()
    
    def _initialize_database(self) -> None:
        """Initialize the application database."""
        init_db()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state with default parameters if not present."""
        if 'standards_nominal' not in self.session_state:
            self.session_state['standards_nominal'] = {**INIT_PARAMS['standards_nominal']}
            #self.session_state['working_gas'] = {**INIT_PARAMS['working_gas']}
    
    def _get_info_content(self) -> str:
        """Generate the welcome page information content.
        
        Returns:
            Formatted markdown string with welcome information.
        """
       
        return rf"""## Welcome to D4Xgui v1.0.2!

[D4Xgui](https://github.com/itsMig/D4Xgui) is developed to enable easy access to state-of-the-art CO₂ clumped isotope (∆₄₇, ∆₄₈ and ∆₄₉) data processing.
A recently developed optimizer algorithm allows pre-processing of mass spectrometric raw intensities utilizing a m/z47.5 half-mass Faraday cup correction to account for the effect of a negative pressure baseline, which is essential for accurate and highest precision clumped isotope analysis of CO₂ ([Bernecker et al., 2023](https://doi.org/10.1016/j.chemgeo.2023.121803)).
It is backed with the recently published processing tool [D47crunch (v{D47crunch.__version__})](https://github.com/mdaeron/D47crunch) (following the methodology outlined in [Daeron, 2021](https://doi.org/10.1029/2020GC009592)), which allows standardization under consideration of full error propagation and has been used for the InterCarb community effort ([Bernasconi et al., 2021](https://doi.org/10.1029/2020GC009588)).
This web-app allows users to discover replicate- or sample-based processing results in interactive spreadsheets and plots.

<br>

Example data is accessible from the Data-IO page, or can be downloaded from [GitHub](https://github.com/itsMig/D4Xgui/tree/main/D4Xgui/static).

<br>

In order to process post-background corrected data (Data-IO page, Upload δ⁴⁵-δ⁴⁹ replicates tab), the following columns need to be provided (`.xlsx`, `.csv`):

| `UID` | `Sample` | `Session` | `Timetag` | `d45` | `d46` | `d47` | `d48` | `d49` |
|----|----|----|----|----|----|----|----|----|

$~$

Baseline correction can be performed using a m/z47.5 half-mass cup. For this purpose either a set of equilibrated gases (via heated-gas-line), or carbonate standards (via target values) is used to determine m/z47, m/z48 and m/z49-specific scaling factors. Please upload a cycle-based spreadsheet (`.xlsx`, `.csv`) including the following columns (Data-IO page, Upload m/z44-m/z49 intensities tab):

| `UID` | `Sample` | `Session` | `Timetag` | `Replicate` |
|----|----|----|----|----|

| `raw_r44` | `raw_r45` | `raw_r46` | `raw_r47` | `raw_r48` | `raw_r49` | `raw_r47.5` |
|----|----|----|----|----|----|----|

| `raw_s44` | `raw_s45` | `raw_s46` | `raw_s47` | `raw_s48` | `raw_s49` | `raw_s47.5` |
|----|----|----|----|----|----|----|


<br><br>
Please find the [code documentation here](https://itsmig.github.io/D4Xgui/index.html).
"""
    
    def _save_readme(self, content: str) -> None:
        """Save the welcome content to README.md file.
        
        Args:
            content: Content to save to README.md.
        """
        try:
            with open(os.path.join(self.app_dir,'../INSTALLATION.md', ),'r') as file:
                content += f"\n" + file.read()

            readme_path = Path('../README.md')
            readme_path.write_text(content, encoding='utf-8')
        except Exception as e:
            st.warning(f"Could not save README.md: {e}")
    
    def display_welcome_page(self) -> None:
        """Display the welcome page content."""
        
        info_content = self._get_info_content()
        # self._save_readme(info_content)
        st.markdown(info_content, unsafe_allow_html=True)


def main():
    """Main function to run the welcome page."""
    welcome_manager = WelcomePageManager()
    welcome_manager.display_welcome_page()


if __name__ == "__main__":
    main()
