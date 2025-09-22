"""Page configuration module for D4Xgui Streamlit application.

This module provides centralized configuration for all pages in the application,
including page titles, icons, layout settings, and menu items.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
from PIL import Image


class PageConfigManager:
    """Manages page configuration for the Streamlit application."""
    
    # Default icon path
    DEFAULT_ICON_PATH = "static/D4Xgui_logo_master_red08.png"
    
    def __init__(self, icon_path: Optional[str] = None):
        """Initialize the page configuration manager.
        
        Args:
            icon_path: Path to the application icon. Uses default if None.
        """
        self.icon_path = Path(icon_path or self.DEFAULT_ICON_PATH)
        self._icon = self._load_icon()
        self._shared_config = self._get_shared_config()
        self._page_configs = self._get_page_configurations()
    
    def _load_icon(self) -> Optional[Image.Image]:
        """Load the application icon.
        
        Returns:
            PIL Image object or None if loading fails.
        """
        try:
            if self.icon_path.exists():
                return Image.open(self.icon_path)
        except Exception as e:
            st.warning(f"Could not load icon from {self.icon_path}: {e}")
        return None
    
    def _get_shared_config(self) -> Dict[str, Any]:
        """Get shared configuration for all pages.
        
        Returns:
            Dictionary containing shared page configuration.
        """
        return {
            "page_icon": self._icon,
            "layout": "wide",
            "initial_sidebar_state": "expanded",
            "menu_items": {
                "About": (
                    "More information on carbonate dual clumped isotope geochemistry:\n"
                    "[Fiebig et al. (2021)](https://doi.org/10.1016/j.gca.2021.07.012)"
                ),
                "Report a bug": (
                    "mailto:bernecker@em.uni-frankfurt.de?"
                    "subject=D4Xgui%20-%20Bug%20report"
                ),
            },
        }
    
    def _get_page_configurations(self) -> Dict[int, Dict[str, Any]]:
        """Get configuration for all pages.
        
        Returns:
            Dictionary mapping page numbers to their configurations.
        """
        return {
            0: {"page_title": "D4Xgui - 👋 Main page", **self._shared_config},
            1: {"page_title": "D4Xgui - 📝 Data I/O", **self._shared_config},
            3: {"page_title": "D4Xgui - 🧩 Baseline correction", **self._shared_config},
            4: {"page_title": "D4Xgui - 🧮 Processing", **self._shared_config},
            5: {"page_title": "D4Xgui - 📊 Standardization Results", **self._shared_config},
            6: {"page_title": "D4Xgui - 🪄 Dual Clumped Space", **self._shared_config},
            7: {"page_title": "D4Xgui - 🔮 Discover Results", **self._shared_config},
            8: {"page_title": "D4Xgui - 🧩 D47crunch plots", **self._shared_config},
            97: {"page_title": "D4Xgui - 🗃️ Database management", **self._shared_config},
            99: {"page_title": "D4Xgui - 🧠 Session state", **self._shared_config},
            100: {"page_title": "D4Xgui - 💾 Save & Reload", **self._shared_config},
        }
    
    def configure_page(self, page_number: int) -> None:
        """Configure a Streamlit page.
        
        Args:
            page_number: Page number to configure.
            
        Raises:
            ValueError: If page number is not found in configuration.
        """
        if page_number not in self._page_configs:
            raise ValueError(f"Page {page_number} not found in configuration")
        
        config = self._page_configs[page_number]
        
        try:
            st.set_page_config(
                page_title=config["page_title"],
                page_icon=config["page_icon"],
                layout=config["layout"],
                initial_sidebar_state=config["initial_sidebar_state"],
                menu_items=config["menu_items"]
            )
        except st.errors.StreamlitAPIException:
            # Page config can only be set once, ignore subsequent calls
            pass
        except Exception as e:
            st.warning(f"Could not set page config: {e}")


# Global page config manager instance
_page_config_manager = PageConfigManager()


def set_page_config(page_number: int) -> None:
    """Configure a Streamlit page (legacy function for backward compatibility).
    
    Args:
        page_number: Page number to configure.
    """
    _page_config_manager.configure_page(page_number)


# Legacy configuration dictionaries for backward compatibility
try:
    icon = Image.open("static/D4Xgui_logo_master_red08.png")
except Exception:
    icon = None

_SHARED = {
    "page_icon": icon,
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "About": (
            "More information on carbonate dual clumped isotope geochemistry:\n"
            "[Fiebig et al. (2021)](https://doi.org/10.1016/j.gca.2021.07.012)"
        ),
        "Report a bug": (
            "mailto:bernecker@em.uni-frankfurt.de?subject=D4Xgui%20-%20Bug%20report"
        ),
    },
}

PAGE_CONFIG = {
    0: {"page_title": "D4Xgui - 👋 Main page", **_SHARED},
    1: {"page_title": "D4Xgui - 📝 Data I/O", **_SHARED},
    3: {"page_title": "D4Xgui - 🧩 Baseline correction", **_SHARED},
    4: {"page_title": "D4Xgui - 🧮 Processing", **_SHARED},
    5: {"page_title": "D4Xgui - 📊 Standardization Results", **_SHARED},
    6: {"page_title": "D4Xgui - 🪄 Dual Clumped Space", **_SHARED},
    7: {"page_title": "D4Xgui - 🔮 Discover Results", **_SHARED},
    8: {"page_title": "D4Xgui - 🧩 D47crunch plots", **_SHARED},
    97: {"page_title": "D4Xgui - 🗃️ Database management", **_SHARED},
    99: {"page_title": "D4Xgui - 🧠 Session state", **_SHARED},
}
