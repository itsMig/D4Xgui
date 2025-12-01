from pathlib import Path
from typing import Optional

import streamlit as st
#from streamlit_extras.app_logo import add_logo as streamlit_add_logo



class SidebarLogoManager:
    """Manages the sidebar logo for the Streamlit application."""
    
    DEFAULT_LOGO_PATH = "static/D4Xgui_logo_master_red08.png"
    DEFAULT_HEIGHT = 130
    
    def __init__(self, logo_path: Optional[str] = None, height: int = DEFAULT_HEIGHT):
        """Initialize the sidebar logo manager.
        
        Args:
            logo_path: Path to the logo image. Uses default if None.
            height: Height of the logo in pixels.
        """
        self.logo_path = Path(logo_path or self.DEFAULT_LOGO_PATH)
        self.height = height
    
    def add_logo(self) -> None:
        """Add logo to the sidebar.
        
        Uses streamlit-extras to add the logo to the sidebar navigation.
        """
        try:
            if self.logo_path.exists():
                #streamlit_add_logo(str(self.logo_path), height=self.height)
                st.logo(
                        str(self.logo_path),
                        #link="https://streamlit.io/gallery",
                        #str(self.logo_path),
                        size='large',#height=self.height,
                        
                    )
               
                
            else:
                st.warning(f"Logo file not found: {self.logo_path}")
        except Exception as e:
            st.warning(f"Could not add logo: {e}")


# Global logo manager instance
_logo_manager = SidebarLogoManager()


def add_logo() -> None:
    """Add logo to sidebar (legacy function for backward compatibility)."""
    _logo_manager.add_logo()
