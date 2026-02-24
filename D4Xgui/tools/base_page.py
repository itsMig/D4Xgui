"""Base page class for D4Xgui Streamlit pages."""

import os
from typing import Any

import streamlit as st

from tools.page_config import PageConfigManager
from tools.sidebar_logo import SidebarLogoManager
from tools.authenticator import Authenticator
from tools import config as cfg


class BasePage:
    """Base class for all D4Xgui Streamlit pages.

    Subclasses must set PAGE_NUMBER and implement run().
    Override _initialize_session_state() and _validate_input_data() as needed.
    """

    PAGE_NUMBER: int = -1
    PAGE_TITLE: str = ""
    REQUIRES_AUTH: bool = True
    SHOW_LOGO: bool = True

    def __init__(self) -> None:
        self.sss: Any = st.session_state
        self.session_state: Any = st.session_state
        self._setup_page()
        self._initialize_session_state()

    def _setup_page(self) -> None:
        cfg.ensure_theme()
        if self.PAGE_NUMBER >= 0:
            PageConfigManager().configure_page(self.PAGE_NUMBER)
        if self.SHOW_LOGO:
            SidebarLogoManager().add_logo()
        if self.PAGE_TITLE:
            st.title(self.PAGE_TITLE)
        if self.REQUIRES_AUTH and "PYTEST_CURRENT_TEST" not in os.environ:
            if not Authenticator().require_authentication():
                st.stop()

    def _initialize_session_state(self) -> None:
        """Override in subclasses to set up defaults."""

    def _validate_input_data(self) -> bool:
        """Override in subclasses. Return True if data is present."""
        return True

    def run(self) -> None:
        raise NotImplementedError
