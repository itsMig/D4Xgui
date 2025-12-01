"""Authentication module for D4Xgui Streamlit application.

This module provides (optional) password-based authentication functionality for the D4Xgui
application using Streamlit's session state and secrets management.
"""

import hmac
import time
from typing import Optional

import streamlit as st


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class Authenticator:
    """Handles user authentication for the Streamlit application."""
    
    # Session state keys
    PASSWORD_KEY = "password"
    PASSWORD_CORRECT_KEY = "password_correct"
    
    # UI messages
    WELCOME_MESSAGE = (
        "This is an early version of D4Xgui. Please enter the password to access this app."
    )
    
    CONTACT_MESSAGE = (
        'Contact <a href="mailto:bernecker@em.uni-frankfurt.de?'
        'subject=D4Xgui - Alpha password request&'
        'body=Hi Miguel,%0D%0Awe are interested in testing your app.%0D%0A%0D%0A'
        'Institute:%0D%0AMass spectrometer:%0D%0APreparation line:%0D%0A%0D%0A'
        'Could you please provide me the password needed to join alpha testing?">'
        'Miguel Bernecker</a> if you want to join alpha testing.'
    )
    
    ERROR_MESSAGE = "😕 Password incorrect"
    
    def __init__(self, delay_seconds: float = 0.25):
        if not 'password' in st.secrets:
            st.session_state[self.PASSWORD_CORRECT_KEY] = True

        """Initialize the authenticator.
        
        Args:
            delay_seconds: Delay after password entry to prevent brute force attacks.
        """
        self.delay_seconds = delay_seconds
    
    def _get_stored_password(self) -> Optional[str]:
        """Get the stored password from Streamlit secrets.
        
        Returns:
            The stored password or None if not found.
        """
        try:
            return st.secrets["password"]
        except KeyError:
            st.error("Password not configured in secrets. Please contact the administrator.")
            return None
    
    def _validate_password(self, entered_password: str, stored_password: str) -> bool:
        """Validate the entered password against the stored password.
        
        Args:
            entered_password: Password entered by the user.
            stored_password: Password stored in secrets.
            
        Returns:
            True if passwords match, False otherwise.
        """
        return hmac.compare_digest(entered_password, stored_password)
    
    def _handle_password_entry(self) -> None:
        """Handle password entry and validation."""
        # Add delay to prevent brute force attacks
        time.sleep(self.delay_seconds)
        
        entered_password = st.session_state.get(self.PASSWORD_KEY, "")
        stored_password = self._get_stored_password()
        
        if stored_password is None:
            return
        
        if self._validate_password(entered_password, stored_password):
            st.session_state[self.PASSWORD_CORRECT_KEY] = True
            # Clear password from session state for security
            if self.PASSWORD_KEY in st.session_state:
                del st.session_state[self.PASSWORD_KEY]
        else:
            st.session_state[self.PASSWORD_CORRECT_KEY] = False
    
    def _show_login_form(self) -> None:
        """Display the login form."""
        st.markdown(self.WELCOME_MESSAGE)
        st.markdown(self.CONTACT_MESSAGE, unsafe_allow_html=True)
        
        st.text_input(
            "Password",
            type="password",
            on_change=self._handle_password_entry,
            key=self.PASSWORD_KEY
        )
        
        # Show error message if password was incorrect
        if st.session_state.get(self.PASSWORD_CORRECT_KEY) is False:
            st.error(self.ERROR_MESSAGE)
    
    def is_authenticated(self) -> bool:
        """Check if the user is authenticated.
        
        Returns:
            True if user is authenticated, False otherwise.
        """
        return st.session_state.get(self.PASSWORD_CORRECT_KEY, False)
    
    def require_authentication(self) -> bool:
        """Require authentication to proceed.
        
        Shows login form if not authenticated and stops execution.
        
        Returns:
            True if authenticated, does not return if not authenticated.
        """
        if self.is_authenticated():
            return True
        
        self._show_login_form()
        return False


# Global authenticator instance
_authenticator = Authenticator()


def check_password() -> bool:
    """Legacy function for backward compatibility.
    
    Returns:
        True if the user has the correct password.
    """
    return _authenticator.require_authentication()


def require_authentication() -> None:
    """Require authentication or stop the app.
    
    This function will stop the Streamlit app execution if the user
    is not authenticated.
    """
    if not _authenticator.require_authentication():
        st.stop()


# Maintain backward compatibility
if not check_password():
    st.stop()