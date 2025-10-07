"""
Common utilities and helper functions for D4Xgui application.
"""

import base64
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dill
import numpy as np
import pandas as pd
import streamlit as st


class Colors:
    """ANSI color codes for terminal output.
    
    Based on: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    """
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class PlotParameters:
    """Constants and parameters for plotting."""
    
    SYMBOLS = [
        "circle", "square", "diamond", "cross", "x",
        "triangle-up", "triangle-down", "triangle-left", "triangle-right",
        "triangle-ne", "triangle-se", "triangle-sw", "triangle-nw",
        "pentagon", "hexagon", "hexagon2", "octagon", "star", "hexagram",
        "star-triangle-up", "star-triangle-down", "star-square", "star-diamond",
        "diamond-tall", "diamond-wide", "hourglass", "bowtie",
        "square-x", "diamond-x",
        "arrow-up", "arrow-down", "arrow-left", "arrow-right",
        "arrow-bar-up", "arrow-bar-down", "arrow-bar-left", "arrow-bar-right",
    ] * 100  # Repeat to ensure enough symbols


# Legacy aliases for backward compatibility
color = Colors
PLOT_PARAMS = PlotParameters

class PlotlyConfig:
    CONFIG= {
        "toImageButtonOptions": {
        "format": "svg",  # one of png, svg, jpeg, webp
        "filename": "D4Xgui_plot",
        # "height": 'original',
       # "width": 800,
        #"scale": 2  # Multiply title/legend/axis/canvas sizes by this factor
    },
    "responsive": True,
    "doubleClick": "reset",
    "showTips": True,
    "editable": False,
     "scrollZoom": False,
    'width':"stretch",
    "height" : "stretch",
    }


def ensure_directory_exists(path):
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create.
    """
    if not os.path.isdir(path):
        try:
            original_umask = os.umask(0)
            os.mkdir(path, 0o755)
        finally:
            os.umask(original_umask)


def install_package(package: str) -> None:
    """Install a Python package using pip.
    
    Args:
        package: Name of the package to install.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def clear_session_cache() -> None:
    """Clear application cache (placeholder for future implementation)."""
    pass
    
    


def delete_all_session_data() -> None:
    """Delete all input/result data from session state.
    
    Preserves essential session state keys like authentication and configuration.
    """
    st.toast('Cleared former processing cache!')
    
    # Keys to preserve during cleanup
    preserved_keys = {'standards_nominal', 'password_correct', 'working_gas'}
    
    # Get all session state keys except preserved ones
    keys_to_delete = [
        key for key in st.session_state.keys() 
        if key not in preserved_keys
    ]
    
    # Delete the keys
    for key in keys_to_delete:
        del st.session_state[key]


def is_json_serializable(data: Any) -> bool:
    """Check if data can be serialized to JSON.
    
    Args:
        data: Data to check for JSON serializability.
        
    Returns:
        True if data is JSON serializable, False otherwise.
    """
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False

class SettingsManager:
    """Manages download and upload of application settings."""
    
    # Prefixes for special data types
    DATAFRAME_PREFIX = "pdDF="
    NUMPY_ARRAY_PREFIX = "npARR="
    
    def __init__(self):
        """Initialize the settings manager."""
        pass
    
    def _get_serializable_settings(self) -> Dict[str, Any]:
        """Get serializable settings from session state.
        
        Returns:
            Dictionary of serializable settings.
        """
        settings = {}
        
        for key, value in st.session_state.items():
            # Skip password-related keys
            if 'password' in key.lower():
                continue
            
            # Skip contour data (too complex)
            if 'contour' in key.lower():
                continue
            
            # Handle different data types
            if is_json_serializable(value):
                settings[key] = value
            elif isinstance(value, pd.DataFrame):
                settings[f"{self.DATAFRAME_PREFIX}{key}"] = value.to_csv(sep=";")
            elif isinstance(value, (np.ndarray, np.generic, list)):
                # Skip complex arrays for now
                pass
            elif isinstance(value, dict):
                # Skip complex dictionaries for now
                pass
        
        return settings
    
    def _apply_settings(self, settings: Dict[str, Any]) -> None:
        """Apply settings to session state.
        
        Args:
            settings: Dictionary of settings to apply.
        """
        for key, value in settings.items():
            if key.startswith(self.DATAFRAME_PREFIX):
                # Restore DataFrame
                original_key = key.split("=", 1)[1]
                st.session_state[original_key] = pd.read_csv(
                    StringIO(value), sep=";", index_col=0
                )
            elif key.startswith(self.NUMPY_ARRAY_PREFIX):
                # Restore numpy array
                original_key = key.split("=", 1)[1]
                st.session_state[original_key] = np.array([
                    float(x) for x in value.split(";")
                ])
            else:
                # Regular value
                st.session_state[key] = value
        
        st.toast('Successfully uploaded data!', icon='💾')
    
    def create_download_upload_interface(self) -> None:
        """Create the download/upload interface."""
        settings = self._get_serializable_settings()
        
        # Download section
        st.markdown("# DOWNLOAD")
        st.download_button(
            label="Download Settings",
            data=json.dumps(settings, indent=2),
            file_name=f"D4Xgui_sessionDump_{int(time.time())}.json",
            help="Click to Download Current Settings",
        )
        
        # Upload section
        st.markdown("# UPLOAD")
        uploaded_file = st.file_uploader(
            label="Select the Settings File to be uploaded",
            help=(
                "Select the Settings File (Downloaded in a previous run) that you want "
                "to be uploaded and then applied (by clicking 'Apply Settings' below)"
            ),
            type=['json']
        )
        
        if uploaded_file is not None:
            try:
                uploaded_settings = json.load(uploaded_file)
                st.button(
                    label="Apply Settings",
                    on_click=self._apply_settings,
                    args=(uploaded_settings,),
                    help="Click to Apply the Settings of the Uploaded file."
                )
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please upload a valid settings file.")
            except Exception as e:
                st.error(f"Error loading settings file: {e}")


def download_upload_settings():
    """Legacy function for backward compatibility."""
    settings_manager = SettingsManager()
    settings_manager.create_download_upload_interface()
    st.markdown("# UPLOAD")
    uploaded_file = st.file_uploader(
        label="Select the Settings File to be uploaded",
        help="Select the Settings File (Downloaded in a previous run) that you want"
        " to be uploaded and then applied (by clicking 'Apply Settings' above)"
        " in order to filter the perimeter",
    )
    if uploaded_file is not None:
        uploaded_settings = json.load(uploaded_file)
    else:
        uploaded_settings = settings_to_download
        # st.warning("Please select the cached file to be uploaded!")

    # 3. Apply Settings
    def upload_json_settings(json_settings):
        """Set session state values to what specified in the json_settings."""
        for k in json_settings.keys():
            if k.startswith("pdDF="):
                st.session_state[k.split("=")[-1]] = pd.read_csv(
                    StringIO(json_settings[k]), sep=";", index_col=0
                )
            elif k.startswith("npARR="):
                st.session_state[k.split("=")[-1]] = np.array([float(_) for _ in json_settings[k].split(";")])
            else:
                st.session_state[k] = json_settings[k]
        return st.toast('Successfully uploaded data!', icon='💾')

    if uploaded_file is not None:
        button_apply_settings = st.button(
            label="Apply Settings",
            on_click=upload_json_settings,
            args=(uploaded_settings,),
            # help="Click to Apply the Settings of the Uploaded file.\\\n"
            #      "Please start by uploading a Settings File below"
        )
        
        


def download_upload_sidebar():
    """Create download/upload interface in sidebar."""
    with st.sidebar:
        with st.container():
            with st.expander(label="Save/reload data here!", expanded=False):
                download_upload_settings()


def modify_plot_text_sizes(fig):
    """Modify text label sizes in plotly figures.
    
    Args:
        fig: Plotly figure object to modify.
        
    Returns:
        Modified figure object.
    """
    fig.update_layout(
        legend=dict(font_size=15),
        legend_title=dict(font_size=25),
        margin=dict(l=10, r=10, t=25, b=10),
    )
    fig.update_traces(textfont_size=15)
    fig.update_xaxes(
        showline=True, 
        linewidth=2, 
        linecolor="grey", 
        mirror=True, 
        title_font=dict(size=25),
        tickfont=dict(size=20)
    )
    fig.update_yaxes(
        showline=True, 
        linewidth=2, 
        linecolor="grey", 
        mirror=True, 
        title_font=dict(size=25),
        tickfont=dict(size=20)
    )
    return fig


import dill  # Better than pickle for complex objects
import sqlite3
import uuid
import base64
from datetime import datetime


class SessionStateManager:
    """Manages saving and loading of Streamlit session states to/from SQLite database."""
    
    def __init__(self, db_path: str = "session_states.db"):
        """Initialize the session state manager.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._initialize_database()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database error: {e}") from e
        finally:
            if conn:
                conn.close()
    
    def _initialize_database(self) -> None:
        """Initialize the database with required tables."""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS session_states (
                    key TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            ''')
            conn.commit()
    
    def save_state(self, state_dict: Dict[str, Any], name: str, expires_days: int = 30) -> Optional[str]:
        """Save session state to database.
        
        Args:
            state_dict: Dictionary containing session state data.
            name: Human-readable name for the saved state.
            expires_days: Number of days until the saved state expires.
            
        Returns:
            Session key if successful, None otherwise.
        """
        session_key = str(uuid.uuid4())[:8]
        
        try:
            # Serialize and encode the data
            serialized_data = dill.dumps(state_dict)
            encoded_data = base64.b64encode(serialized_data)
            
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO session_states (key, name, data, created_at, expires_at)
                    VALUES (?, ?, ?, ?, datetime('now', '+{} days'))
                '''.format(expires_days), (session_key, name, encoded_data, datetime.now()))
                conn.commit()
            
            return session_key
            
        except Exception as e:
            st.error(f"Failed to save state: {e}")
            return None

    def list_states(self) -> List[Tuple[str, str, str]]:
        """Return all available session saves.
        
        Returns:
            List of tuples containing (key, name, created_at).
        """
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT key, name, created_at FROM session_states
                WHERE expires_at > datetime('now')
                ORDER BY created_at DESC
            ''')
            return cursor.fetchall()
    
    def load_state(self, session_key: str) -> Optional[Dict[str, Any]]:
        """Load session state from database.
        
        Args:
            session_key: Key of the session state to load.
            
        Returns:
            Dictionary containing session state data, or None if not found.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT data FROM session_states
                    WHERE key = ? AND expires_at > datetime('now')
                ''', (session_key,))
                result = cursor.fetchone()
            
            if result:
                encoded_data = result[0]
                serialized_data = base64.b64decode(encoded_data)
                return dill.loads(serialized_data)
            return None
            
        except Exception as e:
            st.error(f"Failed to load state: {e}")
            return None
    
    def delete_expired_states(self) -> int:
        """Delete expired session states.
        
        Returns:
            Number of deleted states.
        """
        with self._get_connection() as conn:
            cursor = conn.execute('''
                DELETE FROM session_states
                WHERE expires_at <= datetime('now')
            ''')
            conn.commit()
            return cursor.rowcount


#backward compatibility
check_folder = ensure_directory_exists
delete_data = delete_all_session_data
safe_json = is_json_serializable
modify_text_label_sizes = modify_plot_text_sizes
download_upload_sindebar = download_upload_sidebar