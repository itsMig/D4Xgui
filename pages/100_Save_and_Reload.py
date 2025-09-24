#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sqlite3
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st

from tools.commons import SessionStateManager
from tools.authenticator import Authenticator


class SaveReloadPage:
    """Manages the Save & Reload page of the D4Xgui application."""

    def __init__(self):
        """Initialize the SaveReloadPage."""
        self.sss = st.session_state
        self.state_manager = SessionStateManager()
        self._setup_page()

    def _setup_page(self) -> None:
        """Set up authentication for the page."""
        if "PYTEST_CURRENT_TEST" not in os.environ:
            authenticator = Authenticator()
            if not authenticator.require_authentication():
                st.stop()

    def run(self) -> None:
        """Run the main application page."""
        st.title("Save & Reload")
        self._render_save_section()
        self._render_dashboard_section()

    def _render_save_section(self) -> None:
        """Render the section for saving current session state."""
        st.subheader("💾 Save current dataset")
        state_name = st.text_input("Name this dataset:", "")
        
        if st.button("Save Session State") and state_name.strip():
            self._save_current_state(state_name.strip())

    def _render_dashboard_section(self) -> None:
        """Render the dashboard section showing saved states."""
        st.subheader("📊 Saved Dataset Dashboard")
        available_states = self.state_manager.list_states()
        
        if available_states:
            self._render_available_states(available_states)
        else:
            st.info("⚠️ No saved sessions found.")

    def _save_current_state(self, state_name: str) -> None:
        """Save the current session state with the given name.
        
        Args:
            state_name: The name to save the state under
        """
        try:
            state_dict = dict(self.sss)
            session_key = self.state_manager.save_state(state_dict, state_name)
            
            if session_key:
                st.success(f"✅ Dataset saved as '{state_name}'")
                st.caption(f"(Key: {session_key})")
            else:
                st.error("Failed to save dataset. Please try again.")
        except Exception as e:
            st.error(f"Error saving dataset: {str(e)}")

    def _render_available_states(self, available_states: List[Tuple[str, str, str]]) -> None:
        """Render the list of available saved states.
        
        Args:
            available_states: List of tuples containing (key, name, created)
        """
        st.write(f"**Total datasets available: {len(available_states)}**")
        
        for key, name, created in sorted(available_states):
            self._render_state_item(key, name, created)

    def _render_state_item(self, key: str, name: str, created: str) -> None:
        """Render a single saved state item.
        
        Args:
            key: The unique key for the saved state
            name: The display name of the saved state
            created: The creation timestamp
        """
        try:
            raw_state = self.state_manager.load_state(key)
            summary = self._summarize_state(raw_state) if raw_state else {}
            
            time_interval = self._format_time_interval(summary)
            
            with st.expander(f"📂 {name} — {time_interval}"):
                self._render_state_metrics(summary, created)
                self._render_state_actions(key, name, raw_state)
                
        except Exception as e:
            with st.expander(f"📂 {name} — Error loading summary"):
                st.error(f"Error loading state summary: {str(e)}")
                self._render_state_actions(key, name, None)

    def _render_state_metrics(self, summary: Dict[str, Any], created: str) -> None:
        """Render metrics for a saved state.
        
        Args:
            summary: Dictionary containing state summary metrics
            created: Creation timestamp string
        """
        cols = st.columns(4)
        cols[0].metric("Uploaded:", f"{created[:19]}")
        cols[1].metric("Sessions", summary.get('n_sessions', '—'))
        cols[2].metric("Samples", summary.get('n_sample', '—'))
        cols[3].metric("Replicates", summary.get('n_replicates', '—'))

    def _render_state_actions(self, key: str, name: str, raw_state: Optional[Dict[str, Any]]) -> None:
        """Render action buttons for a saved state.
        
        Args:
            key: The unique key for the saved state
            name: The display name of the saved state
            raw_state: The loaded state data, or None if failed to load
        """
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Load '{name}'", key=f"load_{key}"):
                self._load_state(name, raw_state)
        
        with col2:
            if st.button(f"Delete '{name}'", key=f"delete_{key}"):
                self._delete_state(key, name)

    def _load_state(self, name: str, raw_state: Optional[Dict[str, Any]]) -> None:
        """Load a saved state into the current session.
        
        Args:
            name: The display name of the state
            raw_state: The state data to load
        """
        if raw_state:
            try:
                # Update session state with loaded data
                for key, value in raw_state.items():
                    if not (key.startswith("load_") or key.startswith("delete_")):
                
                        self.sss[key] = value
                st.toast(f"Dataset '{name}' loaded ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading dataset '{name}': {str(e)}")
        else:
            st.error(f"Failed to load dataset '{name}'. Data may be corrupted.")

    def _delete_state(self, key: str, name: str) -> None:
        """Delete a saved state from the database.
        
        Args:
            key: The unique key for the saved state
            name: The display name of the saved state
        """
        try:
            with sqlite3.connect(self.state_manager.db_path) as conn:
                conn.execute("DELETE FROM session_states WHERE key = ?", (key,))
                conn.commit()
            
            st.warning(f"Deleted session '{name}'")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting session '{name}': {str(e)}")

    def _summarize_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary metrics from a session state dictionary.
        
        Args:
            state_dict: The session state dictionary to summarize
            
        Returns:
            Dictionary containing summary metrics
        """
        summary = {}
        
        if not state_dict or 'input_rep' not in state_dict:
            return summary
            
        try:
            input_rep = state_dict['input_rep']
            
            # Ensure input_rep has the expected structure
            if hasattr(input_rep, 'get') and callable(input_rep.get):
                # Handle DataFrame-like objects
                if hasattr(input_rep, 'nunique') and callable(input_rep.nunique):
                    summary.update({
                        'n_sample': len(input_rep['Sample'].unique()) if 'Sample' in input_rep else 0,
                        'n_replicates': len(input_rep['Timetag'].unique()) if 'Timetag' in input_rep else 0,
                        'n_sessions': len(input_rep['Session'].unique()) if 'Session' in input_rep else 0,
                    })
                    
                    if 'Timetag' in input_rep:
                        summary.update({
                            'from': input_rep['Timetag'].min(),
                            'to': input_rep['Timetag'].max(),
                        })
        except (KeyError, AttributeError, TypeError) as e:
            # Log the error but don't fail completely
            st.warning(f"Could not extract complete summary: {str(e)}")
            
        return summary

    def _format_time_interval(self, summary: Dict[str, Any]) -> str:
        """Format the time interval string for display.
        
        Args:
            summary: Dictionary containing summary metrics
            
        Returns:
            Formatted time interval string
        """
        if 'from' in summary and 'to' in summary:
            return f"Time interval: {summary['from']} – {summary['to']}"
        return "Time interval: Not available"


if __name__ == "__main__":
    page = SaveReloadPage()
    page.run()