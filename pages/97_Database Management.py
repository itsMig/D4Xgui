#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
from typing import Optional, List, Tuple, Any

import pandas as pd
import streamlit as st

from tools.authenticator import Authenticator
from tools.page_config import PageConfigManager
from tools.database import DatabaseManager


class DatabaseManagementPage:
    """Manages the Database Management page of the D4Xgui application."""

    SAMPLE_DB_PATH = "static/SampleDatabase.xlsx"
    
    def __init__(self):
        """Initialize the DatabaseManagementPage."""
        self.sss = st.session_state
        self.db_manager = DatabaseManager()
        self._setup_page()

    def _setup_page(self) -> None:
        """Set up page configuration and authentication."""
        page_config_manager = PageConfigManager()
        page_config_manager.configure_page(page_number=97)
        
        if "PYTEST_CURRENT_TEST" not in os.environ:
            authenticator = Authenticator()
            if not authenticator.require_authentication():
                st.stop()

    def run(self) -> None:
        """Run the main database management page."""
        st.title("Database Management Dashboard")
        
        self._initialize_database()
        
        tab1, tab2 = st.tabs(["Replicates Database", "Sample Metadata"])
        
        with tab1:
            self._render_replicates_tab()
        with tab2:
            self._render_sample_metadata_tab()

    def _initialize_database(self) -> None:
        """Initialize the database if not already initialized."""
        if not self.db_manager.is_initialized():
            st.warning("Database not initialized. Initializing now...")
            self.db_manager.initialize()
            st.success("Database initialized.")

    def _render_replicates_tab(self) -> None:
        """Render the replicates database management tab."""
        df = self.db_manager.get_dataframe()
        
        if df.empty:
            self._render_empty_database_info()
        else:
            self._render_database_content(df)

    def _render_empty_database_info(self) -> None:
        """Display information when the database is empty."""
        st.info("No data in the database.")
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            st.write(f"Tables in the database: {tables}")
            
            if 'replicates' in tables:
                cursor.execute("SELECT COUNT(*) FROM replicates;")
                count = cursor.fetchone()[0]
                st.write(f"Number of rows in replicates table: {count}")
            else:
                st.error("replicates table does not exist in the database.")

    def _render_database_content(self, df: pd.DataFrame) -> None:
        """Render the main database content interface."""
        st.write(f"Number of rows in database: {len(df)}")
        st.write(f"Columns: {df.columns.tolist()}")
        
        sample_db = self._load_sample_database()
        filtered_df = self._apply_filters(df, sample_db)
        
        self._render_data_display(filtered_df)
        #self._render_data_manipulation(filtered_df)
        self._render_database_deletion()

    def _load_sample_database(self) -> Optional[pd.DataFrame]:
        """Load and preprocess the sample database."""
        if not os.path.exists(self.SAMPLE_DB_PATH):
            return None
            
        try:
            sample_db = pd.read_excel(self.SAMPLE_DB_PATH, engine="openpyxl")
            # Normalize string columns to lowercase for consistent filtering
            for col in ["Type", "Project", "Mineralogy", "Publication"]:
                if col in sample_db.columns:
                    sample_db[col] = sample_db[col].astype(str).str.lower()
            return sample_db
        except Exception as e:
            st.error(f"Error loading sample database: {e}")
            return None

    def _apply_filters(self, df: pd.DataFrame, sample_db: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Apply user-selected filters to the dataframe."""
        st.sidebar.header("Data Filtering")
        
        filter_mode = st.sidebar.toggle("Select filter functionality", value=False)
        
        if filter_mode and sample_db is not None:
            return self._apply_advanced_filters(df, sample_db)
        else:
            return self._apply_simple_filters(df)

    def _apply_advanced_filters(self, df: pd.DataFrame, sample_db: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced filters based on sample database metadata."""
        # Filter sample_db to only include samples present in the replicates table
        sample_db = sample_db[sample_db["Sample"].isin(df["Sample"].unique())]
        
        with st.sidebar:
            selected_projects = st.multiselect(
                "Project:", 
                sorted(sample_db["Project"].dropna().unique())
            )
            selected_types = st.multiselect(
                "Sample type:", 
                sorted(sample_db["Type"].dropna().unique())
            )
            selected_publications = st.multiselect(
                "Publication:", 
                sorted(sample_db["Publication"].dropna().unique())
            )
            selected_mineralogy = st.multiselect(
                "Mineralogy:", 
                sorted(sample_db["Mineralogy"].dropna().unique())
            )
            selected_sessions = st.multiselect(
                "Session:", 
                sorted(df["Session"].unique())
            )
            
            # Date filtering
            df['Timetag'] = pd.to_datetime(df['Timetag'])
            start_date, end_date = self._render_date_filters(df)
            
            load_entire_sessions = st.checkbox(
                "Load entire sessions when filtering", 
                value=True
            )
        
        # Apply filters sequentially
        filtered_df = self._filter_by_sample_metadata(
            df, sample_db, selected_projects, selected_types, 
            selected_publications, selected_mineralogy
        )
        
        if selected_sessions:
            filtered_df = filtered_df[filtered_df["Session"].isin(selected_sessions)]
        
        filtered_df = filtered_df[
            (filtered_df["Timetag"].dt.date >= start_date) & 
            (filtered_df["Timetag"].dt.date <= end_date)
        ]
        
        if load_entire_sessions and not filtered_df.empty:
            sessions_to_include = filtered_df["Session"].unique()
            filtered_df = self.db_manager.get_dataframe()
            filtered_df = filtered_df[filtered_df["Session"].isin(sessions_to_include)]
        
        return filtered_df

    def _apply_simple_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply simple text-based filters."""
        with st.sidebar:
            sample_contains = st.text_input(
                "Sample name contains (KEEP):",
                help="Set multiple keywords by separating them through semicolons ;"
            )
            sample_not_contains = st.text_input(
                "Sample name contains (DROP):",
                help="Set multiple keywords by separating them through semicolons ;"
            )
        
        filtered_df = df.copy()
        
        if sample_contains:
            pattern = "|".join(sample_contains.split(";"))
            filtered_df = filtered_df[
                filtered_df["Sample"].str.contains(pattern, case=False, na=False)
            ]
        
        if sample_not_contains:
            pattern = "|".join(sample_not_contains.split(";"))
            filtered_df = filtered_df[
                ~filtered_df["Sample"].str.contains(pattern, case=False, na=False)
            ]
        
        return filtered_df

    def _filter_by_sample_metadata(
        self, df: pd.DataFrame, sample_db: pd.DataFrame,
        selected_projects: List[str], selected_types: List[str],
        selected_publications: List[str], selected_mineralogy: List[str]
    ) -> pd.DataFrame:
        """Filter dataframe based on sample metadata selections."""
        filtered_df = df.copy()
        
        filter_mappings = [
            (selected_projects, "Project"),
            (selected_types, "Type"),
            (selected_publications, "Publication"),
            (selected_mineralogy, "Mineralogy")
        ]
        
        for selected_values, column in filter_mappings:
            if selected_values:
                matching_samples = sample_db[
                    sample_db[column].isin(selected_values)
                ]["Sample"]
                filtered_df = filtered_df[
                    filtered_df["Sample"].isin(matching_samples)
                ]
        
        return filtered_df

    def _render_date_filters(self, df: pd.DataFrame) -> Tuple[Any, Any]:
        """Render date filter controls."""
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", df["Timetag"].min())
        with col2:
            end_date = st.date_input("End Date", df["Timetag"].max())
        
        return start_date, end_date

    def _render_data_display(self, df: pd.DataFrame) -> None:
        """Render the main data display section."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Filtered Data")
            st.dataframe(df)
        
        with col2:
            st.subheader("Table Statistics")
            st.metric("Total Replicates", len(df))
            st.metric("Unique Sample Names", df["Sample"].nunique())
            
            excel_data = self._create_excel_download(df, "replicates_filtered.xlsx")
            st.download_button(
                label="Download Filtered Excel",
                data=excel_data,
                file_name="replicates_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    def _create_excel_download(self, df: pd.DataFrame, filename: str) -> bytes:
        """Create Excel file data for download."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)
        return output.getvalue()

    def _render_data_manipulation(self, df: pd.DataFrame) -> None:
        """Render the data manipulation section."""
        if df.empty:
            return
            
        with st.expander("Data Manipulation"):
            st.subheader("Data Manipulation")
            
            row_to_manipulate = st.number_input(
                "Enter row index to manipulate:", 
                min_value=0, 
                max_value=len(df) - 1
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Delete Row"):
                    self._delete_row(df, row_to_manipulate)
            
            with col2:
                if st.button("Mark as Outlier"):
                    self._mark_outlier(df, row_to_manipulate, True)
            
            with col3:
                if st.button("Unmark Outlier"):
                    self._mark_outlier(df, row_to_manipulate, False)
            
            st.subheader("Selected Row")
            if row_to_manipulate < len(df):
                st.dataframe(df.iloc[[row_to_manipulate]])

    def _delete_row(self, df: pd.DataFrame, row_index: int) -> None:
        """Delete a specific row from the database."""
        try:
            index_to_delete = df.index[row_index]
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Get primary key column name
                cursor.execute("PRAGMA table_info(replicates)")
                primary_key = cursor.fetchall()[0][1]
                
                cursor.execute(
                    f"DELETE FROM replicates WHERE {primary_key} = ?",
                    (df.loc[index_to_delete, primary_key],)
                )
                conn.commit()
                
            st.success(f"Row {row_index} deleted successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting row: {e}")

    def _mark_outlier(self, df: pd.DataFrame, row_index: int, is_outlier: bool) -> None:
        """Mark or unmark a row as an outlier."""
        try:
            index_to_modify = df.index[row_index]
            outlier_value = 1 if is_outlier else 0
            action = "marked" if is_outlier else "unmarked"
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE replicates SET Outlier = ? WHERE UID = ?",
                    (outlier_value, df.loc[index_to_modify, 'UID'])
                )
                conn.commit()
                
            st.success(f"Row {row_index} {action} as outlier successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error {'marking' if is_outlier else 'unmarking'} outlier: {e}")

    def _render_database_deletion(self) -> None:
        """Render the database deletion section."""
        with st.expander('Delete Database'):
            st.warning(
                "⚠️ Warning: This action will permanently delete the entire database. "
                "This cannot be undone."
            )
            
            confirm_checkbox = st.checkbox(
                "I understand the consequences and want to proceed with deletion",
                value=False
            )
            
            if confirm_checkbox:
                if st.button(
                    "Permanently Delete Database", 
                    type="primary", 
                    help="This will delete all data"
                ):
                    self._delete_database()

    def _delete_database(self) -> None:
        """Delete the entire database and reinitialize."""
        try:
            # Close any existing connections
            if hasattr(self.db_manager, '_connection'):
                self.db_manager._connection.close()
            
            db_path = 'pre_replicates.db'
            if os.path.exists(db_path):
                os.remove(db_path)
                st.success("Database successfully deleted.")
                
                # Reinitialize the database
                self.db_manager.initialize()
                st.info("A new empty database has been initialized.")
                
                st.rerun()
            else:
                st.error("Database file not found.")
        except Exception as e:
            st.error(f"An error occurred while deleting the database: {e}")

    def _render_sample_metadata_tab(self) -> None:
        """Render the sample metadata management tab."""
        st.header("Sample Metadata Management")
        
        if os.path.exists(self.SAMPLE_DB_PATH):
            self._render_existing_sample_database()
        else:
            self._render_create_sample_database()
        
        self._render_manual_sample_addition()

    def _render_existing_sample_database(self) -> None:
        """Render interface for existing sample database."""
        try:
            sample_db = pd.read_excel(self.SAMPLE_DB_PATH, engine="openpyxl")
            
            st.subheader("Current Sample Database")
            st.dataframe(sample_db)
            
            self._render_sample_database_upload("Update Sample Database", "Replace Current Sample Database")
        except Exception as e:
            st.error(f"Error loading existing sample database: {e}")

    def _render_create_sample_database(self) -> None:
        """Render interface for creating a new sample database."""
        st.info("No Sample Database found. Please upload a new one.")
        self._render_sample_database_upload("Upload Sample Database", "Create Sample Database")

    def _render_sample_database_upload(self, upload_label: str, button_label: str) -> None:
        """Render the sample database upload interface."""
        st.subheader(upload_label)
        uploaded_file = st.file_uploader(
            f"{upload_label} (Excel format)", 
            type=["xlsx"]
        )
        
        if uploaded_file is not None:
            try:
                new_sample_db = pd.read_excel(uploaded_file, engine="openpyxl")
                
                st.subheader("New Sample Database Preview")
                st.dataframe(new_sample_db)
                
                if st.button(button_label):
                    self._save_sample_database(new_sample_db)
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

    def _save_sample_database(self, sample_db: pd.DataFrame) -> None:
        """Save the sample database to file."""
        try:
            os.makedirs("static", exist_ok=True)
            sample_db.to_excel(self.SAMPLE_DB_PATH, index=False, engine="openpyxl")
            st.success("Sample database updated successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error saving sample database: {e}")

    def _render_manual_sample_addition(self) -> None:
        """Render the manual sample addition interface."""
        with st.expander("Add New Sample Metadata Manually"):
            st.subheader("Add New Sample")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_sample_name = st.text_input("Sample Name")
                new_sample_type = st.text_input("Sample Type")
                new_sample_project = st.text_input("Project")
            
            with col2:
                new_sample_mineralogy = st.text_input("Mineralogy")
                new_sample_publication = st.text_input("Publication")
                new_sample_notes = st.text_area("Notes")
            
            if st.button("Add Sample to Database"):
                self._add_sample_manually(
                    new_sample_name, new_sample_type, new_sample_project,
                    new_sample_mineralogy, new_sample_publication, new_sample_notes
                )

    def _add_sample_manually(
        self, name: str, sample_type: str, project: str,
        mineralogy: str, publication: str, notes: str
    ) -> None:
        """Add a new sample to the database manually."""
        if not name:
            st.warning("Sample Name is required.")
            return
        
        try:
            sample_data = {
                "Sample": [name],
                "Type": [sample_type],
                "Project": [project],
                "Mineralogy": [mineralogy],
                "Publication": [publication],
                "Notes": [notes]
            }
            
            if os.path.exists(self.SAMPLE_DB_PATH):
                existing_db = pd.read_excel(self.SAMPLE_DB_PATH, engine="openpyxl")
                
                if name in existing_db["Sample"].values:
                    st.warning(f"Sample '{name}' already exists in the database.")
                    return
                
                new_row = pd.DataFrame(sample_data)
                updated_db = pd.concat([existing_db, new_row], ignore_index=True)
            else:
                updated_db = pd.DataFrame(sample_data)
            
            self._save_sample_database(updated_db)
            st.success(f"Sample '{name}' added successfully!")
        except Exception as e:
            st.error(f"Error adding sample to database: {e}")


if __name__ == "__main__":
    page = DatabaseManagementPage()
    page.run()