#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import base64
import io
from typing import List, Optional, Any

import pandas as pd
import streamlit as st
import plotly.express as px

from tools.authenticator import Authenticator
from tools.page_config import PageConfigManager
from tools.sidebar_logo import SidebarLogoManager
from tools.database import DatabaseManager
from tools.init_params import IsotopeStandards


class DataIOPage:
    """Manages the Data I/O page of the D4Xgui application."""

    def __init__(self):
        """Initialize the DataIOPage."""
        self.sss = st.session_state
        self.db_manager = DatabaseManager()
        self._setup_page()
        self._initialize_session_state()

    def _setup_page(self) -> None:
        """Set up page configuration, logo, and authentication."""
        page_config_manager = PageConfigManager()
        page_config_manager.configure_page(page_number=1)
        
        logo_manager = SidebarLogoManager()
        logo_manager.add_logo()
        
        if "PYTEST_CURRENT_TEST" not in os.environ:
            authenticator = Authenticator()
            if not authenticator.require_authentication():
                st.stop()

    def _initialize_session_state(self) -> None:
        """Initialize session state with default parameters if not present."""
        if 'standards_nominal' not in self.sss:
            self.sss['standards_nominal'] = IsotopeStandards.get_standards()
            #self.sss['working_gas'] = IsotopeStandards.get_working_gas()
        if 'show_overwrite_button' not in self.sss:
            self.sss.show_overwrite_button = False

    def run(self) -> None:
        """Run the main application page."""
        st.title("Data I/O")
        tab1, tab2 = st.tabs(['Upload data', 'Access database'])
        with tab1:
            self._upload_tab()
        with tab2:
            self._database_tab()

    def _upload_tab(self) -> None:
        """Render the data upload tab."""
        st.header("Upload Data")
        self._render_replicates_uploader()
        self._render_intensities_uploader()

    def _render_replicates_uploader(self) -> None:
        """Render the UI for uploading pre-processed replicates."""
        st.subheader("Pre-processed replicates")
        st.markdown(r"Drag & Drop pre-processed $\delta^{45-49}$ data!")
        
        col1, col2 = st.columns(2)
        with col1:
            uploaded_files = st.file_uploader(
                "Drag and Drop .csv or .xls(x) file(s)",
                accept_multiple_files=True,
                type=["xlsx", "xls", "csv"],
                key="uploaded_reps"
            )
        with col2:
            if st.button("Load test data (replicates)", key="loadTest"):
                self._load_test_files("reps")
            if st.button("Reset data", key="reset_preprocessed"):
                self._delete_data()

        if uploaded_files:
            self._read_preprocessed(uploaded_files)

        if "input_rep" in self.sss and not self.sss.input_rep.empty:
            self._handle_database_upload()
            self.sss.input_rep = self._modify_uploaded_df([self.sss.input_rep])[0]
            self._display_metrics(self.sss.input_rep)

    def _render_intensities_uploader(self) -> None:
        """Render the UI for uploading raw intensity data."""
        st.subheader("Raw intensity data")
        st.markdown("Drag & Drop sample and reference gas m/z$_{44-49}$ data!")

        col1, col2 = st.columns(2)
        with col1:
            uploaded_files = st.file_uploader(
                "Drag and Drop .csv, .xlsx or .did file(s)",
                accept_multiple_files=True,
                type=["xlsx", "xls", "csv", "did"],
                key="uploaded_cycles"
            )
        with col2:
            if st.button("Load test data (intensities)", key="loadTestRaw"):
                self._load_test_files("raw")
            if st.button("Reset raw data", key="reset_raw"):
                self._delete_data()

        if uploaded_files:
            self._read_intensities(uploaded_files)

        if "input_intensities" in self.sss and not self.sss.input_intensities.empty:
            self._display_metrics(self.sss.input_intensities, is_intensity_data=True)

    def _handle_database_upload(self) -> None:
        """Handle the logic for uploading data to the database."""
        if st.button("Upload Data to Database"):
            with self.db_manager.get_connection() as conn:
                existing = self._check_existing_timestamps(self.sss.input_rep, conn)
                self.sss.show_overwrite_button = True
                self.sss.existing_timestamps = existing
                if not existing:
                    session = self.sss.input_rep['Session'].iloc[0]
                    rows = self.db_manager.upsert_dataframe_with_conflict_resolution(self.sss.input_rep, session)
                    st.success(f"Data uploaded successfully. {rows} rows affected.")
                else:
                    st.warning(f"Found {len(existing)} existing timestamps in the database.")

        if self.sss.get('show_overwrite_button', False):
            if st.button("Confirm Overwrite"):
                with self.db_manager.get_connection() as conn:
                    session = self.sss.input_rep['Session'].iloc[0]
                    cursor = conn.cursor()
                    cursor.executemany(
                        "DELETE FROM replicates WHERE Timetag = ?",
                        [(str(ts),) for ts in self.sss.existing_timestamps]
                    )
                    rows = self.db_manager.upsert_dataframe_with_conflict_resolution(self.sss.input_rep, session)
                    conn.commit()
                    st.success(f"Data overwritten successfully. {rows} rows affected.")
                    self.sss.show_overwrite_button = False
                    self.sss.existing_timestamps = []

    def _database_tab(self) -> None:
        """Render the database access tab."""
        st.header("Access Database")
        st.write("Use the filters below to query the database.")

        sample_db = self._load_sample_database()
        
        all_df = self.db_manager.get_dataframe()
        all_sessions = sorted(all_df['Session'].unique().tolist()) if not all_df.empty else []
        all_samples = sorted(all_df['Sample'].unique().tolist()) if not all_df.empty else []

        col1, col2 = st.columns(2)
        with col1:
            selected_sessions = st.multiselect("Select Session(s)", all_sessions, default=all_sessions)
            selected_samples = st.multiselect("Select Sample(s)", all_samples)
        
        selected_project = []
        selected_type = []
        selected_mineralogy = []
        selected_publication = []
        
        if sample_db is not None:
            selected_project, selected_type, selected_mineralogy, selected_publication = self._render_sample_db_filters(sample_db, col1, col2)

        load_entire_sessions = st.checkbox("Load entire sessions when filtering", value=True)

        if st.button("Apply Filters"):
            self._apply_database_filters(
                selected_sessions, selected_samples, sample_db, load_entire_sessions,
                selected_project, selected_type, selected_mineralogy, selected_publication
            )

    def _load_sample_database(self) -> Optional[pd.DataFrame]:
        """Load the sample database from Excel."""
        sample_db_path = "static/SampleDatabase.xlsx"
        try:
            return pd.read_excel(sample_db_path, engine="openpyxl")
        except FileNotFoundError:
            st.warning(f"SampleDatabase not found at {sample_db_path}.")
            st.page_link("pages/97_Database_Management.py", label="→ Database Management", icon="🔗")
            return None

    def _render_sample_db_filters(self, sample_db: pd.DataFrame, col1, col2) -> tuple:
        """Render filters based on the sample database."""
        with self.db_manager.get_connection() as conn:
            replicates_df = pd.read_sql_query("SELECT * FROM replicates", conn)
        
        sub_sample_db = sample_db[sample_db['Sample'].isin(replicates_df['Sample'])]

        def get_split_values(column: str) -> List[str]:
            values = sub_sample_db[column].dropna().astype(str)
            split_values = [v.strip() for value in values for v in value.split(',')]
            return sorted(set(split_values))

        with col1:
            selected_project = st.multiselect("Select Project(s)", get_split_values("Project"))
        with col2:
            selected_type = st.multiselect("Select Type(s)", get_split_values("Type"))
            selected_mineralogy = st.multiselect("Select Mineralogy", get_split_values("Mineralogy"))
            selected_publication = st.multiselect("Select Publication", get_split_values("Publication"))
        
        return selected_project, selected_type, selected_mineralogy, selected_publication

    def _apply_database_filters(
        self, selected_sessions: List[str], selected_samples: List[str], 
        sample_db: Optional[pd.DataFrame], load_entire_sessions: bool,
        selected_project: List[str], selected_type: List[str],
        selected_mineralogy: List[str], selected_publication: List[str]
    ) -> None:
        """Apply filters and display data from the database."""
        query = "SELECT * FROM replicates WHERE 1=1"
        params = []
        if selected_sessions:
            query += f" AND Session IN ({','.join(['?'] * len(selected_sessions))})"
            params.extend(selected_sessions)
        if selected_samples:
            query += f" AND Sample IN ({','.join(['?'] * len(selected_samples))})"
            params.extend(selected_samples)

        with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if sample_db is not None:
            df = self._filter_by_sample_db(df, sample_db, selected_project, selected_type, selected_mineralogy, selected_publication)

        if load_entire_sessions and not df.empty:
            sessions_to_load = df['Session'].unique()
            query = f"SELECT * FROM replicates WHERE Session IN ({','.join(['?'] * len(sessions_to_load))})"
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=sessions_to_load.tolist())
        
        self.sss.input_rep = self._modify_uploaded_df([df])[0]
        
        if not df.empty:
            self._display_filtered_data(df, sample_db)
        else:
            st.info("No entries found with your current filter settings.")

    def _filter_by_sample_db(self, df: pd.DataFrame, sample_db: pd.DataFrame, 
                           selected_project: List[str], selected_type: List[str],
                           selected_mineralogy: List[str], selected_publication: List[str]) -> pd.DataFrame:
        """Filter DataFrame based on selections from the sample database."""
        filtered_sample_db = sample_db.copy()
        
        def filter_by_keywords(df_to_filter, column, keywords):
            if not keywords:
                return df_to_filter
            pattern = '|'.join([re.escape(k.strip().lower()) for k in keywords])
            return df_to_filter[df_to_filter[column].fillna('').str.lower().str.contains(pattern, regex=True)]
        
        for field, selected in [('Project', selected_project), ('Type', selected_type),
                                ('Mineralogy', selected_mineralogy), ('Publication', selected_publication)]:
            if selected:
                filtered_sample_db = filter_by_keywords(filtered_sample_db, field, selected)
        
        filtered_samples = filtered_sample_db['Sample'].unique()
        return df[df['Sample'].isin(filtered_samples)]

    @staticmethod
    def _read_file(file: Any) -> pd.DataFrame:
        """Read a file and return a DataFrame."""
        filename = file.name
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(file, engine="openpyxl")
        elif filename.endswith((".csv", ".txt")):
            stringio = io.StringIO(file.getvalue().decode("utf-8"))
            STR = stringio.read()
            if '\t' in STR:
                SEP = '\t'
            elif ';' in STR:
                SEP = ';'
            elif ',' in STR:
                SEP = ','
            else:
                
                SEP = None
            st.info('Could not recognize column separator, please use `\\t`   `;`  or  `,` !')
            df=pd.read_csv(file, sep=SEP)
            return df
        raise ValueError(f"Unsupported file type: {filename}")

    def _delete_data(self) -> None:
        """Clear uploaded data from session state."""
        for key in ['input_intensities', 'raw_files', 'input_rep','raw_data', 'scaling_factors', 'standards', 'correction_output_full_dataset','_06_filtered_reps','_06_filtered_summary','bg_success','correction_output_summary']:
            if key in self.sss:
                del self.sss[key]

    def _modify_uploaded_df(self, dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Clean and modify uploaded DataFrames."""
        for df in dfs:
            df.drop_duplicates(inplace=True)
            if "Sample" not in df.columns:
                df["Sample"] = ""
            df["Sample"] = df["Sample"].astype(str)

            # Sanitize sample names
            for char in ("+", ":", "(", ")", "&"):
                if df["Sample"].str.contains(char, regex=False).any():
                    st.info(f"Sample names must not contain `{char}` --> removed!")
                    df["Sample"] = df["Sample"].str.replace(char, "", regex=False)
            
            # Standardize sample names
            rename_dict = {
                **{f"ETH {i}": f"ETH-{i}" for i in range(1, 5)},
                **{f"ETH{i}": f"ETH-{i}" for i in range(1, 5)},
                "25G": "25C", "EG": "25C", "HG": "1000C", 
                "Heated": "1000C", "GU-1": "GU1"
            }
            df["Sample"].replace(rename_dict, inplace=True)

            for key in ('Session', 'Sample'):
                if key in df.columns:
                    df[key] = df[key].astype('string')

            if "Timetag" not in df.columns:
                datetime_columns = [key for key in df.columns if key.lower() in ("date", "time", "datetime")]
                if datetime_columns:
                    df.rename(columns={datetime_columns[0]: "Timetag"}, inplace=True)
                else:
                    st.error("No Timetag column provided! An artificial Timetag is created.")
                    d0 = pd.to_datetime('1900-01-01 00:00:00')
                    df['Timetag'] = [d0 + pd.Timedelta(days=i) for i in range(len(df))]

            for col in ("Outlier", "Project", "Type"):
                if col not in df.columns:
                    df[col] = None
        return dfs

    def _load_test_files(self, mode: str) -> None:
        """Load example data files."""
        self.sss.raw_files = mode != "reps"
        path = (
            "static/exampleReplicates_anonymized.xlsx" if mode == "reps" 
            else "static/exampleIntensities_anonymized.xlsx"
        )
        df = pd.read_excel(path, engine="openpyxl")
        input_data = self._modify_uploaded_df([df])[0]
        
        if mode == "reps":
            self.sss.input_rep = input_data
        else:
            self.sss.input_intensities = input_data

    def _read_intensities(self, uploaded_files: List[Any]) -> None:
        """Read and process raw intensity data files."""
        self.sss.raw_files = True
        df_list = []
        
        with st.spinner(text="Uploading..."):
            for idx, file in enumerate(uploaded_files):
                new_df = self._read_file(file)
                for col in ["Session", "Type", "Project"]:
                    if col not in new_df.columns:
                        new_df[col] = idx
                df_list.append(new_df)
        
        df = pd.concat(df_list, ignore_index=True)

        if "UID" in df and not df["UID"].is_unique:
            st.error("Non-unique UIDs found! Please provide unique identifiers.")
            st.stop()

        columns_to_keep = [
            "UID", "Sample", "Replicate", "Session", "Timetag",
            "raw_s44", "raw_s45", "raw_s46", "raw_s47", "raw_s48", "raw_s49",
            "raw_r44", "raw_r45", "raw_r46", "raw_r47", "raw_r48", "raw_r49"
        ]
        
        for mz in (47, 48):
            if f"raw_s{mz}.5" in df.columns:
                columns_to_keep.extend([f"raw_s{mz}.5", f"raw_r{mz}.5"])
            else:
                self.sss[f"half-mass-cup{mz}"] = False

        df = self._modify_uploaded_df([df])[0]
        existing_cols = [c for c in columns_to_keep if c in df.columns]
        if len(existing_cols) < 5:
            st.error("Uploaded intensity file(s) are missing required columns. Please provide raw_s44-49 and raw_r44-49 with Session, Sample, Replicate, Timetag.")
            st.stop()
        self.sss.input_intensities = df.loc[:, existing_cols]

    def _read_preprocessed(self, uploaded_files: List[Any]) -> None:
        """Read and process pre-processed replicate files."""
        self.sss.raw_files = False
        df_list = [self._read_file(f) for f in uploaded_files]
        df = pd.concat(df_list, ignore_index=True)
        
        df = self._modify_uploaded_df([df])[0]

        if "UID" in df and not df["UID"].is_unique:
            st.error("Non-unique UIDs found! Please provide unique identifiers.")
            st.stop()

        required = ["UID", "d45", "d46", "d47", "d48", "d49", "Sample", "Session"]
        missing = [k for k in required if k not in df.columns]
        if missing:
            st.error(f"Please provide all required columns: {', '.join(missing)}")
            st.stop()

        self.sss.input_rep = df
        st.info("Data loaded. Click 'Upload Data to Database' to save it.")

    def _display_metrics(self, df: pd.DataFrame, is_intensity_data: bool = False) -> None:
        """Display metrics and charts for the loaded data."""
        st.dataframe(self._calculate_metrics(df, is_intensity_data), width="stretch")
        
        time_col = 'Timetag' if "Timetag" in df.columns else 'datetime'
        if is_intensity_data and "Replicate" in df.columns:
            dist_samples = {
                "Sample": df.groupby("Replicate")["Sample"].first(),
                "Datetime": df.groupby("Replicate")[time_col].first(),
            }
        else:
            dist_samples = {
                "Sample": df["Sample"],
                "Datetime": df[time_col],
            }
        st.scatter_chart(pd.DataFrame(dist_samples), y="Sample", x="Datetime", #width="stretch"
        )

    def _calculate_metrics(self, df: pd.DataFrame, is_intensity_data: bool) -> pd.DataFrame:
        """Calculate metrics from the DataFrame."""
        metrics = []
        time_col = 'Timetag' if "Timetag" in df.columns else 'datetime'
        for session, sdf in df.groupby("Session"):
            sdf["Sample"] = sdf["Sample"].astype(str)
            session_metrics = {
                "Session": session,
                "Duration": f"{sdf[time_col].min()} - {sdf[time_col].max()}",
                "Unique Samples": sdf["Sample"].nunique(),
                "Replicates": sdf["Replicate"].nunique() if is_intensity_data else len(sdf),
                "Samples": sorted(sdf["Sample"].unique()),
            }
            if is_intensity_data:
                session_metrics["Cycles"] = len(sdf)
            metrics.append(session_metrics)
        return pd.DataFrame(metrics)

    def _check_existing_timestamps(self, df: pd.DataFrame, conn: Any) -> List[Any]:
        """Check for existing timestamps in the database."""
        cursor = conn.cursor()
        existing = []
        for timestamp in df['Timetag']:
            cursor.execute("SELECT COUNT(*) FROM replicates WHERE Timetag = ?", (str(timestamp),))
            if cursor.fetchone()[0] > 0:
                existing.append(timestamp)
        return existing

    def _display_filtered_data(self, df: pd.DataFrame, sample_db: Optional[pd.DataFrame]) -> None:
        """Display metrics and data for filtered database results."""
        st.subheader("Data Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Replicates", len(df))
            st.metric("Unique Samples", df['Sample'].nunique(), help=' | '.join(sorted(df['Sample'].unique())))
            st.metric("Unique Sessions", df['Session'].nunique(), help=' | '.join(sorted(df['Session'].unique())))
        with col2:
            tt_series = pd.to_datetime(df['Timetag'], errors='coerce')
            min_dt = tt_series.min()
            max_dt = tt_series.max()
            if pd.isna(min_dt) or pd.isna(max_dt):
                date_range = "—"
            else:
                date_range = f"{min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}"
            st.metric("Date Range", date_range)
            if sample_db is not None:
                self._display_sample_db_metrics(df, sample_db)

        st.markdown(self._create_excel_download_link(df, "filtered_data.xlsx"), unsafe_allow_html=True)
        
        st.subheader("Sample Distribution")
        sample_counts = df['Sample'].value_counts()
        fig = px.bar(x=sample_counts.index, y=sample_counts.values, labels={'x': 'Sample', 'y': 'Count'})
        st.plotly_chart(fig)
        
        st.subheader("Selected Data")
        st.dataframe(df)

    def _display_sample_db_metrics(self, df: pd.DataFrame, sample_db: pd.DataFrame) -> None:
        """Display metrics derived from the sample database."""
        filtered_samples = df['Sample'].unique()
        sub_sample_db = sample_db[sample_db['Sample'].isin(filtered_samples)]
        
        def count_unique_split(column: str) -> int:
            return sub_sample_db[column].str.split(',').explode().str.strip().nunique()
            
        st.metric("Total Projects", count_unique_split('Project'))
        st.metric("Total Types", count_unique_split('Type'))

    def _create_excel_download_link(self, df: pd.DataFrame, filename: str) -> str:
        """Generate a download link for an Excel file."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        b64 = base64.b64encode(output.getvalue()).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel File</a>'


if __name__ == "__main__":
    page = DataIOPage()
    page.run()