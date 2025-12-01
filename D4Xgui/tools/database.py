"""Database management module for D4Xgui application.

This module provides database operations for storing and retrieving replicate data
using SQLite as the backend database.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import streamlit as st


class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass


class DatabaseManager:
    """Manages SQLite database operations for replicate data."""
    
    # Database configuration
    DEFAULT_DB_NAME = 'pre_replicates.db'
    TABLE_NAME = 'replicates'
    
    # Required columns for the replicates table
    REQUIRED_COLUMNS = ['Sample', 'Session', 'Timetag', 'd45', 'd46', 'd47', 'd48', 'd49']
    OPTIONAL_COLUMNS = ['Outlier', 'Project', 'Type']
    
    # SQL queries
    CREATE_TABLE_SQL = f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            UID INTEGER PRIMARY KEY AUTOINCREMENT,
            Sample TEXT NOT NULL,
            Session TEXT NOT NULL,
            Timetag DATETIME UNIQUE NOT NULL,
            d45 REAL,
            d46 REAL,
            d47 REAL,
            d48 REAL,
            d49 REAL,
            Outlier TEXT,
            Project TEXT,
            Type TEXT
        )
    '''
    
    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_NAME):
        """Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._ensure_initialized()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup.
        
        Yields:
            sqlite3.Connection: Database connection.
        """
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
    
    def _ensure_initialized(self) -> None:
        """Ensure the database is initialized with required tables."""
        if not self.is_initialized():
            self.initialize()
    
    def initialize(self) -> None:
        """Initialize the database with required tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self.CREATE_TABLE_SQL)
            conn.commit()
    
    def is_initialized(self) -> bool:
        """Check if the database is initialized.
        
        Returns:
            True if the replicates table exists, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (self.TABLE_NAME,)
                )
                return cursor.fetchone() is not None
        except DatabaseError:
            return False
    
    def delete_table(self) -> None:
        """Delete the replicates table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'DROP TABLE IF EXISTS {self.TABLE_NAME}')
            conn.commit()

    def upsert_dataframe(self, df: pd.DataFrame, session: str) -> int:
        """Insert or update DataFrame data in the database.
        
        Args:
            df: DataFrame containing replicate data.
            session: Session identifier for the data.
            
        Returns:
            Number of rows affected.
        """
        # Validate required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter DataFrame to include only existing columns
        existing_columns = self.REQUIRED_COLUMNS + [
            col for col in self.OPTIONAL_COLUMNS if col in df.columns
        ]
        df_filtered = df[existing_columns].copy()
        df_filtered['Timetag'] = df_filtered['Timetag'].astype(str)
        
        rows_updated = 0
        rows_inserted = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in df_filtered.iterrows():
                # Check if row exists
                cursor.execute(
                    f"SELECT UID FROM {self.TABLE_NAME} WHERE Timetag = ?",
                    (row['Timetag'],)
                )
                existing_row = cursor.fetchone()
                
                if existing_row:
                    # Update existing row
                    update_cols = ', '.join([
                        f"{col} = ?" for col in existing_columns if col != 'Timetag'
                    ])
                    update_query = f"""
                        UPDATE {self.TABLE_NAME} SET {update_cols}
                        WHERE UID = ?
                    """
                    update_values = [row[col] for col in existing_columns if col != 'Timetag']
                    update_values.append(existing_row[0])
                    cursor.execute(update_query, update_values)
                    rows_updated += 1
                else:
                    # Insert new row
                    insert_cols = ', '.join(existing_columns)
                    insert_placeholders = ', '.join(['?' for _ in existing_columns])
                    insert_query = f"""
                        INSERT INTO {self.TABLE_NAME} ({insert_cols})
                        VALUES ({insert_placeholders})
                    """
                    insert_values = [row[col] for col in existing_columns]
                    cursor.execute(insert_query, insert_values)
                    rows_inserted += 1
            
            conn.commit()
        
        total_affected = rows_updated + rows_inserted
        
        # Display success message
        st.success(
            f"Successfully processed {total_affected} rows for session {session}. "
            f"Updated: {rows_updated}, Inserted: {rows_inserted}"
        )
        
        # Verify the data was written
        total_count = self.get_session_count(session)
        st.info(f"Total rows in database for session {session}: {total_count}")
        
        return total_affected

    def upsert_dataframe_with_conflict_resolution(self, df: pd.DataFrame, session: str) -> int:
        """Insert or update DataFrame data using SQLite UPSERT.
        
        Args:
            df: DataFrame containing replicate data.
            session: Session identifier for the data.
            
        Returns:
            Number of rows affected.
        """
        df_prepared = self._prepare_dataframe_for_insert(df)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare UPSERT query
            columns_to_insert = self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS
            insert_cols = ', '.join(columns_to_insert)
            insert_placeholders = ', '.join(['?' for _ in columns_to_insert])
            update_cols = ', '.join([
                f"{col} = excluded.{col}" for col in columns_to_insert if col != 'Timetag'
            ])
            
            upsert_query = f"""
                INSERT INTO {self.TABLE_NAME} ({insert_cols})
                VALUES ({insert_placeholders})
                ON CONFLICT(Timetag) DO UPDATE SET {update_cols}
            """
            
            # Execute UPSERT for each row
            for _, row in df_prepared.iterrows():
                cursor.execute(upsert_query, tuple(row))
            
            conn.commit()
            rows_affected = len(df_prepared)
        
        st.success(f"Successfully inserted/updated {rows_affected} rows for session {session}")
        return rows_affected
    
    def _prepare_dataframe_for_insert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for database insertion.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Prepared DataFrame with correct columns and data types.
        """
        df_copy = df.copy()
        
        # Rename datetime column if present
        if 'datetime' in df_copy.columns:
            df_copy.rename(columns={'datetime': 'Timetag'}, inplace=True)
        
        # Select only required columns
        columns_to_insert = self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS
        df_filtered = df_copy[columns_to_insert].copy()
        
        # Convert Timetag to string
        df_filtered['Timetag'] = df_filtered['Timetag'].astype(str)
        
        return df_filtered

    def get_dataframe(self, session: Optional[str] = None) -> pd.DataFrame:
        """Retrieve data from the database as a DataFrame.
        
        Args:
            session: Optional session filter. If None, returns all data.
            
        Returns:
            DataFrame containing the requested data.
        """
        with self.get_connection() as conn:
            if session:
                query = f"SELECT * FROM {self.TABLE_NAME} WHERE Session = ?"
                df = pd.read_sql(query, conn, params=(session,))
            else:
                df = pd.read_sql_query(f"SELECT * FROM {self.TABLE_NAME}", conn)
        
        # Handle 'Temp' column if present
        if 'Temp' in df.columns:
            df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
        
        return df
    
    def get_sessions(self) -> List[str]:
        """Get list of distinct sessions in the database.
        
        Returns:
            List of session names.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT DISTINCT Session FROM {self.TABLE_NAME}")
            return [row[0] for row in cursor.fetchall()]
    
    def get_session_count(self, session: str) -> int:
        """Get the count of rows for a specific session.
        
        Args:
            session: Session identifier.
            
        Returns:
            Number of rows for the session.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME} WHERE Session = ?", (session,))
            return cursor.fetchone()[0]


# Global database manager instance
_db_manager = DatabaseManager()


# Legacy functions for backward compatibility
def get_connection():
    """Legacy function for backward compatibility."""
    return sqlite3.connect(_db_manager.db_path)


def delete_db():
    """Legacy function for backward compatibility."""
    _db_manager.delete_table()


def init_db():
    """Legacy function for backward compatibility."""
    _db_manager.initialize()


def df_to_db_afterPBLcorrection(df: pd.DataFrame, session: str) -> int:
    """Legacy function for backward compatibility."""
    return _db_manager.upsert_dataframe(df, session)


def df_to_db(df: pd.DataFrame, session: str) -> int:
    """Legacy function for backward compatibility."""
    try:
        return _db_manager.upsert_dataframe_with_conflict_resolution(df, session)
    except DatabaseError as e:
        st.error(str(e))
        return 0
    except Exception as e:
        st.error(f"Error inserting data: {e}")
        return 0


def db_to_df(session: Optional[str] = None) -> pd.DataFrame:
    """Legacy function for backward compatibility."""
    return _db_manager.get_dataframe(session)


def get_sessions() -> List[str]:
    """Legacy function for backward compatibility."""
    return _db_manager.get_sessions()


def is_db_initialized() -> bool:
    """Legacy function for backward compatibility."""
    return _db_manager.is_initialized()


# Initialize database on module import
init_db()