"""Unified sample-name filtering for D4Xgui pages."""

from typing import List

import pandas as pd
import streamlit as st


def parse_filter_keywords(raw: str) -> List[str]:
    """Split a semicolon-separated string into trimmed, non-empty keywords."""
    if not raw or not isinstance(raw, str):
        return []
    return [kw.strip() for kw in raw.split(";") if kw.strip()]


def filter_dataframe(
    df: pd.DataFrame,
    include_str: str = "",
    exclude_str: str = "",
    column: str = "Sample",
) -> pd.DataFrame:
    """Filter df by sample-name keywords (include any / exclude any)."""
    result = df.copy()
    include_kws = parse_filter_keywords(include_str)
    if include_kws:
        result = result[result[column].apply(
            lambda x: any(kw in str(x) for kw in include_kws)
        )]
    exclude_kws = parse_filter_keywords(exclude_str)
    if exclude_kws:
        result = result[~result[column].apply(
            lambda x: any(kw in str(x) for kw in exclude_kws)
        )]
    return result


def render_sample_filter_sidebar(page_prefix: str, use_columns: bool = False) -> None:
    """Render the standard KEEP/DROP text inputs in the sidebar.

    Keys stored: "{page_prefix}_sample_contains", "{page_prefix}_sample_not_contains"
    """
    help_text = "Set multiple keywords by separating them through semicolons ;"
    if use_columns:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.sidebar.text_input(
                "Sample name contains (KEEP):", help=help_text,
                key=f"{page_prefix}_sample_contains", value="",
            )
        with col2:
            st.sidebar.text_input(
                "Sample name contains (DROP):", help=help_text,
                key=f"{page_prefix}_sample_not_contains", value="",
            )
    else:
        st.sidebar.text_input(
            "Sample name contains (KEEP):", help=help_text,
            key=f"{page_prefix}_sample_contains", value="",
        )
        st.sidebar.text_input(
            "Sample name contains (DROP):", help=help_text,
            key=f"{page_prefix}_sample_not_contains", value="",
        )


def apply_session_filters(
    df: pd.DataFrame, page_prefix: str, column: str = "Sample",
) -> pd.DataFrame:
    """Read filter strings from session state and apply."""
    include = st.session_state.get(f"{page_prefix}_sample_contains", "")
    exclude = st.session_state.get(f"{page_prefix}_sample_not_contains", "")
    return filter_dataframe(df, include, exclude, column)
