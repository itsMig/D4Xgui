#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal datetime normalizer for Timetag columns.

Handles:
- pandas datetime64 (pass-through)
- Excel serial numbers (float, origin 1899-12-30)
- String timestamps in various locales (US, China, ISO, etc.)
"""

import re
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser

STANDARD_DATETIME_FORMATS: List[str] = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
]

EXCEL_EPOCH = pd.Timestamp("1899-12-30")
_EXCEL_SERIAL_MIN = 1.0
_EXCEL_SERIAL_MAX = 2_958_466.0  # ~9999-12-31


def _try_standard_formats(timestamp_str: str) -> Optional[datetime]:
    """Try parsing with a short list of common datetime formats."""
    for fmt in STANDARD_DATETIME_FORMATS:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    return None


def _clean_timestamp(timestamp_str: str) -> str:
    """Remove timezone abbreviations and non-datetime characters."""
    cleaned = re.sub(r"\s+[A-Z]{2,4}$", "", timestamp_str)
    cleaned = "".join(
        ch for ch in cleaned
        if ch.isdigit() or ch in (" ", ":", "-", ".", "T", "+", "Z")
    )
    return cleaned.rstrip()


def _parse_single_timestamp(timestamp_str) -> Optional[datetime]:
    """
    Parse a single timestamp value with multiple fallback strategies.

    Order: standard formats -> ISO -> dateutil -> clean-and-retry.
    """
    if pd.isna(timestamp_str) or str(timestamp_str).strip() == "":
        return None

    s = str(timestamp_str).strip()

    result = _try_standard_formats(s)
    if result is not None:
        return result

    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass

    try:
        return dateutil_parser.parse(s, fuzzy=False)
    except (ValueError, TypeError):
        pass

    try:
        cleaned = _clean_timestamp(s)
        result = _try_standard_formats(cleaned)
        if result is not None:
            return result
        return dateutil_parser.parse(cleaned, fuzzy=False)
    except (ValueError, TypeError):
        return None


def _is_excel_serial(series: pd.Series) -> bool:
    """Heuristic: all non-null values are floats in the Excel serial range."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    if not np.issubdtype(non_null.dtype, np.number):
        try:
            non_null = pd.to_numeric(non_null, errors="coerce").dropna()
        except Exception:
            return False
        if non_null.empty:
            return False
    return bool(
        (non_null >= _EXCEL_SERIAL_MIN).all()
        and (non_null <= _EXCEL_SERIAL_MAX).all()
    )


def normalize_datetime_series(series: pd.Series) -> pd.Series:
    """
    Convert *series* to ``datetime64[ns]`` using every reasonable strategy.

    Returns a new Series of the same length.  Unparseable values become NaT.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.tz_localize(None) if hasattr(series.dt, "tz") and series.dt.tz else series

    if _is_excel_serial(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return EXCEL_EPOCH + pd.to_timedelta(numeric, unit="D")

    coerced = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if coerced.notna().all():
        return coerced

    # Row-by-row fallback for remaining NaTs
    result = coerced.copy()
    nat_mask = result.isna()
    for idx in result.index[nat_mask]:
        parsed = _parse_single_timestamp(series.loc[idx])
        if parsed is not None:
            result.at[idx] = pd.Timestamp(parsed)

    return result
