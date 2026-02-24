"""Persistent user configuration for D4Xgui.

Settings are stored as a JSON file so they survive app restarts.
Every getter falls back to the hard-coded default when the key is absent,
so the config file only needs to contain the values the user has changed.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "user_settings.json"

# ── Hard-coded defaults ──────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "acid_temperature": 90,
    "working_gas_ratios": {"d18O": 25.260, "d13C": -4.20},

    "sample_db_filename": "SampleDatabase.xlsx",
    "replicates_db_name": "pre_replicates.db",
    "session_states_db_name": "session_states.db",

    "standard_d47": {
        "1000C": 0.0266, "50C": 0.805, "25C": 0.9196,
        "ETH-1": 0.2052, "ETH-2": 0.2085,
    },
    "standard_d48": {
        "1000C": 0.0, "50C": 0.2607, "25C": 0.345,
        "ETH-1": 0.1286, "ETH-2": 0.1286,
    },
    "standard_d49": {
        "1000C": 0.0, "50C": 2.00, "25C": 2.228,
        "ETH-1": 0.562, "ETH-2": 0.707,
    },

    "isotopic_constants": {
        "R13_VPDB": 0.01118,
        "R17_VSMOW": 0.00038475,
        "R18_VSMOW": 0.0020052,
        "lambda_17": 0.528,
        "R18_initial_guess": 0.002,
    },
    "d18O_VPDB_VSMOW": 30.92,

    "standards_bulk_overrides": {},

    "custom_reference_frames": {},

    # Bulk isotope settings
    "working_gas_via_standards": True,
    "co2_standards": False,

    # Processing defaults
    "default_reference_frame": "CDES",
    "default_correction_method": "pooled",
    "default_calibrations": ["Fiebig24 (original)"],
    "default_process_d47": True,
    "default_process_d48": False,
    "default_process_d49": False,

    # Baseline correction defaults
    "baseline_correction_method": "Minimize equilibrated gase slope",
    "baseline_overwrite_db": True,

    # Appearance
    "theme": "Dark",
}

# ── Isotopic constant presets (not persisted) ────────────────────────
# R18_initial_guess is always 0.002 and not user-configurable.
ISOTOPIC_PRESETS = {
    "Brand": {
        "R13_VPDB": 0.01118,
        "R17_VSMOW": 0.00038475,
        "R18_VSMOW": 0.0020052,
        "lambda_17": 0.528,
    },
    "Gonfiantini": {
        "R13_VPDB": 0.0112372,
        "R17_VSMOW": 0.0003799,
        "R18_VSMOW": 0.0020052,
        "lambda_17": 0.5164,
    },
}


def _load_raw() -> Dict[str, Any]:
    """Read the JSON file, returning {} on any error."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_raw(data: Dict[str, Any]) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── Public API ───────────────────────────────────────────────────────

def get(key: str, default: Any = None) -> Any:
    """Return the user-configured value, falling back to built-in default."""
    data = _load_raw()
    if key in data:
        return data[key]
    if key in _DEFAULTS:
        return _DEFAULTS[key]
    return default


def get_all() -> Dict[str, Any]:
    """Return the full merged config (defaults + user overrides)."""
    merged = dict(_DEFAULTS)
    merged.update(_load_raw())
    return merged


def set(key: str, value: Any) -> None:
    """Persist a single key."""
    data = _load_raw()
    data[key] = value
    _save_raw(data)


def set_many(updates: Dict[str, Any]) -> None:
    """Persist several keys at once."""
    data = _load_raw()
    data.update(updates)
    _save_raw(data)


def reset(key: Optional[str] = None) -> None:
    """Reset one key (or all keys) back to defaults."""
    if key is None:
        if _CONFIG_PATH.exists():
            _CONFIG_PATH.unlink()
        return
    data = _load_raw()
    data.pop(key, None)
    _save_raw(data)


def defaults() -> Dict[str, Any]:
    """Return a copy of the built-in defaults."""
    return dict(_DEFAULTS)


def config_path() -> Path:
    return _CONFIG_PATH


# ── Theme management ─────────────────────────────────────────────────

_STREAMLIT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
)

THEMES: Dict[str, Dict[str, str]] = {
    "Dark": {
        "base": "dark",
        "primaryColor": "#aa80f7",
        "backgroundColor": "#0E1117",
        "secondaryBackgroundColor": "#262730",
        "textColor": "#FAFAFA",
        "font": "sans serif",
    },
    "Light": {
        "base": "light",
        "primaryColor": "#7B68AE",
        "backgroundColor": "#FAFAFE",
        "secondaryBackgroundColor": "#EDEBF4",
        "textColor": "#2D2A3E",
        "font": "sans serif",
    },
}

THEME_CHOICES = list(THEMES.keys())


def apply_theme(theme_name: str) -> None:
    """Write the selected theme to .streamlit/config.toml."""
    colors = THEMES.get(theme_name, THEMES["Dark"])

    config_text = (
        "[theme]\n"
        f'base="{colors["base"]}"\n'
        f'primaryColor="{colors["primaryColor"]}"\n'
        f'backgroundColor="{colors["backgroundColor"]}"\n'
        f'secondaryBackgroundColor="{colors["secondaryBackgroundColor"]}"\n'
        f'textColor="{colors["textColor"]}"\n'
        f'font="{colors["font"]}"\n'
        "\n\n"
        "[browser]\n"
        "gatherUsageStats = false\n"
        "\n"
        "[server]\n"
        "showEmailPrompt = false\n"
        "port = 1337\n"
    )

    _STREAMLIT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STREAMLIT_CONFIG_PATH.write_text(config_text, encoding="utf-8")


def ensure_theme() -> None:
    """Apply the persisted theme on startup (idempotent)."""
    theme_name = get("theme", "Dark")
    apply_theme(theme_name)
