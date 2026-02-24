"""Shared scientific and application constants for D4Xgui.

Values marked as *configurable* are read from the persistent user
settings file (``user_settings.json``) at import time.  Edit them via
the **Settings** page or by calling ``tools.config.set(key, value)``.
"""

from pathlib import Path

from tools import config as _cfg

# ── Package directories (not user-configurable) ──────────────────────
APP_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = APP_DIR / "static"
LOGO_PATH = STATIC_DIR / "D4Xgui_logo_master_red08.png"

# ── Physical constants (not user-configurable) ───────────────────────
KELVIN_OFFSET = 273.15
R18_INITIAL_GUESS = 0.002

# ── Configurable constants ───────────────────────────────────────────
DEFAULT_ACID_TEMPERATURE: int = _cfg.get("acid_temperature")

STANDARD_D47: dict = _cfg.get("standard_d47")
STANDARD_D48: dict = _cfg.get("standard_d48")
STANDARD_D49: dict = _cfg.get("standard_d49")

DEFAULT_WG_RATIOS: dict = _cfg.get("working_gas_ratios")

ISOTOPIC_CONSTANTS: dict = _cfg.get("isotopic_constants")
D18O_VPDB_VSMOW: float = _cfg.get("d18O_VPDB_VSMOW")

def _vpdb_factor(d18o: float) -> float:
    return 1.0 / (1.0 + d18o / 1000.0)

def _vpdb_offset(d18o: float) -> float:
    return d18o / (1.0 + d18o / 1000.0)

VPDB_FACTOR: float = _vpdb_factor(D18O_VPDB_VSMOW)
VPDB_OFFSET: float = _vpdb_offset(D18O_VPDB_VSMOW)

SAMPLE_DB_PATH: Path = STATIC_DIR / _cfg.get("sample_db_filename")
REPLICATES_DB_PATH: Path = APP_DIR / _cfg.get("replicates_db_name")
SESSION_STATES_DB_PATH: Path = APP_DIR / _cfg.get("session_states_db_name")

DEFAULT_REFERENCE_FRAME: str = _cfg.get("default_reference_frame")
DEFAULT_CORRECTION_METHOD: str = _cfg.get("default_correction_method")
DEFAULT_CALIBRATIONS: list = _cfg.get("default_calibrations")
DEFAULT_PROCESS_D47: bool = _cfg.get("default_process_d47")
DEFAULT_PROCESS_D48: bool = _cfg.get("default_process_d48")
DEFAULT_PROCESS_D49: bool = _cfg.get("default_process_d49")
DEFAULT_WG_VIA_STANDARDS: bool = _cfg.get("working_gas_via_standards")
DEFAULT_CO2_STANDARDS: bool = _cfg.get("co2_standards")
BASELINE_CORRECTION_METHOD: str = _cfg.get("baseline_correction_method")
DEFAULT_BASELINE_OVERWRITE_DB: bool = _cfg.get("baseline_overwrite_db")


def reload() -> None:
    """Re-read every configurable constant from the config file.

    Useful after the Settings page writes new values, so that the
    rest of the app picks them up without a full restart.
    """
    global VPDB_FACTOR, VPDB_OFFSET, DEFAULT_ACID_TEMPERATURE
    global STANDARD_D47, STANDARD_D48, STANDARD_D49
    global DEFAULT_WG_RATIOS
    global ISOTOPIC_CONSTANTS, D18O_VPDB_VSMOW
    global SAMPLE_DB_PATH, REPLICATES_DB_PATH, SESSION_STATES_DB_PATH
    global DEFAULT_REFERENCE_FRAME, DEFAULT_CORRECTION_METHOD
    global DEFAULT_CALIBRATIONS
    global DEFAULT_PROCESS_D47, DEFAULT_PROCESS_D48, DEFAULT_PROCESS_D49
    global DEFAULT_WG_VIA_STANDARDS, DEFAULT_CO2_STANDARDS
    global BASELINE_CORRECTION_METHOD, DEFAULT_BASELINE_OVERWRITE_DB

    DEFAULT_ACID_TEMPERATURE = _cfg.get("acid_temperature")
    STANDARD_D47 = _cfg.get("standard_d47")
    STANDARD_D48 = _cfg.get("standard_d48")
    STANDARD_D49 = _cfg.get("standard_d49")
    DEFAULT_WG_RATIOS = _cfg.get("working_gas_ratios")
    ISOTOPIC_CONSTANTS = _cfg.get("isotopic_constants")
    D18O_VPDB_VSMOW = _cfg.get("d18O_VPDB_VSMOW")
    VPDB_FACTOR = _vpdb_factor(D18O_VPDB_VSMOW)
    VPDB_OFFSET = _vpdb_offset(D18O_VPDB_VSMOW)
    SAMPLE_DB_PATH = STATIC_DIR / _cfg.get("sample_db_filename")
    REPLICATES_DB_PATH = APP_DIR / _cfg.get("replicates_db_name")
    SESSION_STATES_DB_PATH = APP_DIR / _cfg.get("session_states_db_name")
    DEFAULT_REFERENCE_FRAME = _cfg.get("default_reference_frame")
    DEFAULT_CORRECTION_METHOD = _cfg.get("default_correction_method")
    DEFAULT_CALIBRATIONS = _cfg.get("default_calibrations")
    DEFAULT_PROCESS_D47 = _cfg.get("default_process_d47")
    DEFAULT_PROCESS_D48 = _cfg.get("default_process_d48")
    DEFAULT_PROCESS_D49 = _cfg.get("default_process_d49")
    DEFAULT_WG_VIA_STANDARDS = _cfg.get("working_gas_via_standards")
    DEFAULT_CO2_STANDARDS = _cfg.get("co2_standards")
    BASELINE_CORRECTION_METHOD = _cfg.get("baseline_correction_method")
    DEFAULT_BASELINE_OVERWRITE_DB = _cfg.get("baseline_overwrite_db")
