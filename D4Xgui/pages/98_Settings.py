#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent application settings page.

All changes are written to ``user_settings.json`` and survive app restarts.
"""

import json
import re
from typing import Any, Dict

import pandas as pd
import streamlit as st

from tools.base_page import BasePage
from tools import config as cfg
from tools.init_params import IsotopeStandards


# ── Helpers for grouping standard_d4X dicts in the JSON display ──────
#
# Internal storage uses three flat keys:
#   standard_d47 = {"1000C": …, "ETH-1": …, "test1": …}
#   standard_d48 = {…}
#   standard_d49 = {…}
#
# For the raw-JSON tab we merge them into group-level keys:
#   baseline_equilibrated_gases = {"D47": {"1000C": …}, "D48": {…}, …}
#   baseline_ETH1-2             = {"D47": {"ETH-1": …}, …}
#   baseline_custom_stds        = {"D47": {"test1": …}, …}

_STD_CONFIG_KEYS = frozenset({"standard_d47", "standard_d48", "standard_d49"})
_GROUP_NAMES = (
    "baseline_equilibrated_gases",
    "baseline_ETH1-2",
    "baseline_custom_stds",
)
_GROUP_NAMES_SET = frozenset(_GROUP_NAMES)

_STD_KEY_TO_ISOTOPE = {
    "standard_d47": "D47",
    "standard_d48": "D48",
    "standard_d49": "D49",
}
_ISOTOPE_TO_STD_KEY = {v: k for k, v in _STD_KEY_TO_ISOTOPE.items()}

_STD_FRAME_PREFIX = "standardization_"
_MASS_TO_DISPLAY = {"47": "D47", "48": "D48", "49": "D49"}
_DISPLAY_TO_MASS = {v: k for k, v in _MASS_TO_DISPLAY.items()}


def _classify_standard(name: str) -> str:
    """Return the group key a standard name belongs to."""
    if re.match(r'^\d+C$', name):
        return "baseline_equilibrated_gases"
    if name.startswith("ETH-"):
        return "baseline_ETH1-2"
    return "baseline_custom_stds"


def _nest_config_for_display(config: dict) -> dict:
    """Replace standard_d4X keys with group-level keys for readable display.

    Also expands standardization reference frames (CDES, ICDES, …) into
    ``standardization_<name>`` keys and hides the raw
    ``custom_reference_frames`` dict.
    """
    groups: dict = {}
    for cfg_key in ("standard_d47", "standard_d48", "standard_d49"):
        val = config.get(cfg_key)
        if not isinstance(val, dict):
            continue
        isotope = _STD_KEY_TO_ISOTOPE[cfg_key]
        for std_name, std_val in val.items():
            grp = _classify_standard(std_name)
            groups.setdefault(grp, {}).setdefault(isotope, {})[std_name] = std_val

    out: dict = {}
    inserted = False
    for key in config:
        if key in _STD_CONFIG_KEYS:
            if not inserted:
                for gname in _GROUP_NAMES:
                    if gname in groups:
                        out[gname] = groups[gname]
                inserted = True
        elif key == "custom_reference_frames":
            continue
        else:
            out[key] = config[key]

    if not inserted and groups:
        for gname in _GROUP_NAMES:
            if gname in groups:
                out[gname] = groups[gname]

    # Expand standardization reference frames into readable keys
    all_frames = IsotopeStandards.get_standards()
    for frame_name, frame_data in all_frames.items():
        display_frame: dict = {}
        for k, v in frame_data.items():
            display_key = _MASS_TO_DISPLAY.get(k, k)
            display_frame[display_key] = v
        out[f"{_STD_FRAME_PREFIX}{frame_name}"] = display_frame

    return out


def _flatten_config_from_display(config: dict) -> dict:
    """Convert group-level keys back to flat standard_d4X keys.

    Also converts ``standardization_<name>`` keys back into
    ``custom_reference_frames`` overrides.
    """
    has_groups = any(k in config for k in _GROUP_NAMES_SET)
    std_frame_keys = [k for k in config if k.startswith(_STD_FRAME_PREFIX)]

    if not has_groups and not std_frame_keys:
        return dict(config)

    flat_stds: dict = {
        "standard_d47": {}, "standard_d48": {}, "standard_d49": {},
    }
    out: dict = {}
    inserted = False
    for key, value in config.items():
        if key in _GROUP_NAMES_SET:
            if isinstance(value, dict):
                for isotope, standards in value.items():
                    cfg_key = _ISOTOPE_TO_STD_KEY.get(isotope)
                    if cfg_key and isinstance(standards, dict):
                        flat_stds[cfg_key].update(standards)
            if not inserted:
                for cfg_key in ("standard_d47", "standard_d48", "standard_d49"):
                    out[cfg_key] = flat_stds[cfg_key]
                inserted = True
        elif key.startswith(_STD_FRAME_PREFIX):
            continue
        else:
            out[key] = value

    if has_groups and not inserted:
        for cfg_key in ("standard_d47", "standard_d48", "standard_d49"):
            out[cfg_key] = flat_stds[cfg_key]

    # Rebuild custom_reference_frames from standardization_* keys
    if std_frame_keys:
        builtin_names = set(IsotopeStandards.STANDARDS_NOMINAL.keys())
        custom_frames: dict = {}
        for key in std_frame_keys:
            frame_name = key[len(_STD_FRAME_PREFIX):]
            raw = config[key]
            if not isinstance(raw, dict):
                continue
            converted: dict = {}
            for k, v in raw.items():
                converted[_DISPLAY_TO_MASS.get(k, k)] = v
            if frame_name in builtin_names:
                if converted != IsotopeStandards.STANDARDS_NOMINAL[frame_name]:
                    custom_frames[frame_name] = converted
            else:
                custom_frames[frame_name] = converted
        out["custom_reference_frames"] = custom_frames

    return out


class SettingsPage(BasePage):
    """User-facing page for editing persistent application settings."""

    PAGE_NUMBER = 98
    PAGE_TITLE = "Settings"
    SHOW_LOGO = False

    _STD_KEYS = ("standard_d47", "standard_d48", "standard_d49")

    def run(self) -> None:
        st.markdown(
            "Changes made here are saved to `user_settings.json` and "
            "persist across restarts."
        )
        saved = False

        tab_iso, tab_bulk, tab_clumped, tab_sw, tab_json = st.tabs([
            "Isotopic constants",
            "Bulk isotopes",
            "Clumped isotopes",
            "Software configuration",
            "Global parameter JSON",
        ])

        with tab_iso:
            saved |= self._render_isotopic_constants()
        with tab_bulk:
            saved |= self._render_bulk_isotopes()
        with tab_clumped:
            saved |= self._render_clumped_isotopes()
        with tab_sw:
            saved |= self._render_software_config()
        with tab_json:
            saved |= self._render_raw_json()

        if saved:
            from tools.constants import reload
            reload()

    # ── Tab 1: Isotopic constants ─────────────────────────────────────

    def _render_isotopic_constants(self) -> bool:
        st.subheader("Isotopic constants (¹⁷O correction)")
        st.info(
            "Brand parameters are the current IUPAC community standard. "
            "These will be restored when resetting to default values."
        )

        current = cfg.get_all()
        defs = cfg.defaults()
        ic = current.get("isotopic_constants", defs["isotopic_constants"])

        with st.form("isotopic_constants_form"):
            col1, col2 = st.columns(2)
            with col1:
                r13 = st.number_input(
                    "R¹³_VPDB", value=float(ic["R13_VPDB"]),
                    format="%.7f", step=1e-7,
                )
                r17 = st.number_input(
                    "R¹⁷_VSMOW", value=float(ic["R17_VSMOW"]),
                    format="%.8f", step=1e-8,
                )
                r18 = st.number_input(
                    "R¹⁸_VSMOW", value=float(ic["R18_VSMOW"]),
                    format="%.7f", step=1e-7,
                )
            with col2:
                lam = st.number_input(
                    "λ₁₇", value=float(ic["lambda_17"]),
                    format="%.4f", step=1e-4,
                )
                d18_vpdb_vsmow = st.number_input(
                    "δ¹⁸O VPDB-VSMOW (‰)",
                    value=float(current.get(
                        "d18O_VPDB_VSMOW", defs["d18O_VPDB_VSMOW"]
                    )),
                    format="%.2f", step=0.01,
                )

            submitted = st.form_submit_button("Save isotopic constants")
            if submitted:
                new_ic = {
                    "R13_VPDB": r13,
                    "R17_VSMOW": r17,
                    "R18_VSMOW": r18,
                    "lambda_17": lam,
                    "R18_initial_guess": 0.002,
                }
                cfg.set_many({
                    "isotopic_constants": new_ic,
                    "d18O_VPDB_VSMOW": d18_vpdb_vsmow,
                })
                st.success("Isotopic constants saved.")
                return True
        return False

    # ── Tab 2: Bulk isotopes ──────────────────────────────────────────

    _WG_MODE_VIA_STDS = "Via standards"
    _WG_MODE_VIA_COMPOSITION = "Via known WG composition"
    _WG_MODES = [_WG_MODE_VIA_STDS, _WG_MODE_VIA_COMPOSITION]

    def _render_bulk_isotopes(self) -> bool:
        st.subheader("Bulk isotope settings")
        saved = False
        current = cfg.get_all()
        defs = cfg.defaults()

        with st.form("bulk_isotopes_form"):
            acid_temp = st.number_input(
                "Default acid temperature (°C)",
                value=current.get("acid_temperature", defs["acid_temperature"]),
                min_value=0, max_value=200, step=1,
            )
            submitted = st.form_submit_button("Save reaction temperature")

        if submitted:
            cfg.set("acid_temperature", int(acid_temp))
            st.success("Acid temperature saved.")
            saved = True

        st.markdown("---")
        st.markdown("**Working gas settings**")

        wg_via_stds = current.get(
            "working_gas_via_standards",
            defs["working_gas_via_standards"],
        )
        idx = 0 if wg_via_stds else 1

        choice = st.radio(
            "Working gas mode",
            options=self._WG_MODES,
            index=idx,
            horizontal=True,
            key="settings_wg_mode",
            help=(
                "**Via standards** — bulk isotopic standardization is "
                "calculated via 1-pt or multi-point ETF using anchor "
                "standards.  \n"
                "**Via known WG composition** — the known working gas "
                "composition is used directly."
            ),
        )

        new_wg_via_stds = choice == self._WG_MODE_VIA_STDS
        if new_wg_via_stds != wg_via_stds:
            cfg.set("working_gas_via_standards", new_wg_via_stds)
            saved = True

        if new_wg_via_stds:
            saved |= self._render_wg_via_standards_on(current, defs)
        else:
            saved |= self._render_wg_via_standards_off(current, defs)

        return saved

    def _render_wg_via_standards_on(
        self, current: dict, defs: dict,
    ) -> bool:
        """Display CO2 standards toggle and bulk anchor tables."""
        saved = False

        co2_stds = st.checkbox(
            "CO₂ standards (sets acid fractionation factor to 1)",
            value=current.get("co2_standards", defs["co2_standards"]),
            key="settings_co2_stds",
        )
        if st.button("Save CO₂ standards setting", key="btn_save_co2"):
            cfg.set("co2_standards", co2_stds)
            st.success("CO₂ standards setting saved.")
            saved = True

        st.markdown("**Bulk isotopic standards**")
        st.caption(
            "These δ¹³C and δ¹⁸O anchor values are used for bulk "
            "isotopic standardization."
        )
        current_bulk = IsotopeStandards.get_bulk()

        raw_18 = current_bulk.get(18, {})
        raw_13 = current_bulk.get(13, {})
        all_bulk_names = list(dict.fromkeys(
            list(raw_18.keys()) + list(raw_13.keys())
        ))
        bulk_rows = [
            {
                "Standard": name,
                "δ¹⁸O": raw_18.get(name),
                "δ¹³C": raw_13.get(name),
            }
            for name in all_bulk_names
        ]
        if bulk_rows:
            bulk_df = pd.DataFrame(bulk_rows)
        else:
            bulk_df = pd.DataFrame(
                columns=["Standard", "δ¹⁸O", "δ¹³C"],
            ).astype({"Standard": str, "δ¹⁸O": float, "δ¹³C": float})

        edited_bulk = st.data_editor(
            bulk_df, num_rows="dynamic", width="stretch",
            key="de_bulk_unified",
            column_config={
                "Standard": st.column_config.TextColumn("Standard"),
                "δ¹⁸O": st.column_config.NumberColumn("δ¹⁸O", format="%.3f"),
                "δ¹³C": st.column_config.NumberColumn("δ¹³C", format="%.3f"),
            },
        )
        col_save, col_reset = st.columns([1, 3])
        with col_save:
            if st.button("Save bulk anchors", key="btn_save_bulk"):
                overrides = cfg.get("standards_bulk_overrides") or {}
                for iso_key, col_name in [(18, "δ¹⁸O"), (13, "δ¹³C")]:
                    new_dict = {}
                    for _, row in edited_bulk.iterrows():
                        name = row["Standard"]
                        if isinstance(name, (list, tuple)):
                            name = name[0] if name else ""
                        name = str(name).strip()
                        if name and pd.notna(row[col_name]):
                            try:
                                new_dict[name] = float(row[col_name])
                            except (ValueError, TypeError):
                                pass
                    overrides[str(iso_key)] = new_dict
                cfg.set("standards_bulk_overrides", overrides)
                st.success("Bulk isotopic standards saved.")
                saved = True
        with col_reset:
            if st.button(
                "Reset bulk anchors to defaults",
                key="btn_reset_bulk",
            ):
                cfg.reset("standards_bulk_overrides")
                st.success("Bulk isotopic standards reset to defaults.")
                saved = True
        return saved

    def _render_wg_via_standards_off(
        self, current: dict, defs: dict,
    ) -> bool:
        """Display WG composition inputs."""
        st.markdown("**Working gas composition**")
        wg = current.get("working_gas_ratios", defs["working_gas_ratios"])

        with st.form("wg_composition_form"):
            col1, col2 = st.columns(2)
            with col1:
                wg_d18o = st.number_input(
                    "δ¹⁸O (‰ VSMOW)",
                    value=float(wg.get("d18O", 25.260)),
                    format="%.3f",
                )
            with col2:
                wg_d13c = st.number_input(
                    "δ¹³C (‰ VPDB)",
                    value=float(wg.get("d13C", -4.20)),
                    format="%.3f",
                )
            submitted = st.form_submit_button("Save working gas composition")
            if submitted:
                cfg.set(
                    "working_gas_ratios",
                    {"d18O": wg_d18o, "d13C": wg_d13c},
                )
                st.success("Working gas composition saved.")
                return True
        return False

    # ── Tab 3: Clumped isotopes ───────────────────────────────────────

    def _render_clumped_isotopes(self) -> bool:
        saved = False
        with st.expander("Baseline correction", expanded=False):
            saved |= self._render_baseline_correction_section()
        with st.expander("Standardization reference frames", expanded=False):
            saved |= self._render_reference_frames_section()
        return saved

    # ── Section 1: Baseline correction ────────────────────────────────

    # Default standard values per method (used for pre-filling tables)
    _GAS_DEFAULTS = {
        "standard_d47": {"1000C": 0.0266, "50C": 0.805, "25C": 0.9196},
        "standard_d48": {"1000C": 0.0, "50C": 0.2607, "25C": 0.345},
        "standard_d49": {"1000C": 0.0, "50C": 2.00, "25C": 2.228},
    }
    _ETH_DEFAULTS = {
        "standard_d47": {"ETH-1": 0.2052, "ETH-2": 0.2085},
        "standard_d48": {"ETH-1": 0.1286, "ETH-2": 0.1286},
        "standard_d49": {"ETH-1": 0.562, "ETH-2": 0.707},
    }

    _METHOD_CHOICES = [
        "Minimize equilibrated gase slope",
        "Correct baseline using ETH-1 & ETH-2",
        "Use custom standards...",
    ]

    def _render_baseline_correction_section(self) -> bool:
        saved = False
        current = cfg.get_all()
        defs = cfg.defaults()

        active = current.get(
            "baseline_correction_method",
            defs.get("baseline_correction_method", self._METHOD_CHOICES[0]),
        )
        idx = (
            self._METHOD_CHOICES.index(active)
            if active in self._METHOD_CHOICES
            else 0
        )
        method = st.selectbox(
            "Default baseline correction method",
            options=self._METHOD_CHOICES,
            index=idx,
            key="settings_bl_method",
        )

        if method != active:
            cfg.set("baseline_correction_method", method)
            saved = True

        st.markdown("**Baseline-correction standard values**")
        st.caption(
            "These values are used as target Δ values during baseline "
            "correction on the Baseline Correction page."
        )

        if method == self._METHOD_CHOICES[0]:
            saved |= self._render_bl_tables(
                current, defs, self._GAS_DEFAULTS,
            )
        elif method == self._METHOD_CHOICES[1]:
            saved |= self._render_bl_tables(
                current, defs, self._ETH_DEFAULTS,
            )
        else:
            saved |= self._render_bl_tables(
                current, defs, None,
            )

        if st.button(
            "Reset baseline correction parameters",
            key="btn_reset_all_bl",
        ):
            for k in ("standard_d47", "standard_d48", "standard_d49",
                       "baseline_correction_method"):
                cfg.reset(k)
            st.success(
                "All baseline correction parameters reset to defaults."
            )
            saved = True

        return saved

    def _render_bl_tables(
        self,
        current: dict,
        defs: dict,
        method_defaults: dict | None,
    ) -> bool:
        """Render a single unified table with Δ₄₇/Δ₄₈/Δ₄₉ columns."""
        saved = False

        displays: dict[str, dict] = {}
        for cfg_key in self._STD_KEYS:
            isotope = _STD_KEY_TO_ISOTOPE[cfg_key]
            if method_defaults is not None:
                source = method_defaults.get(cfg_key, {})
                stored = current.get(cfg_key, defs[cfg_key])
                displays[isotope] = {
                    k: stored.get(k, v) for k, v in source.items()
                }
            else:
                vals = current.get(cfg_key, defs.get(cfg_key, {}))
                known_keys = set(self._GAS_DEFAULTS.get(cfg_key, {})) | set(
                    self._ETH_DEFAULTS.get(cfg_key, {})
                )
                displays[isotope] = {
                    k: v for k, v in vals.items() if k not in known_keys
                }

        all_names = list(dict.fromkeys(
            name for d in displays.values() for name in d
        ))

        rows = [
            {
                "Standard": name,
                "D47": displays["D47"].get(name),
                "D48": displays["D48"].get(name),
                "D49": displays["D49"].get(name),
            }
            for name in all_names
        ]

        if rows:
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(
                columns=["Standard", "D47", "D48", "D49"],
            ).astype({"Standard": str, "D47": float,
                       "D48": float, "D49": float})

        edited = st.data_editor(
            df,
            num_rows="dynamic",
            width="stretch",
            key="de_bl_standards",
            column_config={
                "Standard": st.column_config.TextColumn("Standard"),
                "D47": st.column_config.NumberColumn("Δ₄₇", format="%.4f"),
                "D48": st.column_config.NumberColumn("Δ₄₈", format="%.4f"),
                "D49": st.column_config.NumberColumn("Δ₄₉", format="%.4f"),
            },
        )

        col_save, col_reset = st.columns([1, 3])
        with col_save:
            if st.button("Save standards", key="btn_save_bl_stds"):
                for cfg_key in self._STD_KEYS:
                    isotope = _STD_KEY_TO_ISOTOPE[cfg_key]
                    new_dict = {}
                    for _, row in edited.iterrows():
                        name = row["Standard"]
                        if isinstance(name, (list, tuple)):
                            name = name[0] if name else ""
                        name = str(name).strip()
                        if name and pd.notna(row[isotope]):
                            try:
                                new_dict[name] = float(row[isotope])
                            except (ValueError, TypeError):
                                pass
                    full = dict(current.get(cfg_key, defs[cfg_key]))
                    full.update(new_dict)
                    cfg.set(cfg_key, full)
                st.success("Baseline correction standards saved.")
                saved = True
        with col_reset:
            if st.button(
                "Reset standards to defaults",
                key="btn_reset_bl_stds",
            ):
                for cfg_key in self._STD_KEYS:
                    cfg.reset(cfg_key)
                st.success("Standards reset to defaults.")
                saved = True

        return saved

    # ── Section 2: Reference frames ───────────────────────────────────

    def _render_reference_frames_section(self) -> bool:
        st.caption(
            "View and edit all reference frames (built-in and custom). "
            "Changes to built-in frames are saved as overrides; use "
            "**Reset** to restore the original values."
        )
        saved = False
        builtin_names = set(IsotopeStandards.STANDARDS_NOMINAL.keys())
        custom_frames: Dict[str, Any] = cfg.get("custom_reference_frames") or {}
        all_frames = IsotopeStandards.get_standards()

        for frame_name in list(all_frames.keys()):
            is_builtin = frame_name in builtin_names
            is_modified = is_builtin and frame_name in custom_frames

            if is_builtin and not is_modified:
                badge = " *(built-in)*"
            elif is_modified:
                badge = " *(built-in, modified)*"
            else:
                badge = " *(custom)*"

            with st.expander(f"**{frame_name}**{badge}", expanded=False):
                frame = all_frames[frame_name]
                info_val = st.text_input(
                    "Description",
                    value=frame.get("#info", ""),
                    key=f"ref_info_{frame_name}",
                )

                all_std_names = list(dict.fromkeys(
                    name for mz_key in ("47", "48", "49")
                    if mz_key in frame
                    for name in frame[mz_key]
                ))
                ref_rows = [
                    {
                        "Standard": name,
                        "D47": frame.get("47", {}).get(name),
                        "D48": frame.get("48", {}).get(name),
                        "D49": frame.get("49", {}).get(name),
                    }
                    for name in all_std_names
                ]
                if ref_rows:
                    ref_df = pd.DataFrame(ref_rows)
                else:
                    ref_df = pd.DataFrame(
                        columns=["Standard", "D47", "D48", "D49"],
                    ).astype({"Standard": str, "D47": float,
                              "D48": float, "D49": float})

                edited_ref = st.data_editor(
                    ref_df, num_rows="dynamic", width="stretch",
                    key=f"ref_{frame_name}_unified",
                    column_config={
                        "Standard": st.column_config.TextColumn("Standard"),
                        "D47": st.column_config.NumberColumn("Δ₄₇", format="%.4f"),
                        "D48": st.column_config.NumberColumn("Δ₄₈", format="%.4f"),
                        "D49": st.column_config.NumberColumn("Δ₄₉", format="%.4f"),
                    },
                )

                col_save, col_action, _ = st.columns([1, 1, 2])
                with col_save:
                    if st.button("Save", key=f"save_ref_{frame_name}"):
                        new_frame: Dict[str, Any] = {}
                        if info_val.strip():
                            new_frame["#info"] = info_val.strip()
                        for mz_key, col_name in [("47", "D47"), ("48", "D48"), ("49", "D49")]:
                            new_dict = {}
                            for _, row in edited_ref.iterrows():
                                name = row["Standard"]
                                if isinstance(name, (list, tuple)):
                                    name = name[0] if name else ""
                                name = str(name).strip()
                                if name and pd.notna(row[col_name]):
                                    try:
                                        new_dict[name] = float(row[col_name])
                                    except (ValueError, TypeError):
                                        pass
                            if new_dict:
                                new_frame[mz_key] = new_dict
                        custom_frames[frame_name] = new_frame
                        cfg.set("custom_reference_frames", custom_frames)
                        st.success(f"Saved «{frame_name}».")
                        saved = True
                        st.rerun()
                with col_action:
                    if is_builtin:
                        if is_modified and st.button(
                            "Reset to default",
                            key=f"reset_ref_{frame_name}",
                        ):
                            del custom_frames[frame_name]
                            cfg.set(
                                "custom_reference_frames", custom_frames,
                            )
                            st.success(
                                f"Reset «{frame_name}» to defaults."
                            )
                            saved = True
                            st.rerun()
                    else:
                        if st.button(
                            "Delete", key=f"del_ref_{frame_name}"
                        ):
                            del custom_frames[frame_name]
                            cfg.set(
                                "custom_reference_frames", custom_frames,
                            )
                            st.success(f"Deleted «{frame_name}».")
                            saved = True
                            st.rerun()

        # --- add new frame (table-based) ---
        st.markdown("---")
        st.markdown("#### Add a new reference frame")

        with st.form("add_ref_frame", clear_on_submit=True):
            new_name = st.text_input("Frame name")
            new_info = st.text_input(
                "Description", placeholder="e.g. My lab, 2025"
            )

            st.markdown(
                "Enter anchor values in the table below. Add rows as "
                "needed."
            )
            new_ref_df = pd.DataFrame(
                columns=["Standard", "D47", "D48", "D49"],
            ).astype({"Standard": str, "D47": float,
                       "D48": float, "D49": float})
            edited_new_ref = st.data_editor(
                new_ref_df,
                num_rows="dynamic",
                width="stretch",
                key="new_ref_unified",
                column_config={
                    "Standard": st.column_config.TextColumn("Standard"),
                    "D47": st.column_config.NumberColumn("Δ₄₇", format="%.4f"),
                    "D48": st.column_config.NumberColumn("Δ₄₈", format="%.4f"),
                    "D49": st.column_config.NumberColumn("Δ₄₉", format="%.4f"),
                },
            )

            submitted = st.form_submit_button("Add reference frame")
            if submitted:
                if not new_name.strip():
                    st.error("Please provide a frame name.")
                else:
                    frame_data: Dict[str, Any] = {}
                    if new_info.strip():
                        frame_data["#info"] = new_info.strip()
                    for mz_key, col_name in [("47", "D47"), ("48", "D48"), ("49", "D49")]:
                        anchors = {}
                        for _, row in edited_new_ref.iterrows():
                            name = row["Standard"]
                            if isinstance(name, (list, tuple)):
                                name = name[0] if name else ""
                            name = str(name).strip()
                            if name and pd.notna(row[col_name]):
                                try:
                                    anchors[name] = float(row[col_name])
                                except (ValueError, TypeError):
                                    pass
                        if anchors:
                            frame_data[mz_key] = anchors
                    custom_frames[new_name.strip()] = frame_data
                    cfg.set("custom_reference_frames", custom_frames)
                    st.success(
                        f"Added reference frame «{new_name.strip()}»."
                    )
                    saved = True

        if st.button(
            "Reset all reference frames", key="btn_reset_all_ref"
        ):
            cfg.reset("custom_reference_frames")
            st.success("All custom reference frames removed.")
            saved = True

        return saved

    # ── Tab 4: Software configuration ─────────────────────────────────

    def _render_software_config(self) -> bool:
        saved = False

        # ── Appearance ────────────────────────────────────────────
        st.subheader("Appearance")
        current = cfg.get_all()
        defs = cfg.defaults()

        active_theme = current.get("theme", defs.get("theme", "Dark"))
        idx = (
            cfg.THEME_CHOICES.index(active_theme)
            if active_theme in cfg.THEME_CHOICES
            else 0
        )
        selected_theme = st.selectbox(
            "Color theme",
            options=cfg.THEME_CHOICES,
            index=idx,
            key="settings_theme",
            help="Switch between Dark and Light mode. "
                 "The page will reload to apply the new theme.",
        )

        if selected_theme != active_theme:
            cfg.set("theme", selected_theme)
            cfg.apply_theme(selected_theme)
            st.toast(
                f"Theme changed to **{selected_theme}**. Reloading…",
                icon="🎨",
            )
            import time
            time.sleep(0.6)
            st.rerun()

        # ── Database paths ────────────────────────────────────────
        st.markdown("---")
        st.subheader("Databases")

        with st.form("database_paths"):
            sample_db = st.text_input(
                "Sample Metadata filename (Excel, inside static/)",
                value=current.get(
                    "sample_db_filename", defs["sample_db_filename"]
                ),
            )
            rep_db = st.text_input(
                "Replicates database filename",
                value=current.get(
                    "replicates_db_name", defs["replicates_db_name"]
                ),
            )
            sess_db = st.text_input(
                "Session-states database filename",
                value=current.get(
                    "session_states_db_name", defs["session_states_db_name"]
                ),
            )
            submitted = st.form_submit_button("Save database paths")
            if submitted:
                cfg.set_many({
                    "sample_db_filename": sample_db,
                    "replicates_db_name": rep_db,
                    "session_states_db_name": sess_db,
                })
                st.success("Database paths saved.")
                saved = True

        return saved

    # ── Tab 5: Raw JSON view / editor ─────────────────────────────────

    def _render_raw_json(self) -> bool:
        st.subheader("Global parameter JSON (all parameters)")
        st.caption(
            "View and edit **every** configurable parameter in one place. "
            "Edit the JSON below and press **Submit new parameter set** to "
            "apply.  Use **Reset everything to defaults** to restore the "
            "factory settings.\n\n"
        )
        current = cfg.get_all()
        display = _nest_config_for_display(current)
        pretty = json.dumps(display, indent=2, ensure_ascii=False)

        with st.form("raw_json_form"):
            edited = st.text_area(
                "user_settings.json", value=pretty, height=600,
            )
            col_submit, col_reset, _ = st.columns([2, 2, 4])
            with col_submit:
                submitted = st.form_submit_button(
                    "Submit new parameter set",
                    type="primary",
                )
            with col_reset:
                reset_clicked = st.form_submit_button(
                    "Reset everything to defaults",
                )

        if submitted:
            try:
                parsed = json.loads(edited)
                parsed = _flatten_config_from_display(parsed)
                cfg.set_many(parsed)
                st.success("Configuration saved — all parameters updated.")
                return True
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
        if reset_clicked:
            cfg.reset()
            st.success("All settings reset to defaults.")
            return True
        return False


# ── Entry point ──────────────────────────────────────────────────────

def RUN():
    page = SettingsPage()
    page.run()


if __name__ == "__main__":
    RUN()
