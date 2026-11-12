"""Acid-fractionation factors for mineral δ18O from acid-liberated CO2.

α is (¹⁸O/¹⁶O)_CO2 / (¹⁸O/¹⁶O)_mineral. The published factors are applied
on the VPDB scale, which is how the carbonate literature uses them
(Kim, Coplen & Horita 2015 for calcite; the same order for the dolomite
and siderite factors): both the CO2 and the mineral δ18O are on VPDB
during

    δ_mineral_VPDB = (1000 + δ_CO2_VPDB) / α − 1000

and only that mineral value is then projected to VSMOW. α is rounded to
6 decimal places. T in the published 1000 ln α expressions is kelvin.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence

KELVIN_OFFSET = 273.15

DEFAULT_ACID_FRACTIONATION = "calcite_kim2007"
DEFAULT_ACID_FRACTIONATIONS = [DEFAULT_ACID_FRACTIONATION]


def _kim2007_calcite(temp_k: float) -> float:
    """Kim, Mucci & Taylor 2007, calcite."""
    return 3.59e3 / temp_k - 1.79


def _kim2007_aragonite(temp_k: float) -> float:
    """Kim, Mucci & Taylor 2007, aragonite."""
    return 3.39e3 / temp_k - 0.83


def _rosenbaum1986_dolomite(temp_k: float) -> float:
    """Rosenbaum & Sheppard 1986, dolomite."""
    return 6.65e5 / temp_k**2 + 4.23


def _rosenbaum1986_siderite(temp_k: float) -> float:
    """Rosenbaum & Sheppard 1986, siderite (25–150 °C fit)."""
    return 6.84e5 / temp_k**2 + 3.85


def _fernandez2016_siderite(temp_k: float) -> float:
    """Fernandez et al. 2016, sealed-vessel siderite."""
    return 4.06e3 / temp_k - 1.93


ACID_FRACTIONATIONS: List[Dict[str, object]] = [
    {
        "id": "calcite_kim2007",
        "label": "Calcite — Kim, Mucci & Taylor 2007",
        "mineral": "calcite",
        "tag": "Kim2007",
        "ln1000_alpha": _kim2007_calcite,
    },
    {
        "id": "aragonite_kim2007",
        "label": "Aragonite — Kim, Mucci & Taylor 2007",
        "mineral": "aragonite",
        "tag": "Kim2007",
        "ln1000_alpha": _kim2007_aragonite,
    },
    {
        "id": "dolomite_rosenbaum1986",
        "label": "Dolomite — Rosenbaum & Sheppard 1986",
        "mineral": "dolomite",
        "tag": "Rosenbaum1986",
        "ln1000_alpha": _rosenbaum1986_dolomite,
    },
    {
        "id": "siderite_rosenbaum1986",
        "label": "Siderite — Rosenbaum & Sheppard 1986",
        "mineral": "siderite",
        "tag": "Rosenbaum1986",
        "ln1000_alpha": _rosenbaum1986_siderite,
    },
    {
        "id": "siderite_fernandez2016",
        "label": "Siderite — Fernandez et al. 2016 (sealed vessel)",
        "mineral": "siderite",
        "tag": "Fernandez2016",
        "ln1000_alpha": _fernandez2016_siderite,
    },
]

_BY_ID = {item["id"]: item for item in ACID_FRACTIONATIONS}


def resolve_fractionation(choice_id: str) -> Dict[str, object]:
    """Return the catalog entry, falling back to the calcite default."""
    return _BY_ID.get(choice_id, _BY_ID[DEFAULT_ACID_FRACTIONATION])


def selected_fractionations(value) -> List[str]:
    """Valid choice ids, in the order given.

    A stored string from the earlier single-choice setting is accepted.
    An empty or unknown value falls back to calcite.
    """
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        value = []
    chosen: List[str] = []
    for item in value:
        if item in _BY_ID and item not in chosen:
            chosen.append(item)
    return chosen or list(DEFAULT_ACID_FRACTIONATIONS)


def column_stems(choice_ids: Sequence[str]) -> List[str]:
    """Summary-column stem for each choice.

    A mineral chosen once is ``calcite``. Two equations for the same
    mineral are ``siderite_Rosenbaum1986`` and ``siderite_Fernandez2016``.
    """
    ids = selected_fractionations(choice_ids)
    counts: Dict[str, int] = {}
    for choice_id in ids:
        mineral = mineral_name(choice_id)
        counts[mineral] = counts.get(mineral, 0) + 1
    stems: List[str] = []
    for choice_id in ids:
        mineral = mineral_name(choice_id)
        if counts[mineral] == 1:
            stems.append(mineral)
        else:
            stems.append(f"{mineral}_{resolve_fractionation(choice_id)['tag']}")
    return stems


def mineral_d18o_vpdb(d18o_co2_vpdb, alpha: float):
    """Carbonate δ18O on VPDB from CO2 δ18O already on VPDB."""
    return (1000.0 + d18o_co2_vpdb) / alpha - 1000.0


def vsmow_from_vpdb(d18o_vpdb, vpdb_factor: float, vpdb_offset: float):
    """Project a VPDB δ18O onto VSMOW. Inverse of the app's VPDB conversion."""
    return (d18o_vpdb + vpdb_offset) / vpdb_factor


def acid_alpha(choice_id: str, temp_c: float) -> float:
    """Phosphoric-acid 18O fractionation factor at *temp_c* (°C)."""
    entry = resolve_fractionation(choice_id)
    ln1000: Callable[[float], float] = entry["ln1000_alpha"]  # type: ignore[assignment]
    temp_k = float(temp_c) + KELVIN_OFFSET
    return round(math.exp(ln1000(temp_k) / 1000.0), 6)


def mineral_name(choice_id: str) -> str:
    return str(resolve_fractionation(choice_id)["mineral"])


def fractionation_label(choice_id: str) -> str:
    return str(resolve_fractionation(choice_id)["label"])
