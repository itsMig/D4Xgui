"""Single source-of-truth version for D4Xgui.

The version string is hardcoded in **only one place**:
``D4Xgui/__init__.py`` (``__version__ = "..."``).

This module reads that value at import time by parsing the sibling
``__init__.py`` with a regex, so it works regardless of how the app is
launched:

* ``streamlit run D4Xgui/Welcome.py`` puts ``D4Xgui/`` on ``sys.path``
  (making ``tools`` a top-level package and ``D4Xgui`` itself not
  importable as a package).
* ``pip install D4Xgui`` puts the project root on ``sys.path``, so
  ``import D4Xgui`` works.
* Sphinx / pytest / arbitrary tooling may set ``sys.path`` differently.

By reading the file directly instead of importing ``D4Xgui``, we avoid
all of these ambiguities.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

_INIT_FILE = Path(__file__).resolve().parent.parent / "__init__.py"
_VERSION_RE = re.compile(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', re.MULTILINE)


def _read_version_from_init() -> str:
    try:
        text = _INIT_FILE.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    m = _VERSION_RE.search(text)
    return m.group(1) if m else "unknown"


__version__: Final[str] = _read_version_from_init()
