"""Floating "Report a bug" button.

Injects a small fixed-position button into the *parent* Streamlit document
(the app frame, not the component iframe) at the top-right of the viewport.

Click behaviour:

1. Uses the browser Screen Capture API
   (``navigator.mediaDevices.getDisplayMedia``) to grab a pixel-perfect
   frame of the current tab. Far more faithful than raster libraries such
   as ``html2canvas`` (which mis-renders Streamlit's web fonts and CSS
   variables) and needs no CDN.
2. Copies the PNG frame to the clipboard via the async Clipboard API
   (``navigator.clipboard.write([new ClipboardItem(...)])``).
3. Opens a ``mailto:`` link prefilled with the current environment
   (D4Xgui, D47crunch, Python, OS, and page label) plus a note asking
   the reporter to paste the screenshot into the email body.

If the Screen Capture API is unavailable or the user cancels the share
prompt, the button still opens the mailto with instructions to attach
a screenshot manually.

Design note
-----------
``st.components.v1.html`` renders inside an iframe that is torn down on
every Streamlit rerun. We therefore only use it as a *bootstrap*: the
inline JS injects the button (and its click handler, via a ``<script>``
tag) into ``window.parent.document`` so they live in the app's own
global context and survive reruns.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import urllib.parse
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from tools import config as cfg
from tools.version import __version__ as D4XGUI_VERSION


_FALLBACK_EMAIL = "mbernecker@posteo.de"

# Stable DOM ids so the bootstrap can find / update the same nodes on rerun.
_BTN_ID = "d4x-bug-fab"
_TOAST_ID = "d4x-bug-fab-toast"
_STYLE_ID = "d4x-bug-fab-style"
_HANDLER_ID = "d4x-bug-fab-handler"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"


def _build_env_block(page_label: str) -> str:
    """Prefilled mailto body describing the client environment."""
    lines = [
        f"D4Xgui: {D4XGUI_VERSION}",
        f"D47crunch: {_pkg_version('D47crunch')}",
        f"Python: {platform.python_version()}",
        f"OS: {platform.system()} {platform.release()}",
        f"Page: {page_label or 'unknown'}",
        "",
        "Please paste the screenshot from your clipboard here and describe the bug.",
    ]
    return "\n".join(lines)


def render_report_bug_button(page_label: Optional[str] = None) -> None:
    """Render the floating "Report a bug" button.

    Args:
        page_label: Human-readable label for the current page (used in the
            mailto body). Falls back to "unknown" when not provided.
    """
    to = cfg.get("bug_report_email", _FALLBACK_EMAIL) or _FALLBACK_EMAIL

    subject = urllib.parse.quote(f"D4Xgui bug report (v{D4XGUI_VERSION})")
    body = urllib.parse.quote(_build_env_block(page_label or ""))
    mailto = f"mailto:{to}?subject={subject}&body={body}"

    # Everything below is data passed to the bootstrap iframe. JSON-encoding
    # keeps quoting safe across the JS/Python boundary.
    payload = json.dumps(
        {
            "mailto": mailto,
            "btnId": _BTN_ID,
            "toastId": _TOAST_ID,
            "styleId": _STYLE_ID,
            "handlerId": _HANDLER_ID,
        }
    )

    bootstrap = f"""
<script>
(function() {{
  const CFG = {payload};

  // Streamlit component iframes are same-origin (allow-same-origin sandbox),
  // so we can reach the parent document and inject persistent nodes there.
  let doc;
  try {{
    doc = window.parent && window.parent.document;
  }} catch (e) {{
    doc = null;
  }}
  if (!doc || !doc.body) return;

  // ── Style (injected once) ──────────────────────────────────────────
  if (!doc.getElementById(CFG.styleId)) {{
    const style = doc.createElement('style');
    style.id = CFG.styleId;
    style.textContent = `
      #${{CFG.btnId}} {{
        position: fixed;
        top: 0.65rem;
        right: 20rem;                   /* clear Streamlit's Deploy / menu
                                           and the "File change" rerun popup */
        z-index: 999999;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.32rem 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(128,128,128,0.35);
        background: var(--background-color, transparent);
        color: var(--text-color, inherit);
        font-family: "Source Sans Pro", "Source Sans 3", -apple-system,
                     BlinkMacSystemFont, sans-serif;
        font-size: 0.875rem;
        font-weight: 400;
        line-height: 1.2;
        cursor: pointer;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        transition: background 120ms ease, border-color 120ms ease;
      }}
      #${{CFG.btnId}}:hover {{
        border-color: var(--primary-color, rgba(128,128,128,0.6));
      }}
      #${{CFG.btnId}}:disabled {{
        opacity: 0.6;
        cursor: progress;
      }}
      #${{CFG.toastId}} {{
        position: fixed;
        top: 2.9rem;
        right: 20rem;
        z-index: 999999;
        max-width: 22rem;
        padding: 0.4rem 0.65rem;
        border-radius: 0.4rem;
        background: rgba(0,0,0,0.78);
        color: #fff;
        font-family: "Source Sans Pro", "Source Sans 3", -apple-system,
                     BlinkMacSystemFont, sans-serif;
        font-size: 0.8rem;
        line-height: 1.3;
        opacity: 0;
        pointer-events: none;
        transition: opacity 150ms ease;
      }}
      #${{CFG.toastId}}.show {{ opacity: 1; }}
    `;
    doc.head.appendChild(style);
  }}

  // ── Button (created once, updated on rerun) ───────────────────────
  let btn = doc.getElementById(CFG.btnId);
  if (!btn) {{
    btn = doc.createElement('button');
    btn.id = CFG.btnId;
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Report a bug');
    btn.innerHTML = '<span aria-hidden="true">🐞</span><span>Report a bug</span>';
    doc.body.appendChild(btn);
  }}
  // Refresh the mailto payload on every rerun (page label may change).
  btn.dataset.mailto = CFG.mailto;

  // ── Toast (created once) ──────────────────────────────────────────
  if (!doc.getElementById(CFG.toastId)) {{
    const toast = doc.createElement('div');
    toast.id = CFG.toastId;
    toast.setAttribute('role', 'status');
    doc.body.appendChild(toast);
  }}

  // ── Handler (injected once, lives in the parent's global scope so it
  //     keeps working after this bootstrap iframe is destroyed) ───────
  if (!doc.getElementById(CFG.handlerId)) {{
    const s = doc.createElement('script');
    s.id = CFG.handlerId;
    s.textContent = `
      (function () {{
        const btn = document.getElementById(${{JSON.stringify(CFG.btnId)}});
        const toast = document.getElementById(${{JSON.stringify(CFG.toastId)}});
        if (!btn || btn.dataset.wired === '1') return;
        btn.dataset.wired = '1';
        let busy = false;
        let toastTimer = null;

        function showToast(msg, ms) {{
          if (!toast) return;
          toast.textContent = msg;
          toast.classList.add('show');
          if (toastTimer) clearTimeout(toastTimer);
          if (ms !== 0) {{
            toastTimer = setTimeout(function () {{
              toast.classList.remove('show');
            }}, ms || 5000);
          }}
        }}

        function openMailto() {{
          const url = btn.dataset.mailto;
          if (!url) return;
          window.location.href = url;
        }}

        async function captureFrame() {{
          if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {{
            throw new Error('Screen Capture API unavailable');
          }}
          const stream = await navigator.mediaDevices.getDisplayMedia({{
            video: {{ cursor: 'never' }},
            audio: false,
            preferCurrentTab: true,
            selfBrowserSurface: 'include',
            surfaceSwitching: 'exclude',
          }});
          try {{
            const track = stream.getVideoTracks()[0];
            const video = document.createElement('video');
            video.autoplay = true;
            video.muted = true;
            video.srcObject = stream;
            await video.play();
            await new Promise(function (r) {{ requestAnimationFrame(r); }});
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 1920;
            canvas.height = video.videoHeight || 1080;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            track.stop();
            return await new Promise(function (resolve, reject) {{
              canvas.toBlob(function (blob) {{
                if (blob) resolve(blob);
                else reject(new Error('canvas.toBlob returned null'));
              }}, 'image/png');
            }});
          }} finally {{
            stream.getTracks().forEach(function (t) {{ t.stop(); }});
          }}
        }}

        async function copyBlob(blob) {{
          if (!navigator.clipboard || !window.ClipboardItem) {{
            throw new Error('Clipboard API unavailable');
          }}
          await navigator.clipboard.write([new ClipboardItem({{ 'image/png': blob }})]);
        }}

        btn.addEventListener('click', async function () {{
          if (busy) return;
          busy = true;
          btn.disabled = true;
          showToast('Select the D4Xgui tab in the share dialog…', 0);
          let blob = null;
          try {{
            blob = await captureFrame();
          }} catch (err) {{
            if (err && (err.name === 'NotAllowedError' || err.name === 'AbortError')) {{
              showToast('Screenshot cancelled — opening email without attachment.');
            }} else {{
              showToast('Screenshot capture failed — opening email without attachment.');
            }}
            openMailto();
            busy = false; btn.disabled = false;
            return;
          }}
          try {{
            await copyBlob(blob);
            showToast('Screenshot copied to clipboard. Paste it into the email body.');
          }} catch (err) {{
            showToast('Screenshot captured but clipboard is unavailable — please attach it manually.');
          }}
          openMailto();
          busy = false; btn.disabled = false;
        }});
      }})();
    `;
    doc.body.appendChild(s);
  }}
}})();
</script>
"""

    # Height=0 so the bootstrap iframe takes no visual space. Streamlit
    # still reserves a tiny bit of vertical room; render it at the very
    # end of the sidebar to keep it out of the way.
    with st.sidebar:
        components.html(bootstrap, height=0, width=0)
