"""
3D evidence board for the hero: claim cards on a ring are scanned and stamped
VERIFIED / FLAGGED. A three.js scene in a transparent components iframe
(see static/evidence3d.html). The scene is illustrative and uses no live data.
"""
import os

import streamlit.components.v1 as components

_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "evidence3d.html")


def render_evidence_3d(height=400):
    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=height, scrolling=False)
