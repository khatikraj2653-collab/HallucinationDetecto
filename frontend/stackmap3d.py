"""
3D stack constellation for the Skills section: the skills in the stack as nodes around
the layers of the architecture (orchestration, detection signals, judges, evaluation...).
A three.js scene in a transparent components iframe (see static/stackmap3d.html).
The links are a hand-written map, not live data.
"""
import os

import streamlit.components.v1 as components

_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "stackmap3d.html")


def render_stack_map(height=600):
    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=height, scrolling=False)
