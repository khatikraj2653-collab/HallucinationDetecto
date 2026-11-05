"""Keyboard shortcut: Ctrl/Cmd + Enter submits the primary button on the page.
Pure-JS helper injected via st.components.v1.html so it reaches into
Streamlit's parent document and finds the primary-colored button
(st.button type="primary") to trigger a click when the shortcut fires."""

import streamlit as st
import streamlit.components.v1 as components


def bind_ctrl_enter_submit():
    """Call ONCE near the top of a Streamlit page. Finds the first visible
    primary button (st.button type='primary') and clicks it when the user
    presses Ctrl+Enter (Windows/Linux) or Cmd+Enter (Mac) inside any
    input/textarea/select on the page."""
    components.html(
        """
        <script>
        (function(){
          const doc = window.parent.document;
          if (doc.__sb_ctrl_enter_bound) return;
          doc.__sb_ctrl_enter_bound = true;
          doc.addEventListener('keydown', function(e){
            const isSubmitCombo = (e.ctrlKey || e.metaKey) && e.key === 'Enter';
            if (!isSubmitCombo) return;
            // Only when focus is in a form control (so we don't hijack
            // random keyboard events on the page)
            const tag = (e.target.tagName || '').toUpperCase();
            if (!['INPUT','TEXTAREA','SELECT'].includes(tag) && !e.target.isContentEditable) return;
            // Find the first visible primary button
            const btns = doc.querySelectorAll('button[kind="primary"]');
            for (let i = 0; i < btns.length; i++) {
              const r = btns[i].getBoundingClientRect();
              if (r.width > 0 && r.height > 0) {
                e.preventDefault();
                btns[i].click();
                return;
              }
            }
          }, true);
        })();
        </script>
        """,
        height=0,
    )
