"""Keyboard shortcut: Ctrl/Cmd + Enter.

Behaviour:
- If focus is inside the floating chat bubble panel (an st.container whose
  key ends in `_bubble_panel`), click the Ask button inside that panel
  (the st.button whose key ends in `_bubble_ask_btn`). The bubble's Ask
  button does NOT need `type="primary"` for this to work.
- Otherwise, click the first visible page-level primary button
  (`button[kind="primary"]`), preserving the original behaviour for the
  main-page Analyse / Research / Compare / Send buttons.

Pure-JS helper injected via st.components.v1.html so it reaches into
Streamlit's parent document."""

import streamlit as st
import streamlit.components.v1 as components


def bind_ctrl_enter_submit():
    """Call ONCE near the top of a Streamlit page. Binds Ctrl/Cmd+Enter."""
    components.html(
        """
        <script>
        (function(){
          const doc = window.parent.document;
          if (doc.__sb_ctrl_enter_bound) return;
          doc.__sb_ctrl_enter_bound = true;

          function clickIfVisible(el){
            if (!el) return false;
            const r = el.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) {
              el.click();
              return true;
            }
            return false;
          }

          doc.addEventListener('keydown', function(e){
            const isSubmitCombo = (e.ctrlKey || e.metaKey) && e.key === 'Enter';
            if (!isSubmitCombo) return;
            const tag = (e.target.tagName || '').toUpperCase();
            if (!['INPUT','TEXTAREA','SELECT'].includes(tag) && !e.target.isContentEditable) return;

            // 1) SCOPED bubble: if focus is inside an st.container whose key
            //    ends with `_bubble_panel`, click that panel's Ask button
            //    (st.button whose key ends with `_bubble_ask_btn`). The Ask
            //    button does NOT need type="primary".
            const panel = e.target.closest('[class*="st-key-"][class*="_bubble_panel"]');
            if (panel) {
              const askWrap = panel.querySelector('[class*="st-key-"][class*="_bubble_ask_btn"]');
              const askBtn = askWrap ? askWrap.querySelector('button') : null;
              if (askBtn && !askBtn.disabled && clickIfVisible(askBtn)) {
                e.preventDefault();
                return;
              }
              // Focus is inside the bubble but no Ask button was found or it
              // is disabled. Swallow the shortcut so Ctrl+Enter inside the
              // bubble NEVER hijacks the page's Analyse/Research/Compare/Send
              // button.
              e.preventDefault();
              return;
            }

            // 2) PAGE fallback: first visible primary button.
            const btns = doc.querySelectorAll('button[kind="primary"]');
            for (let i = 0; i < btns.length; i++) {
              if (clickIfVisible(btns[i])) {
                e.preventDefault();
                return;
              }
            }
          }, true);
        })();
        </script>
        """,
        height=0,
    )
