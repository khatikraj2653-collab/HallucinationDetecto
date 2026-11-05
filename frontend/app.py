import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import threading

import streamlit as st
import sys as _sys, os as _os
_sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
from keyboard import bind_ctrl_enter_submit
bind_ctrl_enter_submit()

# Full-screen loader on FIRST visit only (hides heavy transformers/torch
# import cost). Subsequent reruns (Analyze clicks) show a tiny inline
# spinner so clicking Analyze doesn't black out the whole page while
# the LLM judges and NLI model run.
_loading_placeholder = st.empty()
if "_hd_landing_loader_shown" not in st.session_state:
    st.session_state._hd_landing_loader_shown = True
    _loading_placeholder.markdown(
        """
        <div style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center;
                    background:#0F1115;z-index:999999;flex-direction:column;gap:14px;">
          <div style="width:44px;height:44px;border:3px solid rgba(167,139,250,.2);
                      border-top-color:#A78BFA;border-radius:50%;
                      animation:hd-spin 0.9s linear infinite;"></div>
          <div style="color:#C7D1DF;font-family:Inter,sans-serif;font-size:14px;">
            Loading Hallucination Detector…
          </div>
          <style>@keyframes hd-spin{to{transform:rotate(360deg)}}</style>
        </div>
        """,
        unsafe_allow_html=True,
    )

from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv
import re

from db_utils import init_db, log_claim
from rag_utils import build_vectorstore, check_claim_grounding
from refusal_utils import is_refusal
from llm_judge import multi_llm_judge, multi_consistency_check
from consistency_utils import generate_multiple_answers
from scoring_utils import compute_final_verdict
from graph_utils import analyze_answers_parallel
from log_client import log_event
from evidence3d import render_evidence_3d
from stackmap3d import render_stack_map

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "HallucinationDetector")
init_db()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SSRN_URL = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7200222"
DEMO_URL = "https://hallucinationdetecto.streamlit.app"

st.set_page_config(page_title="Hallucination Detector", page_icon="🔮", layout="wide")

# SEO tags injected into the real top-level document (st.markdown runs in
# Streamlit's own DOM, not a sandboxed iframe, so this is what Google's
# crawler actually sees for this page's <head>).
st.markdown("""
<script>
(function() {
  document.title = "Hallucination Detector — LLM Answer Grounding Checker";
  function setMeta(attr, key, content) {
    let el = document.querySelector(`meta[${attr}="${key}"]`);
    if (!el) { el = document.createElement('meta'); el.setAttribute(attr, key); document.head.appendChild(el); }
    el.setAttribute('content', content);
  }
  const desc = "Checks whether an LLM's answer is actually grounded in its source text, using multi-model consistency and NLI-based claim verification, built by Raj Tejpal Khatik.";
  setMeta('name', 'description', desc);
  setMeta('property', 'og:title', "Hallucination Detector — LLM Answer Grounding Checker");
  setMeta('property', 'og:description', desc);
  setMeta('property', 'og:type', 'website');
  setMeta('property', 'og:url', 'https://hallucinationdetecto.streamlit.app/');
  setMeta('property', 'og:image', 'https://raw.githubusercontent.com/khatikraj2653-collab/HallucinationDetecto/main/hallucinationdetector-demo.gif');
  setMeta('property', 'og:image:width', '1200');
  setMeta('property', 'og:image:height', '630');
  setMeta('property', 'og:image:alt', 'Hallucination Detector — LLM Answer Grounding Checker');
  setMeta('name', 'twitter:card', 'summary_large_image');
  setMeta('name', 'twitter:title', "Hallucination Detector — LLM Answer Grounding Checker");
  setMeta('name', 'twitter:description', desc);
  setMeta('name', 'twitter:image', 'https://raw.githubusercontent.com/khatikraj2653-collab/HallucinationDetecto/main/hallucinationdetector-demo.gif');

  let canonical = document.querySelector('link[rel="canonical"]');
  if (!canonical) { canonical = document.createElement('link'); canonical.setAttribute('rel', 'canonical'); document.head.appendChild(canonical); }
  canonical.setAttribute('href', 'https://hallucinationdetecto.streamlit.app/');

  if (!document.getElementById('hallucinationdetector-jsonld')) {
    const s = document.createElement('script');
    s.type = 'application/ld+json';
    s.id = 'hallucinationdetector-jsonld';
    s.textContent = JSON.stringify({
      "@context": "https://schema.org",
      "@type": "SoftwareApplication",
      "name": "Hallucination Detector",
      "applicationCategory": "DeveloperApplication",
      "operatingSystem": "Web",
      "url": "https://hallucinationdetecto.streamlit.app/",
      "description": desc,
      "author": {
        "@type": "Person",
        "name": "Raj Tejpal Khatik",
        "sameAs": [
          "https://www.linkedin.com/in/raj-khatik-6ab086395",
          "https://github.com/khatikraj2653-collab",
          "https://portfolio-raj.pages.dev/"
        ]
      }
    });
    document.head.appendChild(s);
  }
})();
</script>
""", unsafe_allow_html=True)

# Google Analytics (GA4)
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=G-YRDTHQV4D6"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-YRDTHQV4D6');
</script>
""", unsafe_allow_html=True)

if "visit_logged" not in st.session_state:
    st.session_state.visit_logged = True
    log_event("visit", detail="app page")


@st.cache_resource
def load_nli_model():
    return pipeline("text-classification", model="facebook/bart-large-mnli", device=-1, model_kwargs={"low_cpu_mem_usage": False})


nli_model = load_nli_model()
nli_model_lock = threading.Lock()

st.html(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Newsreader:ital,wght@0,500;0,600;0,700;1,500&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  /* Forensic lab: paper, ink, evidence-red. Legacy variable names are remapped so every existing var() keeps working. */
  --paper:#F4F1EA; --card:#FBF9F4; --ink:#14110F; --sub:#5C554B; --dim:#8A8272; --line:#D8D1C0; --line-2:#BDB5A1;
  --red:#D6362B; --red-dk:#8E1F18; --green:#1F8A5B; --amber:#B7791F;
  --bg:var(--paper); --bg2:var(--card); --panel:var(--card); --border:var(--line);
  --violet:var(--red); --cyan:var(--red); --pink:var(--red);
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --serif:'Newsreader',Georgia,'Times New Roman',serif;
}
html, body, [class*="css"]{ font-family:'Inter',sans-serif; }
.stApp{
  background:radial-gradient(rgba(20,17,15,0.085) 1px, transparent 1.3px) 0 0/22px 22px, var(--paper);
  color:var(--ink);
}
.block-container{ padding-top:2.2rem; max-width:1150px; }
h1,h2,h3,h4{ font-family:var(--serif) !important; color:var(--ink) !important; letter-spacing:-0.01em; }
header[data-testid="stHeader"]{ background:transparent !important; }
[data-testid="stDecoration"]{ background:linear-gradient(90deg,var(--red),var(--ink)) !important; height:3px !important; }
[data-testid="stToolbar"]{ background:transparent !important; }

/* ---------- Hero: case file ---------- */
.hd-badge{display:inline-flex;align-items:center;gap:8px;border:1.5px solid var(--red);border-radius:3px;padding:5px 12px;
  font-family:var(--mono);font-size:11px;font-weight:600;color:var(--red);margin-bottom:18px;letter-spacing:0.12em;text-transform:uppercase;
  transform:rotate(-1.2deg);background:rgba(214,54,43,0.05);}
.hd-dot{width:7px;height:7px;border-radius:50%;background:var(--red);display:inline-block;animation:hdblink 1.8s ease-in-out infinite;}
@keyframes hdblink{0%,100%{opacity:0.25;}50%{opacity:1;}}
.hd-title{font-family:var(--serif);font-size:3rem;font-weight:700;line-height:1.08;letter-spacing:-0.02em;color:var(--ink);margin-bottom:16px;}
.hd-title em{font-style:normal;color:var(--ink);background:linear-gradient(transparent 56%, rgba(214,54,43,0.32) 56%, rgba(214,54,43,0.32) 92%, transparent 92%);padding:0 .1em;}
.hd-desc{font-size:1rem;color:var(--sub);line-height:1.8;max-width:760px;margin-bottom:22px;}
.hd-desc strong{color:var(--ink);font-weight:600;}
.hd-chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:26px;}
.hd-chip{font-family:var(--mono);background:transparent;border:1px solid var(--line-2);color:var(--ink);border-radius:3px;padding:4px 11px;font-size:11.5px;font-weight:500;}
.hd-stats{display:flex;gap:38px;flex-wrap:wrap;padding-top:18px;border-top:1.5px solid var(--ink);margin-bottom:8px;}
.hd-stat-num{font-family:var(--mono);font-size:1.5rem;font-weight:600;color:var(--ink);letter-spacing:-0.02em;}
.hd-stat-lbl{font-family:var(--mono);font-size:10px;color:var(--dim);margin-top:3px;font-weight:500;letter-spacing:0.1em;text-transform:uppercase;}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--line-2); }
.stTabs [data-baseweb="tab"]{
  background:transparent; border-radius:0; color:var(--sub); font-family:var(--mono); font-weight:600; font-size:12.5px; letter-spacing:0.08em;
  text-transform:uppercase; padding:10px 16px;
}
.stTabs [aria-selected="true"]{ color:var(--ink) !important; background:transparent !important; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--red) !important; height:3px !important; }
.stTabs [data-baseweb="tab-border"]{ background:transparent !important; }

/* ---------- Cards / inputs ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--card) !important;border:1px solid var(--line) !important;border-radius:6px !important;
  padding:6px 6px !important;
}
.stTextArea textarea, .stTextInput input{
  background:#FFFEFA !important;border:1px solid var(--line-2) !important;border-radius:4px !important;color:var(--ink) !important;
}
.stTextArea textarea:focus, .stTextInput input:focus{ border-color:var(--ink) !important; box-shadow:0 0 0 1px var(--ink) !important; }
label p{ font-family:var(--mono) !important; color:var(--sub) !important; font-weight:600 !important; font-size:11.5px !important; letter-spacing:0.1em; text-transform:uppercase; }

button[kind="primary"], a[kind="primary"]{
  background:var(--ink) !important;border:none !important;border-radius:4px !important;
  box-shadow:0 3px 0 #000, 0 12px 20px -10px rgba(20,17,15,0.5) !important;transition:transform .12s,box-shadow .12s,background .15s;
}
button[kind="primary"] p, a[kind="primary"] p{ color:var(--paper) !important; font-family:var(--mono) !important; font-weight:600 !important; letter-spacing:0.08em; text-transform:uppercase; font-size:12.5px !important; }
button[kind="primary"]:hover, a[kind="primary"]:hover{ background:var(--red) !important; box-shadow:0 5px 0 var(--red-dk), 0 14px 22px -10px rgba(214,54,43,0.5) !important; transform:translateY(-2px); }
button[kind="primary"]:active, a[kind="primary"]:active{ transform:translateY(2px); box-shadow:0 1px 0 var(--red-dk) !important; }
button[kind="secondary"]{ background:transparent !important;border:1px solid var(--ink) !important;border-radius:4px !important;color:var(--ink) !important; }
button[kind="secondary"]:hover{ border-color:var(--red) !important; color:var(--red) !important; }

/* ---------- Answer result cards ---------- */
.hd-ans-head{font-family:var(--mono);font-weight:600;color:var(--ink);font-size:12px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;}
.hd-ans-text{font-size:12.5px;color:var(--sub);line-height:1.65;margin-bottom:10px;}
.hd-claim{font-family:var(--serif);font-size:13px;color:var(--ink);font-style:italic;margin:10px 0 4px;padding-left:12px;border-left:3px solid var(--line-2);animation:hdslide .4s ease both;}
@keyframes hdslide{from{opacity:0;transform:translateX(-6px);}to{opacity:1;transform:none;}}
.hd-verdict{display:inline-flex;align-items:center;gap:6px;border-radius:3px;padding:5px 12px;font-family:var(--mono);font-size:12px;font-weight:700;letter-spacing:0.08em;
  text-transform:uppercase;margin-top:6px;border:2px solid currentColor;transform:rotate(-1.5deg);animation:hdstamp .35s cubic-bezier(.2,1.4,.4,1) both;}
@keyframes hdstamp{from{opacity:0;transform:rotate(-6deg) scale(1.5);}to{opacity:1;transform:rotate(-1.5deg) scale(1);}}
.hd-verdict.hd-green{background:rgba(31,138,91,0.08);color:var(--green);}
.hd-verdict.hd-orange{background:rgba(183,121,31,0.09);color:var(--amber);}
.hd-verdict.hd-red{background:rgba(214,54,43,0.08);color:var(--red);}

/* ---------- Research tab ---------- */
.hd-rp-cat{font-family:var(--mono);font-size:10.5px;font-weight:600;letter-spacing:0.14em;color:var(--red);text-transform:uppercase;margin-bottom:8px;}
.hd-rp-title{font-family:var(--serif);font-size:1.7rem;font-weight:700;color:var(--ink);line-height:1.25;margin-bottom:10px;}
.hd-rp-authors{font-size:13px;color:var(--ink);margin-bottom:4px;}
.hd-rp-venue{font-size:12px;color:var(--dim);margin-bottom:18px;}
.hd-rp-highlight{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--red);border-radius:3px;padding:16px 20px;margin:14px 0;color:var(--ink);font-size:13.5px;line-height:1.8;}
.stTable, .stDataFrame{ border-radius:4px; overflow:hidden; }
hr{ border-color:var(--line-2) !important; }

/* ---------- Contact ---------- */
.hd-contact-wrap{margin:18px 0 28px;}
.hd-contact-title{font-family:var(--mono);font-size:0.78rem;font-weight:600;color:var(--sub);margin-bottom:10px;text-transform:uppercase;letter-spacing:0.14em;}
.hd-contact-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line-2);border:1px solid var(--line-2);border-radius:4px;overflow:hidden;}
.hd-contact-grid a{background:var(--card);padding:14px 16px;display:flex;flex-direction:column;gap:4px;text-decoration:none;transition:background .15s ease;}
.hd-contact-grid a:hover{background:#F0EBDD;}
.hd-contact-label{font-family:var(--mono);font-size:9.5px;text-transform:uppercase;letter-spacing:0.12em;color:var(--dim);font-weight:500;}
.hd-contact-value{font-size:12.5px;font-weight:600;color:var(--ink);word-break:break-word;}
@media (max-width:760px){ .hd-contact-grid{grid-template-columns:1fr 1fr;} }
@media (max-width:480px){ .hd-contact-grid{grid-template-columns:1fr;} }
/* ---------- 3D depth ---------- */
.hd-title{text-shadow:1px 1px 0 #DAD3C1,2px 2px 0 #D0C8B3,3px 3px 0 #C6BDA6;}
.hd-chip{box-shadow:0 2px 0 var(--line-2);}
[data-testid="stVerticalBlockBorderWrapper"]{
  box-shadow:0 1px 0 #D8D1C0,0 2px 0 #FBF9F4,0 3px 0 #D3CBB8,0 4px 0 #F4F1EA,0 5px 0 #CFC7B2,0 28px 40px -26px rgba(20,17,15,0.55) !important;
  transition:transform .25s cubic-bezier(.2,.8,.2,1);
}
[data-testid="stVerticalBlockBorderWrapper"]:hover{ transform:translateY(-2px); }
.hd-verdict{box-shadow:2px 3px 0 rgba(20,17,15,0.18);}

@media (prefers-reduced-motion:reduce){ .hd-verdict,.hd-claim,.hd-dot{animation:none !important;} [data-testid="stVerticalBlockBorderWrapper"]{transition:none;} }

footer{visibility:hidden;}
</style>
"""
)

hero_l, hero_r = st.columns([6, 5], gap="medium")
with hero_l:
    st.markdown(
        """
<div class="hd-badge"><span class="hd-dot"></span> LIVE · 6-Signal Ensemble Detection</div>
<div class="hd-title">Catch What LLMs <em>Make Up</em></div>
<div class="hd-desc">A multi-method ensemble that flags hallucinated claims by combining <strong>NLI entailment</strong>, <strong>RAG-grounding</strong>, a <strong>three-model LLM judge panel</strong> (OpenAI, Groq, Gemini), and <strong>cross-sample consistency</strong> checking &mdash; run in parallel as a LangGraph state machine, so no single blind spot decides the verdict alone.</div>
<div class="hd-chips">
  <span class="hd-chip">LangGraph</span><span class="hd-chip">NLI · BART-MNLI</span><span class="hd-chip">RAG · FAISS</span>
  <span class="hd-chip">GPT-4o-mini Judge</span><span class="hd-chip">Groq Llama 3.3 70B Judge</span><span class="hd-chip">Gemini 2.0 Flash Judge</span>
</div>
""",
        unsafe_allow_html=True,
    )
with hero_r:
    render_evidence_3d(height=400)
st.markdown(
    """
<div class="hd-stats">
  <div><div class="hd-stat-num">6</div><div class="hd-stat-lbl">Detection Signals</div></div>
  <div><div class="hd-stat-num">193</div><div class="hd-stat-lbl">Eval Instances</div></div>
  <div><div class="hd-stat-num">92.2%</div><div class="hd-stat-lbl">TruthfulQA Accuracy</div></div>
  <div><div class="hd-stat-num">SSRN</div><div class="hd-stat-lbl">Live Preprint</div></div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")
tab_analyze, tab_paper = st.tabs(["🔍  Analyze", "📄  Research Paper"])

st.markdown(
    """
<div class="hd-contact-wrap">
  <div class="hd-contact-title">Contact</div>
  <div class="hd-contact-grid">
    <a href="mailto:khatikraj2653@gmail.com">
      <span class="hd-contact-label">Email</span>
      <span class="hd-contact-value">khatikraj2653@gmail.com</span>
    </a>
    <a href="https://www.linkedin.com/in/raj-khatik-6ab086395" target="_blank" rel="noopener">
      <span class="hd-contact-label">LinkedIn</span>
      <span class="hd-contact-value">raj-khatik-6ab086395</span>
    </a>
    <a href="https://github.com/khatikraj2653-collab" target="_blank" rel="noopener">
      <span class="hd-contact-label">GitHub</span>
      <span class="hd-contact-value">khatikraj2653-collab</span>
    </a>
    <a href="https://wa.me/447799394985" target="_blank" rel="noopener">
      <span class="hd-contact-label">Phone / WhatsApp</span>
      <span class="hd-contact-value">+44 7799 394985</span>
    </a>
    <a href="mailto:Raj.Khatik@warwick.ac.uk">
      <span class="hd-contact-label">University email</span>
      <span class="hd-contact-value">Raj.Khatik@warwick.ac.uk</span>
    </a>
    <a href="https://portfolio-raj.pages.dev" target="_blank" rel="noopener">
      <span class="hd-contact-label">Portfolio</span>
      <span class="hd-contact-value">portfolio-raj.pages.dev</span>
    </a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with tab_analyze:
    with st.container(border=True):
        context = st.text_area(
            "Context",
            height=180,
            placeholder=(
                "Paste the source passage here — a paragraph from an article, "
                "Wikipedia, a research paper, a document excerpt, etc.\n\n"
                "Example:\n"
                "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889 "
                "for the World's Fair in Paris. It stands 330 metres tall and was the "
                "world's tallest man-made structure for 41 years."
            ),
        )
        question = st.text_input(
            "Question",
            placeholder="A specific factual question answerable from the context above — e.g. When was the Eiffel Tower built?",
        )
        analyze_clicked = st.button("Analyze", type="primary")

    def split_claims(answer):
        sentences = re.split(r'(?<!\d)\.(?!\d)', answer)
        return [s.strip() for s in sentences if s.strip()]

    def _normalize_for_nli(text: str) -> str:
        """Normalize punctuation/casing so BART-MNLI isn't thrown by trivial
        formatting differences between a short lowercase context and a
        properly-capitalized model answer. We preserve meaning fully — just
        add sentence-case and a terminating period when they're missing."""
        t = (text or "").strip()
        if not t:
            return t
        if t[-1] not in ".!?":
            t = t + "."
        # Sentence-case only if the whole string is lowercase (so we don't
        # mangle things like "ASML Holding" or "The Prime Minister of India")
        if t == t.lower():
            t = t[0].upper() + t[1:]
        return t

    def check_claim(context, claim):
        c_norm = _normalize_for_nli(context)
        k_norm = _normalize_for_nli(claim)
        with nli_model_lock:
            result = nli_model(f"{c_norm}</s></s>{k_norm}")[0]
        return result["label"], round(result["score"], 3)

    if analyze_clicked:
        if not context or not question:
            st.warning("Please fill in both Context and Question.")
        elif len(context.strip().split()) < 5 or len(question.strip().split()) < 2:
            st.warning(
                "⚠️ Context and Question look very short. This tool is designed to fact-check "
                "LLM answers against a **source passage** (a paragraph, article excerpt, or document "
                "snippet), not casual greetings. Try pasting a real passage (e.g. a few sentences from "
                "Wikipedia, a news article, or research paper) and asking a specific factual question "
                "about it — e.g. context: '*The Eiffel Tower was built in 1889 and is 330 metres tall.*', "
                "question: '*When was the Eiffel Tower built?*'"
            )
        else:
            st.markdown(f"##### 🌳 Context")
            st.caption(f"↳ {question}")

            with st.spinner("Generating 3 answers..."):
                answers = generate_multiple_answers(context, question, n=3)

            with st.spinner("Checking cross-sample consistency..."):
                consistency_results = multi_consistency_check(answers[0], answers[1], answers[2])
                consistent_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper() and "CONSISTENT" in v.upper() and "INCONSISTENT" not in v.upper())
                total_votes = sum(1 for v in consistency_results.values() if "ERROR" not in v.upper())
                is_consistent = (consistent_votes >= total_votes / 2) if total_votes > 0 else True

            st.divider()

            vectorstore = build_vectorstore(context)

            with st.spinner("Running parallel detection across all 3 answers..."):
                all_results = analyze_answers_parallel(context, answers, vectorstore, check_claim, split_claims)

            cols = st.columns(3)

            for i, (ans, col, claim_results) in enumerate(zip(answers, cols, all_results), 1):
                with col, st.container(border=True):
                    st.markdown(f'<div class="hd-ans-head">Answer {i}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="hd-ans-text">{ans}</div>', unsafe_allow_html=True)

                    bad_votes = 0
                    total_signals = 0

                    for result in claim_results:
                        claim = result["claim"]
                        claim_display = claim[:40] + "..." if len(claim) > 40 else claim
                        st.markdown(f'<div class="hd-claim">{claim_display}</div>', unsafe_allow_html=True)

                        if result["is_refusal"]:
                            st.badge("refusal — skipped", color="gray")
                            continue

                        nli_flagged = result["nli_label"].upper() == "CONTRADICTION"
                        with st.popover(f"NLI: {result['nli_label']}"):
                            st.write(f"Confidence: {result['nli_score']}")
                        total_signals += 1
                        if nli_flagged:
                            bad_votes += 1

                        with st.popover(f"RAG: {'Grounded' if result['is_grounded'] else 'Not grounded'}"):
                            st.write(f"Score: {result['rag_score']}")
                            st.write(f"Closest chunk: _{result['best_chunk']}_")
                        total_signals += 1
                        if not result["is_grounded"]:
                            bad_votes += 1

                        judge_results = result["judge_results"]
                        unsupported = sum(1 for v in judge_results.values() if "ERROR" not in v.upper() and "UNSUPPORTED" in v.upper())
                        valid_judges = sum(1 for v in judge_results.values() if "ERROR" not in v.upper())
                        with st.popover(f"Judges: {unsupported}/{valid_judges} flagged"):
                            for jn, jv in judge_results.items():
                                st.write(f"**{jn}:** {jv}")
                        total_signals += valid_judges
                        bad_votes += unsupported

                        log_claim(context, question, ans, claim, result["nli_label"], result["nli_score"])

                    total_signals += 1
                    if not is_consistent:
                        bad_votes += 1
                    st.caption(f"Consistency: {'✅ stable' if is_consistent else '🚩 unstable'}")

                    answer_score = round((bad_votes / total_signals) * 100, 1) if total_signals > 0 else 0.0
                    verdict_class = "hd-red" if answer_score >= 67 else "hd-orange" if answer_score >= 34 else "hd-green"
                    st.markdown(f'<div class="hd-verdict {verdict_class}">{answer_score}% flagged</div>', unsafe_allow_html=True)

            st.divider()
            st.caption("Click any signal badge above to expand its detail. Powered by LangGraph subgraphs with parallel threading — trace visible in LangSmith.")

with tab_paper:
    st.markdown('<div class="hd-rp-cat">SSRN Preprint · 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="hd-rp-title">Multi-Method Ensemble Framework for LLM Hallucination Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="hd-rp-authors">Raj Tejpal Khatik · MSc Applied AI, University of Warwick</div>', unsafe_allow_html=True)
    st.markdown('<div class="hd-rp-venue">Live preprint on SSRN · Currently under review at a peer-reviewed journal</div>', unsafe_allow_html=True)

    c1, c2, _ = st.columns([1.1, 1, 3])
    with c1:
        st.link_button("📄 Read Live on SSRN ↗", SSRN_URL, type="primary")
    with c2:
        st.link_button("Live Demo ↗", DEMO_URL)

    st.divider()

    st.markdown("#### Abstract")
    st.markdown(
        """Large language models (LLMs) frequently generate fluent but factually unsupported content — a failure mode
known as hallucination — undermining their reliability in high-stakes applications. Existing detection
approaches typically rely on a single signal, such as retrieval grounding (Lewis et al., 2020) or self-consistency
sampling (Manakul et al., 2023), leaving them vulnerable to failure modes the chosen method was not designed
to catch. This work proposes an equal-weight, six-signal ensemble framework combining natural language
inference (NLI), retrieval-augmented grounding, a three-model LLM-judge panel (OpenAI, Groq, Gemini), and
cross-sample self-consistency checking, implemented as a LangGraph state-machine pipeline with parallel
execution."""
    )
    st.markdown(
        """Evaluated on 76 HaluEval-QA examples, 77 TruthfulQA examples, and a 40-instance hand-labeled dataset
spanning four domains, the system achieves 72.4% (95% CI: 61.4–81.2%), 92.2% (95% CI: 84.0–96.4%), and
87.5% accuracy respectively. McNemar's exact test applied to a 37-instance ablation study found no
statistically significant difference between any signal-subset configuration (all p > 0.05), indicating the study
is underpowered to confirm individual signal contributions despite an 8.2-point raw accuracy drop when adding
RAG-grounding to an NLI-only baseline."""
    )
    st.markdown(
        """Inter-judge agreement analysis (n=38 claims) revealed Groq and Gemini agree with each other substantially
more often (84.6%) than either agrees with OpenAI (71.1%, 73.1%), identifying OpenAI's judge as a
conservative outlier within the panel. A quadrant analysis crossing detector-flagged status with sample-consistency
status found that claims flagged by detectors AND consistent across samples were real hallucinations 100% of
the time (n=4), outperforming the system's own "high-confidence" category (flagged + inconsistent, 66.7%
precision), challenging the original architectural assumption that instability signals maximum confidence."""
    )
    st.markdown(
        """Fifteen hand-crafted adversarial examples further expose concrete, reproducible failure modes: in 8 of 15
cases, RAG-grounding produced false positives on true-but-context-absent claims; self-consistency proved
insensitive to superficial phrasing variation independent of factual content in 2 cases; and a sentence-splitting
parser artifact was identified as a genuine implementation limitation. This work contributes a working, deployed
detection system and an empirically grounded account of where individual detection methods fail, do not fail,
and where ensembling does and does not recover those failures."""
    )

    st.markdown("#### 1. Introduction")
    st.markdown(
        """LLMs are known to generate confident, fluent text that is not grounded in any verifiable source — a problem
documented across summarization, dialogue, and open-domain question answering. As LLMs are increasingly
deployed in domains such as finance, law, and education, undetected hallucination carries direct real-world risk,
compounded by the fact that hallucinated content is often stylistically indistinguishable from accurate content."""
    )
    st.markdown(
        """A growing body of work addresses hallucination detection through distinct, largely independent strategies:
entailment-based checking, retrieval-based fact verification, LLM-as-judge evaluation, and sampling-based
consistency checking. Each has documented strengths and blind spots, but few studies combine them into a
single evaluated system, apply formal significance testing to ablation claims, or report cases where combining
methods provides no measurable benefit or is statistically inconclusive at the tested sample size. This paper
addresses that gap with direct, replicated empirical evidence, including negative and null results."""
    )
    st.markdown(
        """**Contributions:**
1. A six-signal, equal-weight ensemble detection framework, implemented as a LangGraph subgraph with parallel execution.
2. Evaluation across two public benchmarks and one independently constructed hand-labeled dataset, with precision, recall, F1, and Wilson-score confidence intervals throughout.
3. An ablation study with McNemar's exact significance testing, revealing that observed accuracy differences are not statistically confirmed at the tested sample size.
4. Quantified inter-judge agreement, identifying a consistent outlier model within the judge panel.
5. An empirical quadrant analysis testing — and partially falsifying — the system's own architectural assumptions about confidence ordering.
6. Fifteen hand-crafted adversarial examples with full per-signal breakdowns, including two genuine implementation bugs discovered and corrected during evaluation."""
    )

    st.markdown("#### 2. Related Work")
    st.markdown(
        """**Entailment-based detection.** NLI models classify a hypothesis as entailed, contradicted, or neutral given a
premise, and have been repurposed for factual consistency checking in summarization, where entailment scores
correlate with human faithfulness judgments better than n-gram overlap (Kryscinski et al., 2020). Section 6
documents NLI's "neutral" blind spot across multiple adversarial examples, while also documenting a negative
result: negation alone did not reliably degrade NLI performance, refining the scope of this limitation."""
    )
    st.markdown(
        """**Retrieval-augmented grounding.** Retrieval-Augmented Generation (Lewis et al., 2020) conditions generation
on retrieved passages; the same mechanism can be repurposed post-hoc to verify claim support. Sections 5 and 6
provide the paper's most consistent finding: RAG-grounding produced false positives in 8 of 15 targeted
adversarial tests, specifically on claims that are factually true but absent from the narrow source context,
revealing that topical similarity is systematically conflated with factual support."""
    )
    st.markdown(
        """**Self-consistency sampling.** SelfCheckGPT (Manakul et al., 2023) assumes hallucinated facts diverge across
stochastically sampled responses. Section 6 reports two independent counterexamples: a confidently memorized
fabrication that remained stable across samples (evading detection), and cases where consistency checking
flagged instability from superficial phrasing variation despite identical factual content — a previously
undocumented failure mode in the opposite direction."""
    )
    st.markdown(
        """**LLM-as-judge evaluation.** Correlation between LLM judges and human ratings varies by task; gpt-3.5-turbo
as a judge achieved only low-to-moderate correlation with human judgment (Spearman's rho of 0.27–0.46) on
SummEval and FRANK. This paper's inter-judge agreement analysis (Section 5.4) provides a quantified,
model-specific instance of this variability: OpenAI's judge behaves as a measurable outlier relative to Groq and
Gemini."""
    )
    st.markdown(
        """**Benchmarks.** HaluEval evaluates hallucination recognition using HotpotQA-, OpenDialKG-, and
CNN/Daily Mail-derived samples. TruthfulQA consists of 817 questions across 38 categories designed so some
humans would answer falsely due to common misconceptions. Recent analysis argues TruthfulQA is better
characterized as a factuality benchmark than a hallucination benchmark — supported empirically here by a
non-overlapping confidence-interval gap (Section 5.1) and contextualized by the hand-labeled dataset (Section
5.5), whose difficulty falls between the two."""
    )
    st.markdown(
        """**Gap addressed.** No existing work, to our knowledge, combines these strategies into a single evaluated
system, applies formal significance testing to ablation claims, *and* reports cases where combining methods
provides no measurable benefit."""
    )

    st.markdown("#### 3. System Architecture")
    st.markdown(
        """The system accepts a source context and a question, generates three independently sampled answers from
the same LLM (temperature > 0), and evaluates each claim against six equally weighted signals: (1) NLI
entailment (BART-large-MNLI), (2) RAG-grounding (FAISS dense retrieval, thresholded cosine similarity),
(3–5) a three-model LLM judge panel (OpenAI GPT-4o, Groq Llama-3.3-70B, Google Gemini), each classifying
claims as SUPPORTED / UNSUPPORTED / UNCLEAR, and (6) cross-sample consistency, where the same
three-model panel compares all three sampled answers for stability."""
    )
    st.caption(
        "Note: the live demo's judge panel currently runs gpt-4o-mini (OpenAI), llama-3.3-70b-versatile (Groq) and "
        "gemini-2.0-flash (Google), each with the other two as automatic fallbacks, so live scores can differ "
        "from the models reported in the evaluation above."
    )
    st.markdown(
        """Each signal contributes one vote; a claim's hallucination score is the proportion of signals flagging it,
bucketed at 0–33% (faithful), 34–66% (possible hallucination), and 67–100% (high-confidence hallucination).
Sentences classified as refusals by a dedicated LLM-based classifier are excluded from scoring. The pipeline is
implemented as a LangGraph state graph with a per-claim subgraph (check_refusal → run_detectors),
thread-parallel execution both within a claim and across the three sampled answers, and LangSmith tracing for
observability."""
    )

    st.markdown("#### 4. Evaluation Methodology")
    st.markdown(
        """Three data sources were used: **HaluEval-QA** (40 questions across two independent batches, seeds 42/99,
yielding 76 scoreable instances), **TruthfulQA generation config** (40 questions, 77 scoreable instances, using
the correct answer as source context — a disclosed simplification), and a **hand-labeled dataset** (20 original
questions / 40 scoreable instances across four domains — Indian history, science, finance, geography — each
with a correct answer and a deliberately subtle incorrect answer, after an initial iteration that produced a 100%
ceiling-effect accuracy due to overly obvious errors)."""
    )
    st.markdown(
        """An ablation study (n=37, single HaluEval batch) compared four signal-subset configurations. McNemar's
exact test was applied pairwise on discordant predictions. A quadrant analysis (n=37, independently generated
triple-sampled answers, correcting an initial script defect that fed identical repeated text rather than genuine
samples) crossed detector-flagged status against consistency status. Inter-judge agreement (n=38 claims) was
measured via pairwise verdict comparison across a separate 20-question HaluEval sample. Fifteen adversarial
examples were hand-crafted targeting specific hypothesized weaknesses of individual signals."""
    )

    st.markdown("#### 5. Results")
    st.markdown("**5.1 Overall accuracy**")
    st.markdown(
        """| Dataset | n | Accuracy | 95% CI | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| HaluEval-QA | 76 | 72.4% | 61.4–81.2% | 82.8% | 60.0% | 69.6% |
| TruthfulQA | 77 | 92.2% | 84.0–96.4% | 100.0% | 85.0% | 91.9% |
| Hand-labeled (own) | 40 | 87.5% | — | — | — | — |"""
    )
    st.markdown(
        """The hand-labeled dataset's accuracy falls between the two public benchmarks, consistent with its
subtly-incorrect construction sitting at intermediate difficulty. By domain: Indian History 90.0%, Finance 90.0%,
Science 100.0%, Geography 70.0% (n=10 per domain; descriptive only, insufficient for statistical claims). Zero
false positives were observed on TruthfulQA (n=77); the system exhibits a consistent conservative bias across
both public benchmarks (5/76 HaluEval, 0/77 TruthfulQA false-positive rate)."""
    )

    st.markdown("**5.2 Ablation study with significance testing**")
    st.markdown(
        """| Configuration | Accuracy | 95% CI |
|---|---|---|
| NLI only | 54.1% | 38.4–69.0% |
| NLI + RAG | 45.9% | 31.0–61.6% |
| NLI + RAG + Judges | 64.9% | 48.8–78.2% |
| Full ensemble (6) | 64.9% | 48.8–78.2% |"""
    )
    st.markdown(
        """McNemar's exact test (n=37): NLI+RAG+Judges vs. Full Ensemble showed **zero discordant pairs** (identical
predictions on every example). NLI-only vs. NLI+RAG: p=0.549. NLI-only vs. NLI+RAG+Judges: p=0.289.
**No configuration comparison reached statistical significance.** This is an important correction to a naive
reading of the raw percentages: while directional trends are consistent with the topical-similarity RAG failure
documented in Section 6, the sample size is insufficient to confirm these differences are not attributable to
chance."""
    )

    st.markdown("**5.3 Quadrant analysis**")
    st.markdown(
        """An initial quadrant evaluation, using dataset-provided answers fed identically three times to the consistency
checker, produced zero instances in either "Inconsistent" quadrant — a structural artifact of comparing identical
text to itself. After correcting the script to generate three genuinely independent LLM samples, all four
quadrants populated:"""
    )
    st.markdown(
        """| Quadrant | n | % actual hallucinations |
|---|---|---|
| Flagged + Inconsistent (system's "high-confidence" category) | 6 | 66.7% |
| Flagged + Consistent | 4 | 100.0% |
| NotFlagged + Inconsistent | 15 | 40.0% |
| NotFlagged + Consistent | 12 | 50.0% |"""
    )
    st.markdown(
        """This directly challenges the architectural assumption that flagged-plus-inconsistent claims represent
maximum-confidence hallucinations. A plausible mechanism: genuine hallucinations arising from confidently
memorized-but-ungrounded training knowledge manifest as flagged-and-stable rather than flagged-and-unstable,
since the underlying model is not guessing. Given n=4 in the strongest quadrant, this finding should be treated
as a directional hypothesis motivating scoring-scheme revision rather than a confirmed result."""
    )

    st.markdown("**5.4 Inter-judge agreement**")
    st.markdown(
        """| Pair | Agreement (n=26–38) |
|---|---|
| Groq vs. Gemini | 84.6% |
| OpenAI vs. Gemini | 73.1% |
| OpenAI vs. Groq | 71.1% |"""
    )
    st.markdown(
        """OpenAI's judge is a consistent outlier relative to Groq-Gemini agreement, corroborating the qualitative
pattern observed throughout Section 6 (OpenAI returning "UNCLEAR" where Groq and Gemini converge on
"UNSUPPORTED"). A practical implication: a two-model Groq+Gemini configuration may retain most of the
ensemble's effectiveness at reduced API cost relative to the full three-model panel."""
    )

    st.markdown("**5.5 Hand-labeled dataset construction note**")
    st.markdown(
        """An initial dataset version, using directly contradictory wrong answers, produced 100% accuracy (40/40) — a
ceiling effect indicating the task was too easy. The dataset was rewritten with subtly embedded false additions
rather than direct contradictions, yielding the 87.5% result reported above. This iteration is disclosed as part of
the methodology, since the correction itself demonstrates the importance of adversarial dataset difficulty
calibration."""
    )

    st.markdown("#### 6. Adversarial Analysis")
    st.markdown("Fifteen hand-crafted examples were evaluated through the full deployed pipeline with genuine three-sample generation.")
    st.markdown(
        """| # | Test | Target | Result |
|---|---|---|---|
| 1 | Eiffel Tower foundation/funding | RAG | False positive on fabrication |
| 2 | Everest sponsor | NLI | 0/3 flagged; judges caught genuine fabrication |
| 3 | Great Wall visibility (refusal) | Refusal classifier | 1/3 refusals correctly caught |
| 4 | Amazon fish species | Consistency | Confidently memorized fabrication evaded detection |
| 5 | Great Barrier Reef count | RAG | True-but-ungrounded claim penalized |
| 6 | Panama Canal cost/control | RAG | False positive confirmed (true-but-ungrounded) |
| 7 | Marie Curie university | RAG/NLI | Same pattern confirmed |
| 8 | Photosynthesis darkness (compound refusal) | Refusal + NLI | Inconsistent refusal detection; NLI misread inferential negation |
| 9 | Statue of Liberty height | Consistency | True-but-ungrounded, stable |
| 10 | Sahara country count | Consistency | Flagged "unstable" despite identical numeric content |
| 11 | Einstein birth/university | Claim-splitting | Correctly handled as compound unit; true-but-ungrounded |
| 12 | Antibiotics negation | NLI | Negation alone did not degrade NLI |
| 13 | Boeing 747 range | Control | Perfect result; confirms baseline correctness |
| 14 | Wright brothers flight/funding | Parser + consistency | Revealed a genuine sentence-splitting bug |
| 15 | Coffee producer | RAG | Confirmed pattern; Brazil marked "Grounded" despite absence from context |"""
    )
    st.markdown(
        """**Aggregate pattern:** across 15 targeted tests, the single most reproducible failure mode was RAG-grounding
false-positing true-but-context-absent claims as "Grounded" (8 of 15 tests), substantially more frequent than
genuine LLM fabrication (clearly observed in only 2 of 15 tests). This indicates that with a strong generator
model (GPT-4o), the system's practical weakness in deployment is less about catching outright fabrication and
more about correctly scoring true statements that are simply absent from a narrow source document."""
    )

    st.markdown("#### 7. Generalization")
    st.markdown(
        """The system was tested across Indian history, Indian institutional finance, general science, geography, and
aviation/engineering domains without domain-specific signal tuning. The consistent reproduction of the RAG
false-positive pattern across unrelated domains (space exploration, French monuments, Egyptian rivers, coffee
agriculture) indicates this is a structural property of the retrieval-grounding mechanism rather than a domain- or
topic-specific artifact."""
    )

    st.markdown("#### 8. Discussion and Limitations")
    st.markdown(
        """The equal-weight voting scheme does not reflect signals' differing reliability; Section 5.3's quadrant analysis
suggests the current confidence-ordering may be inverted relative to observed precision, motivating a weighted
or reordered scoring scheme as priority future work. Section 5.2's null significance results indicate the current
ablation sample (n=37) is underpowered; a larger-sample ablation is needed before the RAG-degradation and
judge-contribution claims can be stated as confirmed rather than directional. The TruthfulQA adaptation's use of
the correct answer as source context limits its comparability to independent-source evaluations. Two
implementation-level limitations were identified and disclosed rather than silently corrected: a sentence-splitting
parser artifact and an initial evaluation-script defect that structurally prevented the consistency signal from
registering instability, both corrected during this work. The system's dependence on three commercial/free-tier
APIs introduced operational fragility during development, including Gemini free-tier quota exhaustion and four
sequential model deprecations."""
    )

    st.markdown("#### 9. Conclusion")
    st.markdown(
        """This paper presented a six-signal ensemble hallucination detection framework evaluated across two public
benchmarks (153 instances) and an independently constructed hand-labeled dataset (40 instances), with formal
significance testing, inter-judge agreement quantification, a quadrant analysis testing the system's own
architectural assumptions, and fifteen adversarial examples. The work's principal contribution is not a single
accuracy figure but a replicated, honestly-scoped account of specific failure modes: RAG-grounding's systematic
conflation of topical similarity with factual support, self-consistency's bidirectional unreliability, a judge panel
with a quantified outlier model, and evidence that the system's own confidence-ordering assumption may not
match observed precision. Two genuine implementation bugs were discovered, disclosed, and corrected during
evaluation rather than concealed."""
    )

    st.markdown("#### Availability")
    st.markdown(
        f"""The hallucination detection system is publicly deployed at [{DEMO_URL}]({DEMO_URL}), where readers may
test the six-signal ensemble on their own context/question pairs. Source code is maintained in a private
repository; access requests may be directed to the corresponding author at khatikraj2653@gmail.com."""
    )

    st.markdown("#### References")
    st.markdown(
        """Kryscinski, W., McCann, B., Xiong, C., & Socher, R. (2020). Evaluating the factual consistency of abstractive
text summarization. *Proceedings of EMNLP.*

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.,
Rocktaschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP
tasks. *NeurIPS.*

Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., & Wen, J.-R. (2023). HaluEval: A large-scale hallucination evaluation
benchmark for large language models. *EMNLP.*

Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *ACL.*

Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-resource black-box hallucination
detection for generative large language models. *EMNLP.* arXiv:2303.08896.

Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive
summarization. *ACL.*

Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. (2021). Retrieval augmentation reduces hallucination in
conversation. *Findings of EMNLP.*

Williams, A., Nangia, N., & Bowman, S. R. (2018). A broad-coverage challenge corpus for sentence
understanding through inference. *NAACL-HLT.*

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Xing, E. P., Zhang, H.,
Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS.*"""
    )

    st.markdown("#### AI Disclosure Statement")
    st.markdown(
        """This paper was developed with the assistance of Claude (Anthropic), an AI language model, which was used
for: system architecture design and implementation guidance, statistical analysis scripting (confidence intervals,
McNemar's test, precision/recall calculations), and drafting and editing of the manuscript text. All experimental
results, data, and findings reported are derived from the author's own code execution and evaluation runs. The
author reviewed, verified, and takes full responsibility for all content, analyses, and conclusions presented in
this paper."""
    )

    st.divider()
    st.link_button("📄 Read Live on SSRN ↗", SSRN_URL, type="primary")

st.markdown(
    """
<style>
.hd-sk-wrap{margin:34px 0 8px;}
.hd-sk-label{font-family:var(--mono);font-size:12px;letter-spacing:.07em;text-transform:uppercase;color:var(--cyan);margin-bottom:8px;}
.hd-sk-title{font-family:var(--serif);font-size:1.7rem;font-weight:700;color:var(--ink);margin-bottom:6px;letter-spacing:-0.3px;}
.hd-sk-sub{font-size:0.9rem;color:var(--sub);margin-bottom:22px;}
.hd-sk-chips{display:flex;flex-wrap:wrap;gap:12px;}
.hd-sk-chip{font-family:var(--mono);font-size:13px;line-height:1.2;color:var(--sub);border:1px solid var(--border);border-radius:4px;padding:11px 20px;background:var(--card);box-shadow:0 3px 0 var(--line-2);transition:border-color .18s,color .18s,transform .12s,box-shadow .12s;}
.hd-sk-chip:hover{border-color:var(--red);color:var(--ink);transform:translateY(-2px);box-shadow:0 5px 0 var(--red-dk);}
.hd-sk-chip:active{transform:translateY(2px);box-shadow:0 1px 0 var(--red-dk);}
@media (max-width:600px){.hd-sk-chips{gap:8px;}.hd-sk-chip{font-size:11.5px;padding:8px 12px;}}
</style>
<div class="hd-sk-wrap" id="skills">
<div class="hd-sk-label">§ Skills</div>
<div class="hd-sk-title">Skills</div>
<div class="hd-sk-sub">The stack behind Hallucination Detector: agents, retrieval, evaluation and data.</div>
</div>
""",
    unsafe_allow_html=True,
)

render_stack_map(height=600)

st.markdown(
    """
<div class="hd-sk-wrap" style="margin-top:6px">
<div class="hd-sk-chips">
<span class="hd-sk-chip">LangGraph</span>
<span class="hd-sk-chip">LangChain</span>
<span class="hd-sk-chip">LangSmith Tracing</span>
<span class="hd-sk-chip">Generative AI</span>
<span class="hd-sk-chip">LLM-as-Judge</span>
<span class="hd-sk-chip">Multi-Model Ensemble</span>
<span class="hd-sk-chip">NLI · BART-MNLI</span>
<span class="hd-sk-chip">HuggingFace Transformers</span>
<span class="hd-sk-chip">RAG</span>
<span class="hd-sk-chip">FAISS</span>
<span class="hd-sk-chip">Self-Consistency</span>
<span class="hd-sk-chip">Prompt Engineering</span>
<span class="hd-sk-chip">OpenAI GPT-4o-mini</span>
<span class="hd-sk-chip">Groq</span>
<span class="hd-sk-chip">Google Gemini</span>
<span class="hd-sk-chip">Parallel Execution</span>
<span class="hd-sk-chip">HaluEval</span>
<span class="hd-sk-chip">TruthfulQA</span>
<span class="hd-sk-chip">Ablation Studies</span>
<span class="hd-sk-chip">Confidence Intervals</span>
<span class="hd-sk-chip">McNemar's Test</span>
<span class="hd-sk-chip">Inter-Judge Agreement</span>
<span class="hd-sk-chip">SQLite Logging</span>
<span class="hd-sk-chip">Python</span>
<span class="hd-sk-chip">Streamlit</span>
<span class="hd-sk-chip">Streamlit Community Cloud</span>
<span class="hd-sk-chip">Git / GitHub</span>
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div style="margin-top:40px;padding-top:18px;border-top:1.5px solid var(--ink);
display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <span style="font-size:11.5px;color:var(--dim);">Built by Raj Tejpal Khatik · MSc Applied AI · University of Warwick · 2026</span>
  <span style="font-size:11.5px;color:var(--dim);">Paper · SSRN Preprint</span>
</div>
""",
    unsafe_allow_html=True,
)

# Clear the full-screen loader now that the page has rendered.
_loading_placeholder.empty()
