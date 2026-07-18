import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv
import re
import time

from db_utils import init_db, log_claim
from rag_utils import build_vectorstore, check_claim_grounding
from refusal_utils import is_refusal
from llm_judge import multi_llm_judge, multi_consistency_check
from consistency_utils import generate_multiple_answers
from scoring_utils import compute_final_verdict

load_dotenv()
init_db()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_nli_model():
    return pipeline("text-classification", model="facebook/bart-large-mnli", device=-1, model_kwargs={"low_cpu_mem_usage": False})

nli_model = load_nli_model()

st.set_page_config(page_title="Hallucination Detector", layout="wide")
st.title("LLM Hallucination Detector")

context = st.text_area("Context", height=200)
question = st.text_input("Question")

def split_claims(answer):
    sentences = re.split(r'(?<!\d)\.(?!\d)', answer)
    return [s.strip() for s in sentences if s.strip()]

def check_claim(context, claim):
    result = nli_model(f"{context}</s></s>{claim}")[0]
    return result["label"], round(result["score"], 3)

if st.button("Analyze"):
    if not context or not question:
        st.warning("Please fill in both Context and Question.")
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
        cols = st.columns(3)

        for i, (ans, col) in enumerate(zip(answers, cols), 1):
            with col:
                st.markdown(f"**├── Answer {i}**")
                score_ph = st.empty()
                score_ph.badge("analyzing...", color="gray")
                st.caption(ans)

                claims = split_claims(ans)
                bad_votes = 0
                total_signals = 0

                live_score_ph = st.empty()

                for claim in claims:
                    if is_refusal(claim):
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;└── _{claim[:40]}..._")
                        st.badge("refusal — skipped", color="gray")
                        continue

                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;└── _{claim[:40]}..._" if len(claim) > 40 else f"&nbsp;&nbsp;&nbsp;&nbsp;└── _{claim}_")

                    nli_ph = st.empty()
                    nli_ph.badge("NLI checking...", color="gray")
                    time.sleep(0.4)
                    nli_label, nli_score = check_claim(context, claim)
                    nli_flagged = nli_label.upper() == "CONTRADICTION"
                    with nli_ph.popover(f"NLI: {nli_label}", use_container_width=False):
                        st.write(f"Label: **{nli_label}**")
                        st.write(f"Confidence: {nli_score}")
                    total_signals += 1
                    if nli_flagged:
                        bad_votes += 1

                    running_pct = round((bad_votes / total_signals) * 100, 1) if total_signals > 0 else 0.0
                    live_score_ph.caption(f"⏳ Running: {bad_votes}/{total_signals} flagged = {running_pct}%")

                    rag_ph = st.empty()
                    rag_ph.badge("RAG checking...", color="gray")
                    time.sleep(0.4)
                    is_grounded, rag_score, best_chunk = check_claim_grounding(vectorstore, claim)
                    with rag_ph.popover(f"RAG: {'Grounded' if is_grounded else 'Not grounded'}"):
                        st.write(f"Score: {rag_score}")
                        st.write(f"Closest chunk: _{best_chunk}_")
                    total_signals += 1
                    if not is_grounded:
                        bad_votes += 1

                    running_pct = round((bad_votes / total_signals) * 100, 1) if total_signals > 0 else 0.0
                    live_score_ph.caption(f"⏳ Running: {bad_votes}/{total_signals} flagged = {running_pct}%")

                    judges_ph = st.empty()
                    judges_ph.badge("Judges checking...", color="gray")
                    time.sleep(0.4)
                    judge_results = multi_llm_judge(context, claim)
                    unsupported = sum(1 for v in judge_results.values() if "ERROR" not in v.upper() and "UNSUPPORTED" in v.upper())
                    valid_judges = sum(1 for v in judge_results.values() if "ERROR" not in v.upper())
                    with judges_ph.popover(f"Judges: {unsupported}/{valid_judges} flagged"):
                        for jn, jv in judge_results.items():
                            st.write(f"**{jn}:** {jv}")
                    total_signals += valid_judges
                    bad_votes += unsupported

                    log_claim(context, question, ans, claim, nli_label, nli_score)

                    running_pct = round((bad_votes / total_signals) * 100, 1) if total_signals > 0 else 0.0
                    live_score_ph.caption(f"⏳ Running: {bad_votes}/{total_signals} flagged = {running_pct}%")

                total_signals += 1
                if not is_consistent:
                    bad_votes += 1
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;└── Consistency: {'✅ stable' if is_consistent else '🚩 unstable'}")

                answer_score = round((bad_votes / total_signals) * 100, 1) if total_signals > 0 else 0.0
                color = "red" if answer_score >= 67 else "orange" if answer_score >= 34 else "green"
                score_ph.badge(f"{answer_score}% flagged", color=color)
                live_score_ph.caption(f"✅ Final: {bad_votes}/{total_signals} flagged = {answer_score}%")

        st.divider()
        st.caption("Click any signal badge above to expand its detail.")