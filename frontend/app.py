import os

os.environ["HF_HOME"] = "D:\\huggingface_cache"
os.environ["TEMP"] = "D:\\temp"
os.environ["TMP"] = "D:\\temp"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
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

def analyze_answer(context, question, answer, is_consistent_overall, vectorstore):
    claims = split_claims(answer)
    claim_results = []
    hallucinated_count = 0
    scoreable_count = 0

    for claim in claims:
        if is_refusal(claim):
            claim_results.append({
                "claim": claim,
                "type": "refusal",
                "verdict": "Refusal — not a hallucination",
                "reason": "Model honestly admitted uncertainty"
            })
            log_claim(context, question, answer, claim, "REFUSAL", 1.0)
            continue

        nli_label, nli_score = check_claim(context, claim)
        is_grounded, rag_score, best_chunk = check_claim_grounding(vectorstore, claim)
        judge_results = multi_llm_judge(context, claim)

        final_verdict, final_reason = compute_final_verdict(nli_label, is_grounded, judge_results, is_consistent_overall)

        scoreable_count += 1
        if "Hallucination" in final_verdict:
            hallucinated_count += 1

        claim_results.append({
            "claim": claim,
            "type": "checked",
            "nli_label": nli_label,
            "nli_score": nli_score,
            "is_grounded": is_grounded,
            "rag_score": rag_score,
            "best_chunk": best_chunk,
            "judge_results": judge_results,
            "verdict": final_verdict,
            "reason": final_reason
        })

        log_claim(context, question, answer, claim, nli_label, nli_score)

    hallucination_score = round((hallucinated_count / scoreable_count) * 100, 1) if scoreable_count > 0 else 0.0
    return claim_results, hallucination_score

def render_claim_card(r):
    if r["type"] == "refusal":
        st.markdown(f"**{r['claim']}**")
        st.caption(r["verdict"])
        st.divider()
        return

    if "High-Confidence" in r["verdict"]:
        badge_color = "red"
    elif "Possible" in r["verdict"]:
        badge_color = "orange"
    else:
        badge_color = "green"

    st.markdown(f"**{r['claim']}**")
    st.markdown(f":{badge_color}[{r['verdict']}]")
    st.caption(r["reason"])

    nli_icon = "🚩" if r["nli_label"].upper() == "CONTRADICTION" else "✅" if r["nli_label"].upper() == "ENTAILMENT" else "⚪"
    rag_icon = "✅" if r["is_grounded"] else "🚩"

    st.caption(f"{nli_icon} NLI: {r['nli_label']} ({r['nli_score']})  •  {rag_icon} RAG: {'grounded' if r['is_grounded'] else 'not grounded'} ({r['rag_score']})")

    judge_line = "  •  ".join(
        f"{'🚩' if 'UNSUPPORTED' in v.upper() else '✅' if 'SUPPORTED' in v.upper() else '⚪'} {name}"
        for name, v in r["judge_results"].items()
    )
    st.caption(judge_line)
    st.divider()

if st.button("Analyze"):
    if not context or not question:
        st.warning("Please fill in both Context and Question.")
    else:
        st.markdown(f"#### Question: {question}")
        st.divider()

        with st.spinner("Generating 3 answers..."):
            answers = generate_multiple_answers(context, question, n=3)

        with st.spinner("Checking cross-sample consistency..."):
            consistency_results = multi_consistency_check(answers[0], answers[1], answers[2])

        consistent_votes = 0
        total_votes = 0
        for judge_name, verdict in consistency_results.items():
            if "ERROR" not in verdict.upper():
                total_votes += 1
                if "CONSISTENT" in verdict.upper() and "INCONSISTENT" not in verdict.upper():
                    consistent_votes += 1

        is_consistent_overall = (consistent_votes >= (total_votes / 2)) if total_votes > 0 else True

        with st.expander("Cross-sample consistency check"):
            for judge_name, verdict in consistency_results.items():
                flag = "✅" if "CONSISTENT" in verdict.upper() and "INCONSISTENT" not in verdict.upper() else "🚩"
                st.write(f"{flag} **{judge_name}:** {verdict}")

        st.divider()

        with st.spinner("Building context index..."):
            vectorstore = build_vectorstore(context)

        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        headers = [col.empty() for col in columns]
        bodies = [col.container() for col in columns]

        for i, header in enumerate(headers, 1):
            header.markdown(f"**Answer {i}** — analyzing...")

        all_results = []
        for i, (ans, header, body) in enumerate(zip(answers, headers, bodies), 1):
            with body:
                st.write(ans)
            claim_results, hallucination_score = analyze_answer(context, question, ans, is_consistent_overall, vectorstore)
            all_results.append({
                "index": i,
                "answer": ans,
                "claim_results": claim_results,
                "hallucination_score": hallucination_score
            })
            header.markdown(f"**Answer {i}** — {hallucination_score}% hallucinated")

        ranked = sorted(all_results, key=lambda x: x["hallucination_score"])
        best = ranked[0]

        st.divider()
        st.markdown(f"### Best answer: Answer {best['index']} ({best['hallucination_score']}% hallucinated)")
        st.write(best["answer"])
        for r in best["claim_results"]:
            render_claim_card(r)

        st.markdown("### Full breakdown")
        tabs = st.tabs([f"Answer {r['index']} ({r['hallucination_score']}%)" for r in all_results])
        for tab, r in zip(tabs, all_results):
            with tab:
                st.write(r["answer"])
                for cr in r["claim_results"]:
                    render_claim_card(cr)