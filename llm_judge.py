import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from groq import Groq
import google.generativeai as genai

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

JUDGE_PROMPT = """You are a fact-checking judge. Decide whether the CLAIM is logically supported by the CONTEXT.

Context: {context}

Claim: {claim}

Scoring rules:
- SUPPORTED: The claim's core assertion is directly stated OR clearly implied by the context. Ignore trivial differences in capitalization, punctuation, spacing, word order, articles ("the", "a"), minor paraphrasing, or additional non-contradictory elaboration. If the context literally states the same fact in different wording, that is SUPPORTED.
- UNSUPPORTED: The claim directly contradicts the context, OR asserts a new factual detail that cannot be derived or paraphrased from the context.
- UNCLEAR: The context genuinely does not contain enough information to determine support or contradiction.

Bias toward SUPPORTED when the claim is essentially a rephrasing of the context. Reserve UNSUPPORTED for real contradictions or clearly unsupported new facts.

Respond with exactly one word: SUPPORTED, UNSUPPORTED, or UNCLEAR."""

def _parse_verdict(raw: str) -> str:
    """Pick the first SUPPORTED/UNSUPPORTED/UNCLEAR token from a judge
    response. Handles trailing punctuation, brief reasoning, and quotes."""
    txt = (raw or "").strip().upper()
    for word in txt.replace(".", " ").replace(",", " ").replace('"', " ").split():
        if word in ("SUPPORTED", "UNSUPPORTED", "UNCLEAR"):
            return word
    return txt or "UNCLEAR"


def judge_openai(context, claim):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, claim=claim)}],
        temperature=0
    )
    return _parse_verdict(response.choices[0].message.content)

def judge_groq(context, claim):
    # Upgraded from gpt-oss-20b (noisy, over-flagged simple paraphrases) to
    # Llama 3.3 70B which follows the rubric far more reliably.
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, claim=claim)}],
        temperature=0
    )
    return _parse_verdict(response.choices[0].message.content)

def judge_gemini(context, claim):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(JUDGE_PROMPT.format(context=context, claim=claim))
    return _parse_verdict(response.text)

def ask_openai(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def ask_groq(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def ask_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

JUDGE_FNS = {"OpenAI": judge_openai, "Groq": judge_groq, "Gemini": judge_gemini}
ASK_FNS = {"OpenAI": ask_openai, "Groq": ask_groq, "Gemini": ask_gemini}

# slot -> providers to try, in order (primary first, then backups)
FALLBACK_CHAINS = {
    "OpenAI": ["OpenAI", "Groq", "Gemini"],
    "Groq": ["Groq", "OpenAI", "Gemini"],
    "Gemini": ["Gemini", "OpenAI", "Groq"],
}


def _run_with_fallback(slot, call):
    """Try each provider in the slot's chain until one succeeds.
    Returns (result, provider_used). If all fail, result is an ERROR string
    and provider_used is None."""
    errors = []
    for provider in FALLBACK_CHAINS[slot]:
        try:
            return call(provider), provider
        except Exception as e:
            errors.append(f"{provider}: {e}")
    return "ERROR: all providers failed (" + "; ".join(errors) + ")", None


def multi_consistency_check_detailed(a1, a2, a3):
    from consistency_utils import check_consistency
    results, used = {}, {}
    for slot in FALLBACK_CHAINS:
        results[slot], used[slot] = _run_with_fallback(
            slot, lambda p: check_consistency(ASK_FNS[p], a1, a2, a3)
        )
    return results, used


def multi_consistency_check(a1, a2, a3):
    return multi_consistency_check_detailed(a1, a2, a3)[0]


def multi_llm_judge_detailed(context, claim):
    """Returns (verdicts, used). verdicts keeps the original slot keys
    (OpenAI/Groq/Gemini) with plain verdict strings; used maps each slot to
    the provider that actually answered (differs from the slot on fallback)."""
    results, used = {}, {}
    for slot in FALLBACK_CHAINS:
        results[slot], used[slot] = _run_with_fallback(
            slot, lambda p: JUDGE_FNS[p](context, claim)
        )
    return results, used


def multi_llm_judge(context, claim):
    return multi_llm_judge_detailed(context, claim)[0]