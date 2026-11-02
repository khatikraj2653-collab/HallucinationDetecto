import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from groq import Groq
import google.generativeai as genai

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

JUDGE_PROMPT = """You are a fact-checking judge. Given a source context and a claim, determine if the claim is supported by the context.

Context: {context}

Claim: {claim}

Respond with exactly one word: SUPPORTED, UNSUPPORTED, or UNCLEAR."""

def judge_openai(context, claim):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, claim=claim)}],
        temperature=0
    )
    return response.choices[0].message.content.strip().upper()

def judge_groq(context, claim):
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, claim=claim)}],
        temperature=0
    )
    return response.choices[0].message.content.strip().upper()

def judge_gemini(context, claim):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(JUDGE_PROMPT.format(context=context, claim=claim))
    return response.text.strip().upper()

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