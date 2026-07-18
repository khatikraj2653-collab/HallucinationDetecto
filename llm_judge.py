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
        model="gpt-4o",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, claim=claim)}],
        temperature=0
    )
    return response.choices[0].message.content.strip().upper()

def judge_groq(context, claim):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, claim=claim)}],
        temperature=0
    )
    return response.choices[0].message.content.strip().upper()

def judge_gemini(context, claim):
    model = genai.GenerativeModel("models/gemini-3.1-flash-lite")
    response = model.generate_content(JUDGE_PROMPT.format(context=context, claim=claim))
    return response.text.strip().upper()

def ask_openai(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def ask_groq(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def ask_gemini(prompt):
    model = genai.GenerativeModel("models/gemini-3.1-flash-lite")
    response = model.generate_content(prompt)
    return response.text.strip()

def multi_consistency_check(a1, a2, a3):
    from consistency_utils import check_consistency
    results = {}
    try:
        results["OpenAI"] = check_consistency(ask_openai, a1, a2, a3)
    except Exception as e:
        results["OpenAI"] = f"ERROR: {e}"
    try:
        results["Groq"] = check_consistency(ask_groq, a1, a2, a3)
    except Exception as e:
        results["Groq"] = f"ERROR: {e}"
    try:
        results["Gemini"] = check_consistency(ask_gemini, a1, a2, a3)
    except Exception as e:
        results["Gemini"] = f"ERROR: {e}"
    return results

def multi_llm_judge(context, claim):
    results = {}
    try:
        results["OpenAI"] = judge_openai(context, claim)
    except Exception as e:
        results["OpenAI"] = f"ERROR: {e}"
    try:
        results["Groq"] = judge_groq(context, claim)
    except Exception as e:
        results["Groq"] = f"ERROR: {e}"
    try:
        results["Gemini"] = judge_gemini(context, claim)
    except Exception as e:
        results["Gemini"] = f"ERROR: {e}"
    return results