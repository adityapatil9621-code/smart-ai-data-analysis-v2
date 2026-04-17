"""
chat_engine.py

Chat Engine for Smart AI Data Intelligence System.

Improvements:
- Timeout configurable via env / config
- Strips <think>…</think> blocks robustly
- generate_dynamic_questions returns deduplicated list
- build_context truncates large driver lists
- Graceful fallback when Ollama is unreachable
"""

import ast
import os
from typing import Dict, List

import requests


OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://192.168.1.102:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
TIMEOUT      = int(os.getenv("OLLAMA_TIMEOUT", "60"))


# ============================================================
# Low-level LLM call
# ============================================================

def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "top_p": 0.9},
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        output = resp.json().get("response", "")

        # Strip chain-of-thought blocks
        if "</think>" in output:
            output = output.split("</think>")[-1].strip()

        return output.strip()

    except requests.exceptions.ConnectionError:
        return "⚠️ Could not reach the AI model. Please ensure Ollama is running."
    except requests.exceptions.Timeout:
        return "⚠️ AI model timed out. Try again or reduce the complexity of your question."
    except Exception as e:
        return f"⚠️ Error: {e}"


# ============================================================
# Context Builder
# ============================================================

def build_context(memory: Dict) -> str:
    insight   = memory.get("insight_intelligence", {})
    drivers   = insight.get("top_positive_drivers", [])[:3]
    forecast  = memory.get("forecast_intelligence")
    meta      = memory.get("metadata", {})
    model_info = memory.get("model_intelligence", {})

    driver_text   = ", ".join(d["feature"] for d in drivers) if drivers else "N/A"
    forecast_text = (
        f"Trend={forecast.get('trend_direction', 'N/A')}, "
        f"Confidence={forecast.get('forecast_confidence', 'N/A')}"
        if forecast else "Not available (no time-series detected)"
    )

    return (
        f"Dataset: {meta.get('rows', '?')} rows × {meta.get('columns', '?')} columns | "
        f"Quality: {meta.get('quality_score', '?')} | Domain: {meta.get('domain', 'general')}\n"
        f"Model: {model_info.get('selected_model', 'N/A')} | "
        f"Confidence: {model_info.get('confidence', 'N/A')}\n"
        f"Top Drivers: {driver_text}\n"
        f"Risk Score: {insight.get('risk_score', 'N/A')}\n"
        f"Forecast: {forecast_text}"
    )


# ============================================================
# Dynamic Question Generator
# ============================================================

def generate_dynamic_questions(memory: Dict) -> List[str]:
    context = build_context(memory)
    prompt  = f"""You are a business analyst.

Context:
{context}

Generate 6 short, business-friendly questions a user might ask about this dataset.

Rules:
- Plain English, no jargon
- Each question on its own line
- Return ONLY a valid Python list of strings, nothing else

Example:
["What are the key drivers?", "Is the trend going up?"]
"""
    output = ollama_generate(prompt)
    try:
        questions = ast.literal_eval(output)
        if isinstance(questions, list):
            # Deduplicate while preserving order
            seen, unique = set(), []
            for q in questions:
                if q not in seen:
                    seen.add(q)
                    unique.append(q)
            return unique
    except Exception:
        pass

    # Fallback questions
    return [
        "What are the key drivers of this dataset?",
        "Is the model reliable?",
        "What risks exist in the data?",
        "What is the current trend?",
        "Which features should I focus on?",
        "Are there any anomalies I should worry about?",
    ]


# ============================================================
# Chat Engine
# ============================================================

class ChatEngine:

    def respond(self, user_query: str, memory: Dict) -> str:
        context = build_context(memory)
        prompt  = f"""You are a professional business data analyst.

Context:
{context}

User Question:
{user_query}

Instructions:
- Answer clearly and concisely in plain business language
- Do not expose internal model details or technical jargon unless asked
- Keep the answer to 2–4 sentences
"""
        return ollama_generate(prompt)