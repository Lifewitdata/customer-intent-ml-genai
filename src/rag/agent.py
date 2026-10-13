"""Agentic routing: answer, clarify, or escalate based on intent + retrieved FAQ.

The FastAPI app must import cleanly even when the optional LLM stack (LangChain +
OpenAI-compatible key) is missing or incompatible, so all LangChain imports are lazy
and `agent_answer` degrades to rule-based `route` on any failure.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from src.intent.models import predict_baseline
from src.rag.retriever import top_faq


SYSTEM_PROMPT = """You are a helpful bank support assistant. Your job is to either answer the
customer's question, ask a single clarifying question, or escalate to a human agent.

Workflow:
1. Use classify_intent to label the customer's request.
2. Use retrieve_faq to fetch the most relevant FAQ answer for that intent.
3. If the FAQ is relevant and no escalation is flagged, give a concise Final Answer based on it.
   If the FAQ is missing, ambiguous, or flagged [ESCALATE], respond with a single clarifying
   question or start your answer with "ESCALATE".
"""


def _classify_text(text: str) -> str:
    return predict_baseline([text])[0]["intent"]


def _retrieve_faq(query: str) -> str:
    faq = top_faq(query)
    if not faq:
        return "No relevant FAQ found."
    meta = faq["metadata"]
    esc = " [ESCALATE]" if meta.get("escalation") else ""
    return f"Intent: {meta['intent']}{esc}\nQ: {meta['question']}\nA: {meta['answer']}"


def make_agent(temperature: float = 0.0):
    """Build a LangChain ReAct-style agent. Returns None if the LLM stack is unavailable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.strip() in ("", "your-key-here"):
        return None
    try:
        from langchain.agents import create_agent
        from langchain_core.tools import Tool
        from langchain_openai import ChatOpenAI

        tools = [
            Tool(
                name="classify_intent",
                func=_classify_text,
                description="Classify the customer's intent into one of 77 banking support categories.",
            ),
            Tool(
                name="retrieve_faq",
                func=_retrieve_faq,
                description="Retrieve the most relevant FAQ answer from the knowledge base.",
            ),
        ]
        model_name = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        kwargs = {"model_name": model_name, "temperature": temperature, "api_key": api_key}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        llm = ChatOpenAI(**kwargs)
        return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    except Exception:
        return None


def route(text: str) -> dict:
    """Rule-based fallback that mimics the agent's decision logic without an LLM."""
    intent = _classify_text(text)
    faq = top_faq(text)
    if not faq:
        return {"action": "clarify", "response": "Could you rephrase that? I want to make sure I route you correctly.", "intent": intent}
    meta = faq["metadata"]
    if meta.get("escalation"):
        return {"action": "escalate", "response": "I'm connecting you with a specialist.", "intent": intent, "faq": meta}
    return {"action": "answer", "response": meta["answer"], "intent": intent, "faq": meta}


def agent_answer(text: str) -> dict:
    """Use the LangChain agent if available, otherwise use rule-based routing."""
    agent = make_agent()
    if agent is None:
        return route(text)
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": text}]})
        messages = result.get("messages", [])
        output = messages[-1].content if messages else ""
    except Exception:
        return route(text)
    if not output:
        return route(text)
    if "ESCALATE" in output.upper():
        cleaned = output.upper().replace("ESCALATE", "").strip() or "I'm connecting you with a specialist."
        return {"action": "escalate", "response": cleaned, "source": "agent"}
    return {"action": "answer", "response": output, "source": "agent"}
