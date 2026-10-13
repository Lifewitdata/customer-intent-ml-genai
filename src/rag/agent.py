"""Agentic routing: answer, clarify, or escalate based on intent + retrieved FAQ."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent

load_dotenv()
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from src.intent.models import predict_baseline
from src.rag.retriever import top_faq


REACT_PROMPT = PromptTemplate.from_template(
    """You are a helpful bank support assistant. Your job is to either answer the customer's question,
ask a single clarifying question, or escalate to a human agent.

Use the following format:

Question: the input question you must answer
Thought: think about what the customer is asking
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question, or "ESCALATE" if a human is needed

Available tools:
{tools}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)


def _classify_text(text: str) -> str:
    return predict_baseline([text])[0]["intent"]


def _retrieve_faq(query: str) -> str:
    faq = top_faq(query)
    if not faq:
        return "No relevant FAQ found."
    meta = faq["metadata"]
    esc = " [ESCALATE]" if meta.get("escalation") else ""
    return f"Intent: {meta['intent']}{esc}\nQ: {meta['question']}\nA: {meta['answer']}"


def make_tools() -> list[Tool]:
    return [
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


def make_agent(model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
    """Build a ReAct agent. Falls back to rule-based routing if no OpenAI key is set."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    tools = make_tools()
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


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
    executor = make_agent()
    if executor is None:
        return route(text)
    result = executor.invoke({"input": text})
    output = result.get("output", "")
    if "ESCALATE" in output.upper():
        return {"action": "escalate", "response": output.replace("ESCALATE", "").strip() or "I'm connecting you with a specialist.", "source": "agent"}
    return {"action": "answer", "response": output, "source": "agent"}
