"""Streamlit demo for the intent classifier + RAG agent."""

from __future__ import annotations

import requests
import streamlit as st

API_URL = st.secrets.get("api_url", "http://127.0.0.1:8000")

st.set_page_config(page_title="Customer Intent Demo", page_icon="🤖")
st.title("Customer Intent Classifier + RAG Agent")

query = st.text_area("Customer message", "My card is about to expire. What should I do?")

if st.button("Run pipeline"):
    with st.spinner("Classifying intent..."):
        intent_resp = requests.post(f"{API_URL}/predict", json={"text": query, "model": "baseline"})
    if intent_resp.status_code == 200:
        st.json(intent_resp.json())
    else:
        st.error(intent_resp.text)

    with st.spinner("Retrieving FAQ..."):
        rag_resp = requests.post(f"{API_URL}/rag", json={"text": query, "k": 3})
    if rag_resp.status_code == 200:
        st.subheader("Top FAQs")
        for item in rag_resp.json()["results"]:
            meta = item["metadata"]
            with st.expander(meta["question"]):
                st.write(meta["answer"])
                st.caption(f"Intent: {meta['intent']} | Escalate: {meta.get('escalation', False)}")
    else:
        st.error(rag_resp.text)

    with st.spinner("Agent routing..."):
        agent_resp = requests.post(f"{API_URL}/agent", json={"text": query})
    if agent_resp.status_code == 200:
        st.subheader("Agent decision")
        st.json(agent_resp.json())
    else:
        st.error(agent_resp.text)
