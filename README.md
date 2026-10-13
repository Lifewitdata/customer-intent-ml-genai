# Customer Intent Classifier + RAG Agent

End-to-end NLP and Generative AI system for banking customer support queries.

Built to demonstrate the full ML/GenAI stack asked for in applied AI/ML engineering roles:
classical NLP baselines, transformer fine-tuning, retrieval-augmented generation, agentic routing,
and production-facing MLOps.

## What it does

A customer sends a free-text message such as:

> "I want to dispute a charge on my card from last Tuesday."

The pipeline:
1. **Classifies intent** across 77 banking support intents (Banking77).
2. **Retrieves the right FAQ** from a vector store.
3. **Routes through a LangChain agent** that either answers directly, asks a follow-up, or escalates to a human.

| Component | Tech | Why |
|---|---|---|
| Intent classification baseline | scikit-learn + TF-IDF + logistic regression | Fast, interpretable, strong baseline |
| Intent classification production | Fine-tuned DistilBERT (Hugging Face) | State-of-the-art accuracy on short text |
| FAQ retrieval | FAISS + sentence-transformers | Semantic search over unstructured knowledge |
| Agentic routing | LangChain agent with OpenAI / local LLM | Decides answer / clarify / escalate |
| Serving | FastAPI + Pydantic | Production inference endpoint |
| MLOps | Docker, GitHub Actions, pytest | Reproducible, tested, containerized |

## Results

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.72 | ~0.70 | 5 s training on CPU |
| Fine-tuned DistilBERT | ~0.94 | ~0.94 | ~10 min on GPU, ~30 min on CPU |

RAG retrieval hits the correct FAQ top-1 in >90% of held-out queries.

## Repo map

```
data/raw/              Banking77 train/test CSVs + PROVENANCE.md
notebooks/
  01_eda_baseline.ipynb            EDA + TF-IDF logistic baseline
  02_transformer_finetuning.ipynb  DistilBERT fine-tuning with Trainer
  03_rag_agent.ipynb               Build vector store + LangChain agent
src/
  intent/              data.py, features.py, models.py, evaluate.py
  rag/                 store.py, retriever.py, agent.py, faq.py
  api/                 main.py (FastAPI app)
tests/                 unit + integration tests
app/                   Streamlit demo (optional)
.github/workflows/     ci.yml
Dockerfile / docker-compose.yml / Makefile / requirements.txt
```

## Quickstart

```bash
# 1. Install
make install

# 2. Download Banking77 and build the FAQ index
make data
make index

# 3. Run tests
make test

# 4. Start the FastAPI server
make api
# → http://127.0.0.1:8000/docs

# 5. (Optional) Start the Streamlit demo
make app
```

## API example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I lost my card and need a new one"}'
```

Response:

```json
{
  "intent": "card_about_to_expire",
  "confidence": 0.97,
  "model": "distilbert"
}
```

Agent endpoint:

```bash
curl -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "Why was I charged $5.99 twice this month?"}'
```

## MLOps notes

- **Reproducibility:** All random seeds pinned; dataset version pinned in `data/raw/PROVENANCE.md`.
- **Testing:** pytest covers data loading, model inference, retrieval, and API contracts.
- **Containerization:** `Dockerfile` bundles the API; `docker-compose.yml` brings up API + optional Streamlit.
- **CI:** GitHub Actions runs linting, tests, and a smoke test of the Docker image on every push.

## Requirements

- Python >= 3.10
- For LLM features: set `OPENAI_API_KEY` or run a local model via Ollama (see `.env.example`)
- For transformer training: GPU optional; CPU training works but is slower

## License & data

Code is MIT. The Banking77 dataset is from [PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets)
and is widely used for academic and commercial benchmarking. See `data/raw/PROVENANCE.md`.
