# Data provenance

## Banking77

Customer service query classification dataset from PolyAI.

- Source: https://github.com/PolyAI-LDN/task-specific-datasets
- Hugging Face `datasets` name: `banking77`
- Split: `train` (10,003 examples) and `test` (3,080 examples)
- Labels: 77 fine-grained intents (e.g., `card_about_to_expire`, `direct_debit_payment_not_recognised`)
- License: Creative Commons BY 4.0
- Downloaded via: `python -m src.intent.data`

No pre-processing is applied beyond lower-casing and stripping extra whitespace; the original text is preserved.
