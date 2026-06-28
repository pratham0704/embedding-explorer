# F2 — Embedding Explorer

A diagnostic tool that makes embedding model behavior visible and debuggable
on 10,000 realistic fintech support tickets.

## The core insight

High cosine similarity scores can be wrong. This tool shows exactly where
and why — with measured evidence.

## What it proves

### Identifier Confusion

Different case numbers score 0.99 similarity across all three models tested.
Dense retrieval cannot distinguish CASE-2024-000123 from CASE-2024-000124.
BM25 exact match is required for all identifier lookups.

### Negation Blindness

"The transaction is suspicious" vs "not suspicious":
all-MiniLM-L6-v2: 0.9229 ← fails
all-mpnet-base-v2: 0.8298 ← fails  
 paraphrase-MiniLM-L6-v2: 0.3542 ← handles correctly

Model choice matters for negation-sensitive domains like fraud detection.

### Domain Vocabulary Gap

"NACH mandate rejected" vs "auto debit instruction cancelled":
All models score below 0.22.
General embedding models do not understand Indian banking terminology.
Domain fine-tuning is required for production RAG on Indian financial data.

### Semantic vs Lexical Confusion

Same words, different meaning scores higher than different words, same meaning.
Cross-encoder reranking required to fix ranking quality.

## Architecture

Dataset → Embedder → Similarity → FailureDetector → Reporter
→ ModelComparison

## Results

Dataset: 10,000 fintech support tickets across 10 categories
Models compared: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2
Pairs analyzed: 124,750
Failure rate: 0.52%

Failures by mode:
IDENTIFIER_CONFUSION: 60.58%
DOMAIN_VOCABULARY_GAP: 31.75%
NEGATION_BLINDNESS: 7.67%

## Getting Started

### 1. Install dependencies

pip install -r requirements.txt

### 2. Generate dataset

python3 scripts/generate_dataset.py

### 3. Generate embeddings

python3 scripts/run_explorer.py
