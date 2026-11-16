---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 fa73664e11613252610a6f06da0acca6a580a54a9aa4554632dba899183ff5d2
  data_files:
  - split: train
    path: sft_alpaca.jsonl
---

# Dataset Card: hinglish_fresh_curation

## Summary

- **Run timestamp**: 2026-11-16T03:50:29.336998Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `4b6cb35889e9795f`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 80 | 80 | — | 0 | 80 |
| SchemaGate | 80 | 80 | 0 | 0 | 80 |
| ExactDeduplicator | 80 | 80 | — | — | 80 |
| TextCleaner | 80 | 80 | — | — | 80 |
| AlpacaExporter | — | 80 | — | — | 80 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 80/0 passed (N/A), 0 rejected
- **SchemaGate**: 80/80 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 0

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.21
- **python**: 3.12.3

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_alpaca` | train (80) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
