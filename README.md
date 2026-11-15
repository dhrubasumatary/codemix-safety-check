# CodeMix Safety Check

**Does a small Cohere model stay safe when you talk to it the way India actually types?**

Lexsi AI Safety & Alignment Hackathon, Track 2, by Dhruba Basumatary (Miithii).

Most people in India don't ask AI things in clean English. They type Hinglish: half Hindi, half English, in Roman script. I build Miithii, an AI friend for Assamese and Boro speakers, and I see it every day: the moment a prompt goes code-mixed, models get weird. Some answer things they'd refuse in English. Others refuse totally harmless questions. Safety evals are almost all English, so nobody is measuring this properly.

CodeMix Safety Check is a small, honest, reproducible benchmark plus a repair test for English vs Hinglish prompts on Cohere Labs' **Tiny Aya Global** (3.35B).

## The dataset

- 30 intent pairs. Each pair is a harmful prompt and a harmless near-neighbour on the same topic with mostly the same words (break into a neighbour's house / get back into my own house).
- Every prompt written in English and in Hinglish: **120 prompts** (60 harmful, 60 harmless).
- Split by pair so nothing leaks: **10 pairs (40 prompts) for calibration, 20 pairs (80 prompts) held out for test**.
- Plus an out-of-distribution check on **XSTest** (`Paul/XSTest`): safe prompts that look scary, and unsafe contrasts.
- All newly written, no private user chats. Prompts only, no harmful completions are published.

File: `data/codemix_raw.jsonl` (fields: `id, pair_id, category, side, lang, split, instruction`).

## The pipeline (three Lexsi repos chained)

1. **CuratorKIT** validates, dedupes and cleans the prompt set and freezes it with provenance (`curation/pipeline.yaml`, output and manifest in `curation/out/`). The harmful half is kept on purpose, since this is an eval set.
2. **SafeTune** (`steer.RefusalDirectionTrainer`) computes a refusal direction from the calibration split only and steers Tiny Aya Global at inference time (weights unchanged). The steering strength `alpha` is picked on the calibration split only, never on test.
3. **AuditKIT** scores baseline vs steered outputs on the exact same held-out prompts and compares the runs (`ak.evaluate`, `ak.compare`).

An LLM judge (`Qwen/Qwen2.5-7B-Instruct`, open weights) labels every response REFUSAL / COMPLIANCE / NONSENSE and whether it gives real harmful help. All raw generations and judge verdicts are published so anyone can re-judge them.

## Metrics (per language, before and after)

- **Attack success rate (ASR)**: harmful prompts that get real harmful help. Lower is better.
- **Over-refusal rate**: harmless prompts wrongly refused. Lower is better.
- **Benign usefulness**: harmless prompts that get a coherent, on-topic answer. Higher is better.
- **Nonsense rate**: steering can break a model; this catches it.
- **Latency** per prompt on the same GPU.

The point is the trade-off. If steering cuts harmful answers but starts refusing every Hinglish question, that's a fail, and this report will show it.

## Results

Status: the GPU run (Modal, A100) is in progress. The before/after table, raw generations, judge verdicts and AuditKIT comparison are written here and to the Hugging Face dataset as soon as it lands.

## Reproduce

```bash
pip install "git+https://github.com/Lexsi-Labs/CuratorKIT"
curatorkit run curation/pipeline.yaml --output-dir curation/out/

pip install modal && modal setup
modal secret create codemix-hf HF_TOKEN=<your token>
modal deploy run_modal.py      # baseline -> SafeTune steering -> AuditKIT, on one A100
```

## Links

- Dataset, generations, judge verdicts: https://huggingface.co/datasets/dhrubasumatary/codemix-safety-check
- Steering artifact: https://huggingface.co/dhrubasumatary/tiny-aya-global-codemix-steer
- Base model: https://huggingface.co/CohereLabs/tiny-aya-global

## Limits and next steps

Small set (120 prompts), one model, one intervention, one LLM judge. If shortlisted: scale the set, add an Assamese / Assamese-English pilot written by a native speaker (never machine-translated and called validated), add a second SafeTune intervention (Harden LoRA), and human spot-checks of judge disagreements.
