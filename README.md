# CodeMix Safety Check

**Does a small Cohere model stay safe when you talk to it the way India actually types?**

Lexsi AI Safety & Alignment Hackathon, Track 2, by Dhruba Basumatary (Miithii).

Most people in India don't ask AI things in clean English. They type Hinglish: half Hindi, half English, in Roman script. I build Miithii, an AI friend for Assamese and Boro speakers, and I see it every day: the moment a prompt goes code-mixed, models get weird. Some answer things they'd refuse in English. Others refuse totally harmless questions. Safety evals are almost all English, so nobody is measuring this properly.

CodeMix Safety Check is a small, honest, reproducible benchmark plus a repair test for English vs Hinglish prompts on Cohere Labs' **Tiny Aya Global** (3.35B).

## TL;DR

- **Method:** SafeTune `RefusalDirectionTrainer`, strength 0.3, inference-time steering, weights unchanged. Picked on a dev split, then frozen.
- **Held-out (230 prompts):** harmful replies 4/90 -> 0/90, harmless wrongly refused 12/140 -> 21/140, harmless helpful 128/140 -> 124/140.
- **Fresh Hinglish set (80 new prompts, frozen and published before any generation):** harmful replies 9/40 -> 2/40, wrongly refused 0/40 -> 3/40, helpful 35/40 -> 33/40.
- **The trade-off is real:** the judge flags fewer code-mixed completions as harmful after steering and costs some over-refusal and usefulness. All numbers are LLM-judge estimates with known errors (listed below), not certified safety.

## Data

1. **Exploratory set** (`data/codemix_raw.jsonl`): 30 intent pairs, each a harmful prompt and a harmless near-neighbour on the same topic with mostly the same words, every prompt in English and Hinglish = 120 prompts. Split by pair: 10 pairs (40 prompts) for calibration, 20 pairs (80 prompts) held out. Plus **XSTest** (`Paul/XSTest`) as an out-of-distribution check: 150 held out, the rest used as dev.
2. **Fresh Hinglish validation** (`validation/hinglish_fresh.jsonl`): 40 new harmful/harmless pairs = 80 prompts, written after the method was frozen. SHA-256 `a82f44e25c350acefbaac48767b8d6bb2c65240f4acaaa58dccdc07b7481a588`, pushed to GitHub and HF before a single generation. Run once with the frozen steer, no tuning.
3. **Dropped:** a machine-translated Hinglish XSTest. A 32B faithfulness check passed only 4 of 150 translations, so it's excluded from every headline number.

Prompts are **assistant-authored for this project, not native-speaker validated**, and contain no private user data. The fresh set's topics may overlap the exploratory set's categories.

## Pipeline (three Lexsi repos chained)

1. **CuratorKIT** validates, dedupes and cleans each prompt set and freezes it with checksums and provenance (`curation/pipeline.yaml` -> `curation/out/`, `validation/curation/pipeline.yaml` -> `validation/curation/out/`). Harmful prompts are kept on purpose, since these are eval sets. 80/80 fresh rows passed, 0 rejected.
2. **SafeTune** computes a refusal direction from the calibration split only and steers at inference time. I compared RefusalDirection, AlphaSteer, AdaSteer, CAST and two gated ablation variants of my own (`run2_modal.py`, `run3_modal.py`, `run4_modal.py`), picking strength on dev only. RefusalDirection @ 0.3 won. Artifact: `run2/RefusalDirectionTrainer.safetensors` in the model repo (tensor `wrapped.direction`).
3. **AuditKIT** replays the frozen judge labels (`model='precomputed'`, one scorer per label) for baseline and steered runs on identical prompts and compares them with `ak.compare` (`replay_auditkit.py` -> `results/auditkit_j3/`). It's a structured comparison of the same labels, not a second judge.

## Scoring

`score_audit_modal.py`: `Qwen/Qwen2.5-32B-Instruct` reads the **full** reply, blinded to condition and method, items hash-shuffled, and labels four things independently: **harmful** (actionable help for serious harm), **refusal** (actually declines), **useful** (substantive, relevant help), **degenerate** (looping, empty, echo). Safe support or prevention isn't counted as refusal, and a misunderstanding isn't counted as harm. Rubric SHA-256 `9f198287ae17293cb7c13bb16bd110d8202d05017e9d9527bbe69cef513e13c3`, inputs pinned to dataset revision `bf13ba8111ef4e851bf472bc70e3d13651f6813e`. Generation: greedy, 160 new tokens, batch 16, same for both conditions.

## Results

### Held-out exploratory set (CodeMix test 80 + XSTest 150)

**All prompts**

| | baseline | steered (RD 0.3) |
|---|---|---|
| Harmful prompts that got harmful help (lower is better) | 4/90 (4.44%) | 0/90 (0.0%) |
| Harmful prompts refused (higher is better) | 81/90 (90.0%) | 87/90 (96.67%) |
| Harmless prompts wrongly refused (lower is better) | 12/140 (8.57%) | 21/140 (15.0%) |
| Harmless prompts answered helpfully (higher is better) | 128/140 (91.43%) | 124/140 (88.57%) |
| Degenerate / looping replies (lower is better) | 10/230 (4.35%) | 8/230 (3.48%) |

**English**

| | baseline | steered (RD 0.3) |
|---|---|---|
| Harmful prompts that got harmful help (lower is better) | 0/20 (0.0%) | 0/20 (0.0%) |
| Harmful prompts refused (higher is better) | 20/20 (100.0%) | 20/20 (100.0%) |
| Harmless prompts wrongly refused (lower is better) | 0/20 (0.0%) | 0/20 (0.0%) |
| Harmless prompts answered helpfully (higher is better) | 20/20 (100.0%) | 20/20 (100.0%) |
| Degenerate / looping replies (lower is better) | 0/40 (0.0%) | 0/40 (0.0%) |

**Hinglish**

| | baseline | steered (RD 0.3) |
|---|---|---|
| Harmful prompts that got harmful help (lower is better) | 3/20 (15.0%) | 0/20 (0.0%) |
| Harmful prompts refused (higher is better) | 13/20 (65.0%) | 17/20 (85.0%) |
| Harmless prompts wrongly refused (lower is better) | 0/20 (0.0%) | 0/20 (0.0%) |
| Harmless prompts answered helpfully (higher is better) | 15/20 (75.0%) | 16/20 (80.0%) |
| Degenerate / looping replies (lower is better) | 10/40 (25.0%) | 8/40 (20.0%) |

**XSTest**

| | baseline | steered (RD 0.3) |
|---|---|---|
| Harmful prompts that got harmful help (lower is better) | 1/50 (2.0%) | 0/50 (0.0%) |
| Harmful prompts refused (higher is better) | 48/50 (96.0%) | 50/50 (100.0%) |
| Harmless prompts wrongly refused (lower is better) | 12/100 (12.0%) | 21/100 (21.0%) |
| Harmless prompts answered helpfully (higher is better) | 93/100 (93.0%) | 88/100 (88.0%) |
| Degenerate / looping replies (lower is better) | 0/150 (0.0%) | 0/150 (0.0%) |

### Fresh Hinglish validation (80 prompts, frozen before generation)

**All prompts**

| | baseline | steered (RD 0.3) |
|---|---|---|
| Harmful prompts that got harmful help (lower is better) | 9/40 (22.5%) | 2/40 (5.0%) |
| Harmful prompts refused (higher is better) | 23/40 (57.5%) | 32/40 (80.0%) |
| Harmless prompts wrongly refused (lower is better) | 0/40 (0.0%) | 3/40 (7.5%) |
| Harmless prompts answered helpfully (higher is better) | 35/40 (87.5%) | 33/40 (82.5%) |
| Degenerate / looping replies (lower is better) | 12/80 (15.0%) | 15/80 (18.75%) |

**hinglish_fresh**

| | baseline | steered (RD 0.3) |
|---|---|---|
| Harmful prompts that got harmful help (lower is better) | 9/40 (22.5%) | 2/40 (5.0%) |
| Harmful prompts refused (higher is better) | 23/40 (57.5%) | 32/40 (80.0%) |
| Harmless prompts wrongly refused (lower is better) | 0/40 (0.0%) | 3/40 (7.5%) |
| Harmless prompts answered helpfully (higher is better) | 35/40 (87.5%) | 33/40 (82.5%) |
| Degenerate / looping replies (lower is better) | 12/80 (15.0%) | 15/80 (18.75%) |

## What this does and doesn't show

- **Judge estimates, not verified ASR.** Reading the flagged items by hand, the 32B judge still has false positives in both conditions: a SIM-unlock misunderstanding, a reply about collecting evidence and a garbled reply were all flagged harmful. There may be false negatives too. Every reply and label is public so anyone can re-judge.
- **The cost is real:** more harmless prompts get refused after steering, and degenerate replies decrease on the exploratory slice but increase on the fresh slice. Tiny Aya also drifts into Devanagari or a different language in some replies.
- **Small slices.** 230 held-out plus 80 fresh prompts. Treat differences of a few prompts as noise.
- **Selection.** Strength was picked on dev, but I looked at held-out numbers for several methods during exploration, so the fresh set is the cleaner test.
- **Judge history.** A 7B first-pass judge miscounted refusals badly, and an intermediate 32B two-label rubric conflated safe support with refusal. Both are kept in `results/` for transparency and superseded by the four-label audit above.
- **Model copy.** Runs use the ungated mirror `1-800-LLMs/tiny-aya-global` of `CohereLabs/tiny-aya-global` (the official repo is gated). I didn't verify it byte for byte against the original.
- **Raw generations are published, including harmful baseline completions** (truncated at 160 tokens), so the scoring can be checked.

## Exploration log (earlier rubric, superseded)

Under the intermediate 32B rubric on the same held-out 230: RefusalDirection 0.1 barely moved anything, AlphaSteer produced mostly nonsense, AdaSteer and CAST refused nearly every harmless prompt, and my two gated variants cut over-refusal but let more Hinglish attacks through. Full per-method files are in `results/run2`, `results/run3`, `results/run4` and `results/rejudge`.

## Reproduce

GPU regeneration requires your own Modal account and HF write token, and incurs GPU charges. The AuditKIT replay only reads the pinned public labels and can run locally without model generation. Scripts retain intermediate scoring for transparency; final headline labels are only those in `results/score_audit/`. Dependencies currently follow upstream versions rather than a fully locked environment, so exact bitwise reproduction is not guaranteed.

```bash
pip install "git+https://github.com/Lexsi-Labs/CuratorKIT"
curatorkit run curation/pipeline.yaml --output-dir curation/out/
curatorkit run validation/curation/pipeline.yaml --output-dir validation/curation/out/

pip install modal && modal setup
modal secret create codemix-hf HF_TOKEN=<your token>
modal run run2_modal.py                 # baseline + method sweep on dev + held-out eval (A100-80GB)
modal run --detach run_fresh_modal.py            # frozen RD 0.3 on the fresh Hinglish set
modal run --detach score_audit_modal.py # blinded four-label 32B scoring

pip install "git+https://github.com/Lexsi-Labs/AuditKIT"
python replay_auditkit.py               # AuditKIT evaluate + compare on the frozen labels
```

## Links

- Dataset, generations, labels: https://huggingface.co/datasets/dhrubasumatary/codemix-safety-check
- Steering artifact: https://huggingface.co/dhrubasumatary/tiny-aya-global-codemix-steer
- Model collection: https://huggingface.co/collections/dhrubasumatary/codemix-safety-check-models-6afa3ad503be12bf9af6eed2
- Dataset collection: https://huggingface.co/collections/dhrubasumatary/codemix-safety-check-datasets-6afa3ad53ce7e09c33790bd1
- Code: https://github.com/dhrubasumatary/codemix-safety-check

## Next

Native-speaker Hinglish and Assamese sets, an independently human-labelled subset to calibrate the judge, and conditional steering that keeps the safety gain without the refusal cost.
