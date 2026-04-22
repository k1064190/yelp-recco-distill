# `scripts/` — Pipeline entrypoints

This directory holds every command-line tool the project exposes. Files are
grouped by **lifecycle stage**, not by ownership. Each subfolder has its own
`README.md` with deeper detail; this file is the index.

## Layout

| Folder | What lives there |
|---|---|
| [`env/`](env/README.md) | micromamba env bootstrap (`llm_blackwell`, `llm_gkd`, `llm_ptq_qwen35`) |
| [`data/`](data/README.md) | Yelp preprocessing, sequence-length profiling, schema migration |
| [`teacher/`](teacher/README.md) | Teacher generation (Qwen3.5-35B-A3B via vLLM), permutation debiasing, position-bias audit, schema validator, rubric perturbation |
| [`serve/`](serve/README.md) | vLLM / TRL serve launchers (teacher on PRO 6000, student bf16 / W4A16), Slurm wrappers, `HTTPTeacherAdapter` |
| [`train/`](train/README.md) | SFT (`train_student.py`), legacy GKD (`train_student_gkd.py`), DistillationTrainer (`train_student_distill.py`), Guided GKD subclass (`guided_gkd.py`) |
| [`quantize/`](quantize/README.md) | W4A16 GPTQ, NF4 (bnb), GGUF Q4_K_M, W8A16 |
| [`vllm_compat/`](vllm_compat/README.md) | VLM-shell re-hydration so vLLM can load Qwen3.5 fine-tunes |
| [`eval/`](eval/README.md) | Retrieval metrics (HF / vLLM-guided / GGUF), latency bench, throughput sweep, inference cache, comparison dashboard |
| [`judge/`](judge/README.md) | Listwise LLM-as-a-Judge (rubric v3: Groundedness + Personalization + Ranking Coherence), validation, analyzer |

Pin rationale for each env lives in [`../ENV_VERSION.md`](../ENV_VERSION.md).

## Conventions

- All Python files start with a 2-line `# ABOUTME:` comment.
- Imports use the absolute `scripts.<subfolder>.<module>` form. `tests/conftest.py` puts the project root on `sys.path` so this works without `__init__.py`.
- Shell launchers always reference siblings by **full path from the project root** (`scripts/serve/serve_teacher_vllm.sh`, never `./serve_teacher_vllm.sh`). Run them from the project root.
- DeepSpeed / training configs that are not entrypoints live in the project-root [`configs/`](../configs/) directory (e.g. `configs/ds_zero2.json`, `configs/teacher_prompt.py`).

## Discovering the right entrypoint

| Goal | File |
|---|---|
| Build a fresh teacher dataset (HTTP identity path) | `teacher/generate_teacher_qwen.py` or `teacher/generate_teacher_permutation.py` + `teacher/merge_teacher_permutations.py` |
| Audit a teacher dataset for position bias | `teacher/analyze_position_bias.py`, `teacher/visualize_position_bias.py`, `teacher/plot_slot1_matrix.py`, `teacher/plot_v4_rank_slot_heatmap.py` |
| Train an SFT baseline | `train/train_student.py` |
| Train on-policy guided GKD (winning path in this project) | `train/guided_gkd.py` |
| Try TRL DistillationTrainer (server-based, experimental) | `train/train_student_distill.py` |
| Stand up the teacher endpoint | `serve/launch_teacher_pro6000.sh` |
| Eval a checkpoint (HF free generation) | `eval/eval_metrics.py` |
| Eval base vs SFT under guided JSON | `eval/eval_metrics_vllm.py` |
| Compare every checkpoint side-by-side | `eval/compare_results.py` |
| Latency / throughput bench | `eval/bench_latency.py`, `eval/eval_metrics_vllm_offline.py` |
| LLM-as-a-Judge scoring (rubric v3) | `judge/judge_listwise.py` + `judge/analyze_judge_listwise.py` |
| Rubric counterfactual validation | `teacher/perturb_teacher_outputs.py` + `judge/analyze_judge_validation.py` |

## Reading order for new readers

1. Project root [`README.md`](../README.md) (full portfolio narrative with numbers, methodology, and citations to the specific scripts above).
2. The subfolder `README.md` for the area you are touching.
3. [`ENV_VERSION.md`](../ENV_VERSION.md) for env pins, forbidden operations, and
   reproduction setup.
