# `scripts/serve/` — vLLM serve launchers + HTTP teacher adapter

Long-running processes (vLLM HTTP/OpenAI endpoints, TRL `vllm-serve`) plus the
adapter the trainer uses to consume them. Every launcher assumes you run from
the **project root** so the relative paths in `exec` lines resolve.

## Teacher serve

| Script | Hardware | Endpoint shape | Notes |
|---|---|---|---|
| `serve_teacher_vllm.sh` | Generic vLLM-serve wrapper. Originally 4× 4090 TP=4. | OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`) | Bf16 Qwen3.5-35B-A3B, `--max-logprobs 100`, CUDA graphs **on** since  (was `--enforce-eager`, audit removed it for 10× speedup). Tunables via env vars: `TP`, `DEVICES`, `GPU_MEM_UTIL`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `PORT`. |
| `launch_teacher_pro6000.sh` | **Single PRO 6000 (96 GB, SM 12.0)** | Same OpenAI endpoint | Wraps `serve_teacher_vllm.sh` with `module load cuda/13.0` + `FLASHINFER_CUDA_ARCH_LIST="12.0f"` (: flashinfer 0.6.7 explicitly gates SM 12.x on CUDA ≥ 12.9). Current production teacher launcher. |
| `slurm_teacher_pro6000.sbatch` | Slurm wrapper around `launch_teacher_pro6000.sh` | — | Same config; submit with `sbatch`. |
| `launch_teacher_trl_serve.sh` | Same PRO 6000 box | **TRL custom endpoints** (`/get_world_size`, `/init_communicator`, `/update_named_param`) | Used by `train_student_distill.py`'s `VLLMClient` mode. **Not OpenAI-compatible** — the `serve_teacher_vllm.sh` endpoint will not work for `DistillationTrainer`. |

## Student serve

| Script | What it serves |
|---|---|
| `serve_vllm.sh` | Background launcher for the FP16 (port 8000) **and** W4A16 (port 8001) student vLLM servers. Used by `eval/bench_latency.py` and `eval/eval_metrics_vllm.py`. Subcommands: bare = launch both, `stop` = kill, `status` = ping. |
| `launch_student_trl_serve.sh` | Stands up the v3-sft (or v3-9b-sft) student via `trl vllm-serve` so `DistillationTrainer` in **server mode** can connect to it for on-policy generation. |

## Adapter

| Script | Class | Used by |
|---|---|---|
| `http_teacher_adapter.py` | `HTTPTeacherAdapter` (`nn.Module` shim around vLLM `/v1/completions`) + `TeacherHTTPError` | `train/train_student_gkd.py`. Exposes `.forward(input_ids, ...)` returning a dense `[B, T, V]` logits tensor where the top-K logprobs are filled and the remaining `vocab − K` slots sit at `--teacher-fill-logit` (default `-50.0`). Lives in `serve/` because it is the **client side** of the serve contract — bug fixes flow with the launcher, not the trainer. |

## Operational notes

- Teacher endpoint runs as `tmux teacher` on the serve host. Survives shell exit; relaunch via `tmux new-session -d -s teacher 'bash scripts/serve/launch_teacher_pro6000.sh'`.
- Cold start on PRO 6000 first boot ≈ 7 min (flashinfer JIT-compiles SM 12.0 CUTLASS MoE kernels into `~/.cache/flashinfer/0.6.7/120f/`). Subsequent boots hit the cache and start in ~30 s.
- `prompt_logprobs=K` materializes a `[B, T, V]` logits tensor — Qwen3.5's V = 248 044 makes this expensive. PRO 6000 absorbs it with margin; 4× 4090 TP=4 needed `MAX_MODEL_LEN=4096 + MAX_NUM_SEQS=64` to fit ( / ).
- Defensive flag inventory: `--enforce-eager` was kept around from  (vLLM 0.11 + Ada SM 8.9 + W4A16 + flashinfer crash) and silently cost 10× latency on Ada bf16. **Audit every workaround flag when the stack changes** .
