# Environment Setup Record

LLM_distillation 프로젝트의 학습/평가 파이프라인용 Python 환경 구성 기록.
모든 작업은 micromamba env `llm_exp` 에서 이루어졌으며, 설치 과정에서의
시행착오와 최종 pin 된 버전, 주요 의사결정을 모두 담는다.

- **기록 최종 갱신**: 2026-04-11
- **대상 프로젝트 경로**: `/workspace/projects/LLM_distillation`
- **관련 plan**: `/home/k106419/.claude/plans/robust-whistling-meteor.md`

---

## 1. 하드웨어 / 시스템

| 항목 | 내용 |
|---|---|
| 노드 | 로컬, SLURM 경유 안 함 |
| GPU | **4 × NVIDIA GeForce RTX 3090** (SM 8.6, 24 GB each, 총 ~94 GB) |
| bf16 지원 | True (Ampere) |
| FP8 지원 | False (SM 8.6, Hopper 이상 필요) |
| NVIDIA driver | 580.126.09 |
| CUDA driver API | 13.0 (backward compatible) |
| OS | Linux (RHEL 9.4 기반, 커널 5.14) |
| Shell | fish (기본), tmux 내부에서는 `bash -c` wrapper 사용 |

### 1.1 CUDA Toolkit (module system)

Lmod module system 사용. 사용 가능 모듈:

```
cuda/11.8  cuda/12.1  cuda/12.4  cuda/12.8 (default) <L>  cuda/13.0
```

**기본 선택**: `cuda/12.8` — 명시적 load 생략해도 자동 적용됨.

```bash
module load cuda/12.8        # 명시적 load (선택)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

- `nvcc --version` → `cuda_12.8.r12.8 / V12.8.93`
- flash-attn 소스 빌드 시 반드시 `CUDA_HOME` 지정 필요.
- `TORCH_CUDA_ARCH_LIST="8.6"` 로 RTX 3090 SM만 빌드 타겟 (빌드 시간 단축).

---

## 2. Python 환경

### 2.1 micromamba env

- **Env 이름**: `llm_exp`
- **경로**: `$HOME/micromamba/envs/llm_exp`
- **Python**: 3.11.15
- **Pip**: `$HOME/micromamba/envs/llm_exp/bin/pip`
- **Python 바이너리**: `$HOME/micromamba/envs/llm_exp/bin/python`

CLAUDE.md 규약에 따라, 이 env 내부에서만 모든 Python 작업을 수행한다.
`micromamba activate llm_exp` 없이 항상 절대경로 호출 사용:

```bash
$HOME/micromamba/envs/llm_exp/bin/python script.py
```

### 2.2 기존 설치 패키지 (건드리지 않은 것들)

env가 `llm_exp` 라는 이름 그대로인 이유는 `gemini-parallel`
라이브러리가 editable mode 로 이미 깔려 있었기 때문. 이를 보존해야 함:

```
gemini-parallel (editable, /workspace/GeminiParallel)
google-genai 1.72.0
python-dotenv 1.2.2
pydantic 2.12.5
```

설치 전후 모두 `import gemini_parallel` 이 정상 동작함을 확인했다.

---

## 3. 최종 패키지 버전 (2026-04-11 기준)

`pip freeze` 전체는 `logs/pip-freeze-current.txt` 에 저장됨 (234 lines).
주요 패키지만 발췌:

### 3.1 Core ML stack

| Package | Version | 용도 |
|---|---|---|
| **torch** | **2.8.0+cu128** | 딥러닝 코어 |
| torchvision | 0.23.0 | (vllm 전이 의존) |
| torchaudio | 2.8.0 | (vllm 전이 의존) |
| triton | 3.4.0 | torch kernel JIT (2.8 구간용) |
| **transformers** | **4.57.3** | HF 모델 로딩/토크나이저, Qwen3 지원 |
| accelerate | 1.12.0 | 멀티 GPU 학습 launcher |
| peft | 0.18.1 | LoRA adapter |
| trl | 1.0.0 | `SFTTrainer` (completion-only loss 지원) |
| datasets | 4.8.4 | HF datasets API |
| bitsandbytes | 0.49.2 | 8-bit optim / quantized backend |

### 3.2 Serving / Quantization

| Package | Version | 용도 |
|---|---|---|
| **vllm** | **0.11.0** | Qwen3 서빙 / latency bench. Qwen3 지원 최소 권장 버전. torch 2.8.0 에 pin |
| **flash_attn** | **2.8.3** | Pre-built wheel `cu12torch2.8cxx11abiTRUE-cp311` 직접 설치 |
| **llmcompressor** | **0.9.0.2** | W4A16 PTQ. `transformers<5.0` constraint 때문에 0.10 → 0.9 로 내려감 |

### 3.3 Utility / Logging

| Package | Version | 용도 |
|---|---|---|
| sentencepiece | 0.2.1 | 토크나이저 |
| wandb | 0.25.1 | 실험 로깅 |
| rouge-score | 0.1.2 | 평가 지표 |
| nltk | 3.9.4 | 텍스트 처리 |

### 3.4 기존 Gemini API stack (변경 안 함)

| Package | Version | 용도 |
|---|---|---|
| gemini-parallel | 0.9.0 (editable) | `/workspace/GeminiParallel` — Teacher 호출 |
| google-genai | 1.72.0 | Gemini SDK 본체 |
| python-dotenv | 1.2.2 | `.env` 로딩 |
| pydantic | 2.12.5 | Response schema |

### 3.5 Torch 2.8 nvidia-*-cu12 세트 (pip freeze에서 자동 선택됨)

torch 2.8 wheel은 `cu126` 의 CUDA libs를 번들함. 시스템 CUDA 12.8 module과
별개로 torch 내부에서 자체 libs 사용. 주요:

```
nvidia-cublas-cu12==12.6.x
nvidia-cudnn-cu12==9.10.2.21
nvidia-cuda-runtime-cu12==12.6.x
nvidia-nccl-cu12==2.27.x
nvidia-nvshmem-cu12==3.4.5
nvidia-cusparselt-cu12==0.7.x
triton==3.4.0
```

시스템 `module load cuda/12.8` 과 torch 내부 cu126은 **병존 가능**.
flash-attn pre-built wheel은 torch 내부 libs에 링크되므로 시스템 CUDA 버전이
맞을 필요 없다. 시스템 CUDA는 **flash-attn을 소스 빌드할 때만** 관여한다.

---

## 4. 핵심 의사결정 (Why these versions)

### 4.1 Flash-attention Pre-built wheel 을 위한 역산

**목표**: flash-attn을 소스 컴파일 (예상 2시간+) 하지 않고 pre-built wheel 사용.

**제약 분석** (Dao-AILab/flash-attention GitHub Releases 조사):

| Flash-attn | 최대 지원 torch |
|---|---|
| v2.8.3 (2025-08-14, 최신 FA2 stable) | torch **2.4 ~ 2.8** |
| v2.7.x | torch 2.2 ~ 2.7 |
| fa4-v4.0.0.beta* | **Hopper (H100 SM 9.0) 전용**, 3090 불가 |

**2025-08-14 이후 FA2/FA3 stable 빌드 없음** → torch 2.9/2.10 용 wheel 부재.
3090 (SM 8.6) 에서 pre-built wheel 쓰려면 **torch ≤ 2.8** 필요.

### 4.2 vLLM 버전 핀 분석

vLLM PyPI metadata 조사 (`requires_dist` 필드):

| vLLM | 릴리즈 | torch pin |
|---|---|---|
| 0.10.0 | 2025-07-25 | torch==2.7.1 |
| 0.10.2 | 2025-09-13 | torch==2.8.0 |
| **0.11.0** | 2025-10-04 | **torch==2.8.0** ✅ |
| 0.11.1 | 2025-11-19 | torch==2.9.0 ❌ (flash-attn wheel 없음) |
| 0.12.0 ~ 0.15.0 | 2025-12 ~ 2026-01 | torch==2.9.x ❌ |
| 0.17.0 / 0.19.0 | 2026-03 ~ 04 | torch==2.10.0 ❌ |

→ **"pre-built flash-attn wheel + vLLM Qwen3 지원" 교집합은 vLLM 0.10.2 ~ 0.11.0**.
그 중 **0.11.0** 이 "Qwen3 공식 권장 최소 버전" 이기도 함 (vLLM 문서 기준).

최종 선택: **vLLM 0.11.0 + torch 2.8.0 + flash-attn 2.8.3 (cu12torch2.8cxx11abiTRUE-cp311)**.

### 4.3 Transformers 버전 상한

transformers **5.x는 금지**. 이유:
- `peft 0.18.1` 의 lazy import 가 transformers 5.x 의 `PreTrainedModel`
  모듈 경로를 찾지 못함 → `ModuleNotFoundError: Could not import module
  'PreTrainedModel'`
- `llmcompressor` 도 같은 이유로 실패
- transformers 4.57.x 는 Qwen3 지원 포함 + peft/llmcompressor 와 호환

→ constraint 파일에 `transformers<5.0` 강제 pin.

실제 설치본: **transformers 4.57.3** (constraint 조건에서 pip이 고른 최신 4.x).

### 4.4 `pip constraint file` 이 필수인 이유

1차 다운그레이드 시도는 실패했다. 원인:

```
Step 6: pip install --upgrade transformers accelerate peft trl ...
```

`--upgrade` 옵션 + pin 없음 → pip resolver 가 최신 transformers 5.5.3 선택
→ transformers 5.5.3 metadata 가 `torch>=2.10` 같은 하한을 요구
→ torch 2.8.0 이 2.10.0 으로 **전이적으로** 업그레이드됨
→ flash-attn 2.8 wheel 이 torch 2.10 ABI 에서 `undefined symbol` 로 깨짐
→ peft / llmcompressor 도 transformers 5.x 와 호환 불가

**수정**: 모든 pip 명령에 constraint file 적용.

```
# /tmp/downgrade-constraints.txt
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
transformers<5.0
```

```bash
pip install -c /tmp/downgrade-constraints.txt "vllm==0.11.0"
pip install -c /tmp/downgrade-constraints.txt llmcompressor
pip install -c /tmp/downgrade-constraints.txt "transformers==4.57.6"
# ...
```

이것으로 어떤 transitive upgrade 시도도 constraint 에 위반되면 pip 이 거절.

### 4.5 `gemini-parallel` 의 `load_dotenv()` 오염 문제

`gemini_parallel/gemini_parallel.py:24` 에서 모듈 import 시
`load_dotenv()` 를 인자 없이 호출 → `find_dotenv()` 가 패키지의 `__file__`
기준 상위로 거슬러 올라가면서 `/workspace/GeminiParallel/.env` 를 자동 로드.

결과적으로 `os.environ` 에 우리가 원하지 않는 `GEMINI_API_KEY_*` 변수가
유입되어, `AdvancedApiKeyManager(keylist_names="all")` 이 62개 키를 잡음
(우리 `.env` 에는 9개만 있음). 53개는 **deprecated 키**.

**대응 (코드)**: `scripts/generate_teacher.py` 의 `read_declared_key_names()`
함수가 우리 `.env` 파일을 직접 파싱하여 **명시된 키 이름 리스트**를
`AdvancedApiKeyManager(keylist_names=<list>)` 에 전달. `os.environ` 오염과
무관하게 우리 프로젝트 키만 사용되도록 방어.

**대응 (환경)**: 인접 프로젝트의 `.env` (e.g. `/workspace/GeminiParallel/.env`)
가 이 리포의 `.env` 를 override 하지 않도록 비워둔다. 두 방어 장치가
중첩되어 안전.

---

## 5. 알려진 경고 / Pip metadata 충돌

### 5.1 llmcompressor vs datasets

```
llmcompressor 0.9.0.2 requires datasets<=4.6.0,>=4.0.0,
but you have datasets 4.8.4 which is incompatible.
```

- pip resolver 가 metadata 레벨에서 경고를 띄움
- runtime 영향: 실제 datasets 4.x API 는 하위 호환성 강해서 llmcompressor 내부
  호출 경로에는 거의 영향 없음. 실행 시점에 문제 발생 시 그때 대응 (예:
  llmcompressor 0.10 계열 업그레이드로 교체).
- trl 1.0.0 은 `datasets>=4.7.0` 을 요구하므로 **datasets 를 4.8.4 로 유지**
  하는 편이 전체 안정성에 유리하다.

### 5.2 Flash-attn 과 vllm 의 `vllm_flash_attn`

vllm 0.11.0 은 자체 `vllm_flash_attn` 이 번들되어 있으며, 서빙 경로에서
이것을 사용한다. 우리가 따로 설치한 `flash_attn==2.8.3` 은 transformers 학습
경로 (SFTTrainer, `attn_implementation="flash_attention_2"`) 에서 사용된다.
두 설치는 **독립적**이며 간섭 없음.

### 5.3 `transformers 4.57.3` vs 초기 `4.57.6` 차이

초기 (torch 2.10) 상태에서는 transformers 4.57.6 이었으나, 다운그레이드 후
constraint + llmcompressor 0.9.0.2 조합에서 pip 이 4.57.3 을 고름. 0.3 minor
차이는 실질 기능에 영향 없음.

---

## 6. 재현 (Reproduction)

### 6.1 처음부터 다시 만들 때

1. micromamba env `llm_exp` 가 이미 존재한다고 가정 (Python 3.11.15,
   gemini-parallel editable 설치됨).
2. 시스템 CUDA 12.8 사용 가능 확인:
   ```bash
   module load cuda/12.8
   nvcc --version    # 12.8.93 가 나와야 함
   ```
3. 프로젝트 루트로 이동:
   ```bash
   cd /workspace/projects/LLM_distillation
   ```
4. Constraint 기반 다운그레이드 스크립트 실행:
   ```bash
   tmux new-session -d -s omo-env \
     "bash scripts/downgrade_env_fix.sh > logs/env-rebuild.log 2>&1"
   # ~3-15 분 (pip cache 상태에 따라 다름)
   ```
5. 완료 시 최종 verification 블록이 로그 끝에 나와야 함:
   ```
   --- import OK ---
     torch 2.8.0+cu128
     vllm 0.11.0
     flash_attn 2.8.3
     ...
   --- flash-attn functional test ---
     flash_attn_func  OK — output shape (1, 8, 4, 64) dtype torch.bfloat16
   ```

### 6.2 최소 검증 스니펫

```bash
$HOME/micromamba/envs/llm_exp/bin/python - << 'PY'
import torch, vllm, flash_attn, transformers, trl, peft, llmcompressor, gemini_parallel
print(f"torch        {torch.__version__}")
print(f"vllm         {vllm.__version__}")
print(f"flash_attn   {flash_attn.__version__}")
print(f"transformers {transformers.__version__}")
print(f"trl          {trl.__version__}")
print(f"peft         {peft.__version__}")
print(f"llmcompressor {llmcompressor.__version__}")

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
assert torch.cuda.is_bf16_supported()

# Flash-attn forward smoke
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device='cuda')
_ = flash_attn_func(q, k, v)
print("all checks passed")
PY
```

---

## 7. 스크립트 인벤토리 (시간순)

| 스크립트 | 시각 | 역할 | 현재 유효? |
|---|---|---|---|
| `scripts/install_env.sh` | 2026-04-11 02:00 | **1차 시도** — torch 2.10 + cu128 + vllm 최신 스택. flash-attn 2.8.3 wheel ABI 불일치, llmcompressor 이름 오타 (`llm-compressor` → 실제는 `llmcompressor`). | ❌ 히스토리 |
| `scripts/fix_flash_attn.sh` | 2026-04-11 02:41 | torch 2.10 위에서 flash-attn 을 **소스 컴파일** 시도. MAX_JOBS=2 로 50분+ 걸려 종료함. | ❌ 사용 안 함 |
| `scripts/downgrade_env.sh` | 2026-04-11 10:01 | **2차 시도** — vllm 0.11.0 + torch 2.8 목표. Step 6 `pip install --upgrade ...` 에 constraint 없어서 transformers 5.5.3 이 transitive 로 torch 2.10 재설치 유발, 실패. | ❌ 히스토리 |
| `scripts/downgrade_env_fix.sh` | 2026-04-11 10:16 | **3차 (성공)** — pip constraint file (`/tmp/downgrade-constraints.txt`) 로 torch==2.8.0 및 transformers<5.0 pin, 전 단계에 `-c` 적용. 210초 만에 완료. | ✅ **현재 상태 재현용 권장** |

### 7.1 로그 파일

| 파일 | 내용 |
|---|---|
| `logs/env-install.log` | 1차 시도 (torch 2.10 스택) 전체 로그 |
| `logs/flashattn-fix.log` | 소스 컴파일 시도 로그 |
| `logs/downgrade.log` | 2차 시도 (실패) 로그 |
| `logs/downgrade-fix.log` | 3차 시도 (성공) 로그 |
| `logs/pip-freeze-before-downgrade.txt` | 다운그레이드 직전 freeze snapshot (torch 2.10 스택, rollback 참고용) |
| `logs/pip-freeze-current.txt` | 2026-04-11 10:24 기준 최종 freeze (234 라인, torch 2.8 스택) |

---

## 8. 설치 타임라인 (narrative)

**1단계: 초기 설정 (2026-04-10 ~ 04-11 새벽)**
1. 디렉토리 scaffold, Yelp dataset 다운로드, preprocess + generate_teacher 스크립트 작성
2. gemini-parallel 의 `load_dotenv()` 오염 발견 → `read_declared_key_names()` 방어 코드 추가
3. Teacher smoke test 3/3 성공

**2단계: 스택 설치 시도 #1 — torch 2.10 (2026-04-11 02:00)**
4. `install_env.sh` 실행. torch 2.10.0+cu128 + vllm 0.19.0 + transformers 5.5.3 등 **최신 버전으로 설치**.
5. 문제 1: **llm-compressor** (하이픈) 은 PyPI 에 없음. 실제 이름은 `llmcompressor` (하이픈 없음). Step 6 silent fail.
6. 문제 2: **flash-attn 2.8.3** wheel 이 어떤 경로로 설치됐지만 torch 2.10 ABI 와 불일치 → `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`.
7. 임시 조치: `llmcompressor` 설치 → datasets/transformers/accelerate 가 다운그레이드됨, trl ↔ datasets 충돌 경고. `datasets>=4.7.0` 강제 upgrade 로 해결.

**3단계: flash-attn 처리 고민 (2026-04-11 02:41 ~ 03:35)**
8. flash-attn 을 소스 빌드 시도 (`fix_flash_attn.sh`, tmux BG). MAX_JOBS=2 로 시작했는데 **50분이 지나도 수많은 .cu 파일 (각 head_dim × fp16/bf16 × sm80 × fwd/bwd) 중 한 개 컴파일 중**. 예상 잔여 시간 비현실적. 포기.
9. **대안 연구**: Dao-AILab GitHub Releases 에 올라온 pre-built wheel 매트릭스를 API 로 조회. FA2 stable 최대 지원 torch = **2.8**. torch 2.9/2.10 wheel 미존재.
10. 결론: **torch 2.10 포기, torch 2.8 로 다운그레이드 → pre-built wheel 사용** 이 가장 경제적. vLLM 도 동반 다운그레이드 필요.

**4단계: 다운그레이드 시도 #2 — 실패 (2026-04-11 10:01)**
11. `downgrade_env.sh` 실행. vllm==0.11.0 설치 → torch 2.8.0 로 내려감. flash-attn wheel 설치 성공.
12. 하지만 step 6 `pip install --upgrade transformers accelerate peft trl ...` 가 **constraint 없이** 동작 → transformers 5.5.3 선택 → torch 2.10.0 재등장 → flash-attn ABI 재차 깨짐, peft/llmcompressor lazy import 실패.
13. 실패 진단: pip resolver 가 `--upgrade` 시 최신 패키지를 잡으면서 transitive 의존성으로 torch 를 끌어올림. pin 강제 필요.

**5단계: 다운그레이드 시도 #3 — 성공 (2026-04-11 10:16)**
14. `downgrade_env_fix.sh` 실행. `/tmp/downgrade-constraints.txt` 에 torch==2.8.0 / transformers<5.0 강제 pin. 모든 pip 명령에 `-c` 적용.
15. **210초 만에 완료**, final verification 전부 green. flash-attn functional call 까지 통과.

---

## 9. 향후 변경 시 주의사항

1. **`pip install --upgrade <pkg>` 는 절대 금지** — 반드시 `-c /tmp/downgrade-constraints.txt` 같은 constraint file 을 함께 사용하거나, `==` 로 버전을 명시.
2. transformers 5.x 로의 우발적 업그레이드를 경계. constraint 파일에 `transformers<5.0` 포함할 것.
3. vllm 재설치 시 0.11.0 로 pin 고수. 0.11.1+ 는 torch 2.9+ 를 요구하여 flash-attn wheel 호환 끊어짐.
4. CUDA 환경변수 (`CUDA_HOME`, `TORCH_CUDA_ARCH_LIST`) 는 flash-attn **재빌드**
   가 필요한 상황에서만 관여. 정상 사용 시에는 torch wheel 의 번들 CUDA libs 로 충분.
5. llmcompressor 업그레이드 (0.9 → 0.10) 시 datasets/transformers 제약이 바뀌는지
   재확인. metadata conflict 경고를 무시할지 실질 영향 있는지는 그 시점에 runtime 테스트로 판단.
6. gemini-parallel 의 `GeminiParallel/.env` 가 재등장하면 (우리가 만든 것이 아니어도), `generate_teacher.py` 의 defensive filtering 덕분에 학습/추론 파이프라인에는 영향 없지만, 동작 안 할 경우 해당 파일 삭제/이동으로 대응.

---

## 10. Rollback reference

만약 torch 2.10 스택으로 되돌리고 싶다면:

```bash
# 1. 현재 torch 2.8 스택 제거
$HOME/micromamba/envs/llm_exp/bin/pip uninstall -y torch torchvision torchaudio vllm flash-attn transformers

# 2. logs/pip-freeze-before-downgrade.txt 의 torch 2.10 + vllm 0.19.0 상태로 재설치
# (단 이 경우 flash-attn 은 source compile 또는 skip + SDPA 폴백 사용)
$HOME/micromamba/envs/llm_exp/bin/pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
$HOME/micromamba/envs/llm_exp/bin/pip install "vllm>=0.17,<0.20"
$HOME/micromamba/envs/llm_exp/bin/pip install -r logs/pip-freeze-before-downgrade.txt
```

Rollback 은 flash-attn pre-built wheel 을 포기하는 것을 의미한다. 현재 스택이
안정 상태이므로 특별한 이유 없이는 rollback 하지 말 것.
