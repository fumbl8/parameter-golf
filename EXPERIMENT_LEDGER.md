# EXPERIMENT_LEDGER.md

## 2026-03-25 - Baseline Smoke Audit

### Timestamp

- `2026-03-25`
- Time zone: `America/Indianapolis`

### Environment Summary

- OS / shell: `Windows + PowerShell`
- Python: `3.14.3`
- Branch: `codex/helix-genome-lm`
- Commit: `0e5b19843bf4d313da3fb9f9f9e97ae766b18524`

### Command

```bash
python --version

@'
import importlib.util
mods=['torch','numpy','sentencepiece','tqdm']
for m in mods:
    print(m, bool(importlib.util.find_spec(m)))
'@ | python -

Get-ChildItem data -Recurse | Select-Object FullName, Length | Format-Table -AutoSize

nvidia-smi

@'
paths = [
  'train_gpt.py',
  'records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py',
  'records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py',
]
for path in paths:
    src = open(path, 'r', encoding='utf-8').read()
    compile(src, path, 'exec')
    print(f'OK {path}')
'@ | python -

@'
from pathlib import Path
checks = {
  'root_dataset_dir': Path('data/datasets/fineweb10B_sp1024').exists(),
  'root_tokenizer_file': Path('data/tokenizers/fineweb_1024_bpe.model').exists(),
  'baseline_record_log': Path('records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log').exists(),
}
for k,v in checks.items():
    print(f'{k}={v}')
'@ | python -
```

### Outcome

- `python --version`: `Python 3.14.3`
- Module probe: `torch=False`, `numpy=False`, `sentencepiece=False`, `tqdm=False`
- Local `data/` tree contains helper scripts only. The published dataset and tokenizer payloads are absent.
- `nvidia-smi` is not available in this environment.
- Syntax smoke passed:
  - `OK train_gpt.py`
  - `OK records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py`
  - `OK records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- Path checks:
  - `root_dataset_dir=False`
  - `root_tokenizer_file=False`
  - `baseline_record_log=True`
- Baseline status: real training smoke not run locally. This environment only supports a syntax-level smoke audit.

### Key Metrics

- Local baseline result: blocked before training start.
- Repo-documented reference baseline from `records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`:
  - `val_loss=2.07269931`
  - `val_bpb=1.2243657` post-quant roundtrip
  - `val_bpb=1.2172` pre-quant at stop
  - `bytes_total=15,863,489`
  - `train_time=600038ms`
- Current accepted SOTA in this repo snapshot:
  - `1.1194 val_bpb`
  - `LeakyReLU^2 + Legal Score-First TTT + Parallel Muon`
  - `2026-03-23`

### Blocker Notes

- Missing Python packages needed for training and evaluation in this environment: `torch`, `numpy`, `sentencepiece`, `tqdm`.
- Missing published dataset directory: `data/datasets/fineweb10B_sp1024`
- Missing tokenizer file: `data/tokenizers/fineweb_1024_bpe.model`
- No visible CUDA tooling or GPU access: `nvidia-smi` command not found

### Recommendation

- Treat this entry as a truthful smoke-baseline status report, not a reproduced baseline.
- On proper hardware, run the next commands in order:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=hf_verify_sp1024_8gpu \
DATA_PATH=/root/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Record any future run with exact command, runtime, `val_loss`, `val_bpb`, compressed model size, total bytes, and blockers if the run fails.

## 2026-03-26 - Donor Execution Gate For HelixRecur v1

### Timestamp

- `2026-03-26`
- Time zone: `America/Indianapolis`

### Repo State

- Branch: `codex/helix-genome-lm`
- Commit: `0e5b19843bf4d313da3fb9f9f9e97ae766b18524`
- `upstream/main` available locally: `yes`, at the same commit
- Target donor: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`

### Command

```bash
git branch --show-current
git rev-parse HEAD
git show-ref --verify refs/remotes/upstream/main

python --version

@'
import importlib.util
mods=['torch','numpy','sentencepiece','tqdm']
for m in mods:
    print(f'{m}={bool(importlib.util.find_spec(m))}')
'@ | python -

nvidia-smi

@'
from pathlib import Path
for p in [
    Path('data/datasets/fineweb10B_sp1024'),
    Path('data/tokenizers/fineweb_1024_bpe.model'),
]:
    print(f'{p} exists={p.exists()}')
'@ | python -
```

### Outcome

- Repo state matches the intended donor-planning snapshot.
- The current shell still fails the execution gate:
  - `python --version`: `Python 3.14.3`
  - `torch=False`, `numpy=False`, `sentencepiece=False`, `tqdm=False`
  - `nvidia-smi` command not found
  - `data/datasets/fineweb10B_sp1024` absent
  - `data/tokenizers/fineweb_1024_bpe.model` absent
- Donor reproduction was not attempted in this shell.
- Recurrence implementation was not started in this shell.

### Donor Summary

- Architecture:
  - `11` layers, `512` dim, `8` heads, `4` KV heads, `3x` MLP
  - U-Net layout with `5` encoder passes and `6` decoder passes
  - tied embeddings, logit softcap `30.0`
  - BigramHash `2048`, SmearGate, VE `128` on layers `9,10`
  - Partial RoPE `16/64`, XSA on last `4` layers, LN scale
- Training:
  - Muon matrices `lr=0.025`, momentum `0.99`, WD `0.04`
  - AdamW embeddings `lr=0.035`, scalars `lr=0.025`, WD `0.04`
  - batch `786,432` tokens/step, train/eval seq len `2048`
  - warmdown `3500`, EMA `0.997`, tight SWA every `50`, late QAT threshold `0.15`
- Quantization / compression:
  - GPTQ-lite per-row clip search
  - int6 per-row for MLP and attention weights
  - int8 per-row for embeddings
  - fp32 control tensors
  - zstd level `22`
- Eval:
  - standard int6 roundtrip eval
  - sliding-window eval with stride `64`
- Reference metrics from donor log:
  - `val_loss=1.89576235`
  - `sliding val_bpb=1.12278022`
  - `int6 roundtrip val_bpb=1.14656250`
  - `bytes_total=15555017`
  - `bytes_code=67603`
  - `model_params=26993756`
  - `train_time=600036ms`
  - `compressed_model_bytes=15487414`

### Blocker Notes

- The donor script hard-imports `flash_attn_interface`, so the intended reproduction path is a Linux/CUDA/Hopper environment rather than this Windows shell.
- Creating `records/track_non_record_16mb/2026-03-26_HelixRecur_v1` or editing a donor fork here would produce unmeasured code without a runnable donor baseline, which would confound the recurrence-only experiment.

### Recommendation

- Do the next execution on a proper Linux/CUDA host and start with these exact commands:

```bash
RUN_ID=donor_smoke SEED=1337 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=180 ITERATIONS=400 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

```bash
RUN_ID=donor_seed1337 SEED=1337 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- Only start the recurrence fork after the donor is reproduced within the planned gate.
