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

## 2026-03-26 Donor Reproduction

- Branch: `codex/helix-genome-lm`
- Target donor: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- Status: completed

### Environment Verification

- Python: `3.12.3`
- Packages: `torch 2.9.1+cu128`, `numpy 2.4.3`, `sentencepiece 0.2.1`, `tqdm 4.67.3`
- CUDA: visible, `1x NVIDIA H100 80GB HBM3`
- Dataset path present: `./data/datasets/fineweb10B_sp1024`
- Tokenizer path present: `./data/tokenizers/fineweb_1024_bpe.model`
- Corrected during session: installed `zstandard 0.25.0`; downloaded `80` train shards for `fineweb10B_sp1024`.

### Donor Summary

- Donor command: `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- Donor documented result: `val_loss 1.89576235`, `val_bpb 1.12278022`, `bytes_total 15555017`.
- Local faithful command:

```bash
env RUN_ID=codex-donor-repro SEED=1337 MAX_WALLCLOCK_SECONDS=4800 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=1 \
records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py > codex_donor_repro.out 2>&1
```

### Smoke Summary

- Smoke attempt 1 failed before training because the donor script was launched from the donor subdirectory and its default relative tokenizer path resolved incorrectly.
- Smoke attempt 2 succeeded from repo root with explicit `DATA_PATH` and `TOKENIZER_PATH`.
- Smoke metrics: `step 90`, `60166ms` train time, `step_avg 668.51ms`, `post_ema val_bpb 3.5332`, `final_int6_roundtrip val_bpb 3.54495443`.

### Faithful Reproduction Outcome

- Training stop: `step 7235`, `train_time 4800031ms`, `step_avg 663.45ms`
- Pre-EMA stop metric: `val_loss 1.9235`, `val_bpb 1.1392`
- Post-EMA diagnostic: `val_loss 1.9219`, `val_bpb 1.1383`
- Int6 roundtrip: `val_loss 1.93603026`, `val_bpb 1.14662617`
- Final stride-64 sliding: `val_loss 1.89588330`, `val_bpb 1.12285185`
- Sliding eval wallclock: `584544ms`
- End-to-end run time excluding setup/downloads: about `5384.6s`

### Artifact Sizes

- `final_model.pt`: `106178569` bytes
- `final_model.int6.ptz`: `16073037` bytes
- Total submission size: `16140640` bytes

### Comparison To Donor

- Final sliding `val_bpb` delta vs donor: `+0.00007163`
- This is within the requested `+0.004` donor reproduction tolerance.
- Donor quality was reproduced within tolerance on this pod.

### Remaining Deviations

- Local reproduction used `WORLD_SIZE=1` with grad accumulation `8`, not donor `WORLD_SIZE=8`.
- Local wallclock cap was increased to `4800s` to match donor step depth on one GPU.
- SWA and late-QAT landed slightly later than donor: `swa:start 6550` vs `6450`, `late_qat 6713` vs `6574`.
- Compressed bytes did not match donor: total bytes were `+585623` over donor and exceeded the donor's `15.55 MB` artifact.

## 2026-03-27 HelixRecur v1 Byte Budget Note

- Base donor for recurrence fork: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- Donor compressed model bytes: `16073037`
- Donor total submission bytes: `16140640`
- Current overage vs `16,000,000`: `140640` bytes
- HelixRecur v1 code bytes: `69074` vs donor `67603` (`+1471` code bytes risk)
- HelixRecur v1 parameter count estimate: `15186996` vs donor `26993756`
- Expected recurrence effect: replacing 11 distinct blocks with 6 shared blocks should materially reduce compressed model bytes by reusing transformer weights while preserving donor embeddings, BigramHash, SmearGate, value embeddings, quantization, and eval.
- Main byte risk introduced by the recurrence refactor: extra schedule/runtime plumbing increases counted code modestly, so byte improvement must come primarily from lower model payload rather than code shrinkage.

## 2026-03-27 HelixRecur v1 Implementation And Evaluation

- New folder: `records/track_non_record_16mb/2026-03-26_HelixRecur_v1`
- Major hypothesis only: replace donor's 11 distinct depth blocks with 6 shared recurrent blocks on schedule `0,1,2,3,4,5,4,3,2,1,0`.
- Preserved unchanged from donor: tied embeddings, BigramHash, SmearGate, shared value embeddings, optimizer family, quantization/compression path, and eval path.
- Extra recurrence machinery kept minimal: runtime schedule lookup plus virtual-depth ln-scale/XSA overrides; no routing, no TTT, no tokenizer or dataset changes.

### Commands Run

- Compile sanity:
  - `python -m py_compile records/track_non_record_16mb/2026-03-26_HelixRecur_v1/train_gpt.py`
- Parameter estimate:
  - `python - <<'PY' ... instantiate GPT from records/track_non_record_16mb/2026-03-26_HelixRecur_v1/train_gpt.py ... PY`
- HelixRecur train smoke:
  - `env RUN_ID=helixrecur-train-smoke SEED=1337 MAX_WALLCLOCK_SECONDS=45 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v1/train_gpt.py > helixrecur_train_smoke.out 2>&1`
- HelixRecur eval smoke:
  - `env RUN_ID=helixrecur-eval-smoke SEED=1337 MAX_WALLCLOCK_SECONDS=1 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v1/train_gpt.py > helixrecur_eval_smoke.out 2>&1`
- Donor quick comparison:
  - `env RUN_ID=donor-quickcmp SEED=1337 MAX_WALLCLOCK_SECONDS=180 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py > donor_quickcmp.out 2>&1`
- HelixRecur quick comparison:
  - `env RUN_ID=helixrecur-quickcmp SEED=1337 MAX_WALLCLOCK_SECONDS=180 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v1/train_gpt.py > helixrecur_quickcmp.out 2>&1`

### Results

- HelixRecur train smoke:
  - stop `45.116s`, `step 69`, `step_avg 653.85ms`
  - final roundtrip `val_loss 6.08586880`, `val_bpb 3.60439430`
  - compressed bytes `2,797,394`, total bytes `2,866,468`
- HelixRecur eval smoke:
  - stop `1.329s`, `step 2`
  - final roundtrip `val_loss 6.91002561`, `val_bpb 4.09250639`
  - compressed bytes `2,636,270`, total bytes `2,705,344`
- Donor quick comparison:
  - stop `180.427s`, `step 270`, `step_avg 668.25ms`
  - final roundtrip `val_loss 7.55493163`, `val_bpb 4.47445606`
  - compressed bytes `5,019,273`, total bytes `5,086,876`
- HelixRecur quick comparison:
  - stop `180.212s`, `step 275`, `step_avg 655.32ms`
  - final roundtrip `val_loss 7.85509273`, `val_bpb 4.65222837`
  - compressed bytes `3,081,539`, total bytes `3,150,613`

### Donor vs HelixRecur Quick Comparison

- `val_loss`: recurrence worse by `+0.30016110`
- `val_bpb`: recurrence worse by `+0.17777231`
- Train wallclock: recurrence better by about `0.215s` at stop, with better `step_avg` (`655.32ms` vs `668.25ms`)
- Compressed model bytes: recurrence better by `-1,937,734`
- Total bytes: recurrence better by `-1,936,263`
- Byte compliance story: recurrence moves strongly toward compliance on quick runs, but the quality collapse is too large to justify a longer pass.

### Stability Notes

- Compile sanity passed.
- Train/eval/quantization/compression paths all completed successfully in the new folder.
- Code churn stayed isolated to the new submission folder plus `EXPERIMENT_LEDGER.md`.
- Observed behavior suggests the recurrence fork remains operational but loses too much depth-specific capacity in this first form.

### Judgment

- Kill criteria triggered: `val_bpb` is worse by far more than `0.010` on the quick comparison.
- The byte-efficiency upside is real, but not enough to justify a longer non-record pass for this exact v1 design.
- HelixRecur v1 is a no-go as currently implemented.

## 2026-03-27 HelixRecur v2 Byte Budget Note

- Base for v2: `records/track_non_record_16mb/2026-03-26_HelixRecur_v1`
- Donor reproduced artifact: compressed `16073037`, total `16140640`, over cap by `140640`
- v1 quick artifact: compressed `3081539`, total `3150613`
- v2 adds exactly `44` trainable parameters via an `11 x 4` virtual-depth conditioning table
- v2 parameter count: `15187040` vs v1 `15186996` (`+44`)
- v2 code bytes: `70777` vs v1 `69074` (`+1703` code bytes risk)
- Expected byte effect: preserve recurrence-driven model reuse while spending a negligible parameter budget to recover depth-specific behavior
- Main code-size risk: extra override plumbing for LN scale, attention scale, MLP scale, and `q_gain`

## 2026-03-27 HelixRecur v2 Implementation And Evaluation

- New folder: `records/track_non_record_16mb/2026-03-26_HelixRecur_v2`
- Hypothesis only: keep the v1 recurrent stack and add a tiny virtual-depth conditioning mechanism so repeated passes can recover a small amount of depth-specific specialization
- Preserved unchanged from v1 and donor: tied embeddings, BigramHash, SmearGate, shared value embeddings, optimizer family, quantization/compression path, and eval path
- Added conditioning only on existing scalar pathways: LN scale, attention scale, MLP scale, and attention `q_gain`

### Commands Run

- Compile sanity:
  - `python -m py_compile records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py`
- Instantiate sanity:
  - `python - <<'PY' ... import records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py and instantiate GPT ... PY`
- Train smoke:
  - `env RUN_ID=helixrecur2-train-smoke SEED=1337 MAX_WALLCLOCK_SECONDS=45 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_train_smoke.out 2>&1`
- Eval smoke:
  - `env RUN_ID=helixrecur2-eval-smoke SEED=1337 MAX_WALLCLOCK_SECONDS=1 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_eval_smoke.out 2>&1`
- Initial quick comparison attempt:
  - `env RUN_ID=helixrecur2-quickcmp SEED=1337 MAX_WALLCLOCK_SECONDS=180 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_quickcmp.out 2>&1`
- Fair solo quick comparison used for judgment:
  - `env RUN_ID=helixrecur2-quickcmp-solo SEED=1337 MAX_WALLCLOCK_SECONDS=180 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_quickcmp_solo.out 2>&1`
- Longer non-record pass:
  - `env RUN_ID=helixrecur2-long SEED=1337 MAX_WALLCLOCK_SECONDS=600 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-26_HelixRecur_v2/train_gpt.py > helixrecur2_long.out 2>&1`

### Sanity Results

- Compile sanity passed
- Instantiate sanity:
  - `v2_model_params 15187040`
  - `v2_depth_condition_params 44`
  - `v2_shared_num_layers 6`
  - `v2_virtual_schedule 0,1,2,3,4,5,4,3,2,1,0`

### Smoke Results

- Train smoke:
  - stop `45.986s`, `step 21`, `step_avg 2189.81ms`
  - `val_loss 8.5216`, `val_bpb 5.0470`
  - post-EMA `val_loss 6.6014`, `val_bpb 3.9097`
  - compressed `2674700`, total `2745477`
- Eval smoke:
  - stop `1.587s`, `step 1`
  - `val_loss 8.7284`, `val_bpb 5.1694`
  - post-EMA `val_loss 6.9191`, `val_bpb 4.0979`
  - compressed `2654379`, total `2725156`
- Note on methodology:
  - the first smoke and initial quick run were launched concurrently on one GPU; those runs remain valid sanity checks, but not fair wallclock comparisons against v1

### Quick Comparison Used For Judgment

- Donor quick reference from prior ledger entry:
  - `val_loss 7.55493163`, `val_bpb 4.47445606`, compressed `5019273`, total `5086876`, `step_avg 668.25ms`
- HelixRecur v1 quick reference from prior ledger entry:
  - `val_loss 7.85509273`, `val_bpb 4.65222837`, compressed `3081539`, total `3150613`, `step_avg 655.32ms`
- HelixRecur v2 solo quick:
  - stop `180.484s`, `step 267`, `step_avg 675.97ms`
  - pre-roundtrip stop metric: `val_loss 3.7819`, `val_bpb 2.2398`
  - final roundtrip exact: `val_loss 7.54165596`, `val_bpb 4.46659346`
  - compressed `3042658`, total `3113435`

### v1 vs v2 Quick Delta

- `val_loss`: `-0.31343677`
- `val_bpb`: `-0.18563491`
- `step_avg`: `+20.65ms` (`+3.15%`)
- compressed bytes: `-38881`
- total bytes: `-37178`
- Added conditioning parameter count: `+44`

### Longer Non-record Pass

- HelixRecur v2 long:
  - stop `600.504s`, `step 888`, `step_avg 676.24ms`
  - stop metric: `val_loss 2.4185`, `val_bpb 1.4324`
  - post-EMA `val_loss 2.6283`, `val_bpb 1.5566`
  - final roundtrip exact: `val_loss 4.63764717`, `val_bpb 2.74667588`
  - compressed `4224324`, total `4295101`

### Judgment

- The narrow rescue hypothesis worked on the quick comparison: v2 recovers quality materially relative to v1 while keeping the recurrence byte win.
- The solo runtime story is acceptable: v2 remains close to v1 in `step_avg` once measured without GPU contention.
- Byte efficiency remains clearly favorable versus donor and slightly better than v1.
- HelixRecur v2 is worth keeping as the current recurrence line.

## 2026-03-27 HelixRecur v3 Byte Budget Note

- Base for v3: `records/track_non_record_16mb/2026-03-26_HelixRecur_v2`
- v2 quick artifact: compressed `3042658`, total `3113435`
- v3 change adds exactly `4` trainable parameters: bounded learnable amplitudes for the existing four conditioning channels
- v3 parameter count: `15187044` vs v2 `15187040` (`+4`)
- v3 code bytes: `70888` vs v2 `70777` (`+111` code bytes risk)
- Expected effect: keep the same recurrence hypothesis and same `11 x 4` virtual-depth table, but let each conditioning channel learn its own bounded strength instead of sharing fixed `0.05`
- Main risk: the extra freedom may help short-run specialization but not longer-run convergence

## 2026-03-27 HelixRecur v3 Implementation And Evaluation

- New folder: `records/track_non_record_16mb/2026-03-27_HelixRecur_v3`
- Hypothesis only: keep HelixRecur v2 intact except for replacing the fixed conditioning strength with bounded learnable per-channel amplitudes
- Preserved unchanged from v2: shared blocks, virtual schedule, tied embeddings, BigramHash, SmearGate, shared value embeddings, optimizer family, quantization/compression path, and eval path
- Added parameters only in the conditioning path: `4` amplitude logits, one for each existing conditioned channel

### Commands Run

- Compile sanity:
  - `python -m py_compile records/track_non_record_16mb/2026-03-27_HelixRecur_v3/train_gpt.py`
- Instantiate sanity:
  - `python - <<'PY' ... import records/track_non_record_16mb/2026-03-27_HelixRecur_v3/train_gpt.py and instantiate GPT ... PY`
- Train smoke:
  - `env RUN_ID=helixrecur3-train-smoke SEED=1337 MAX_WALLCLOCK_SECONDS=45 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-27_HelixRecur_v3/train_gpt.py > helixrecur3_train_smoke.out 2>&1`
- Eval smoke:
  - `env RUN_ID=helixrecur3-eval-smoke SEED=1337 MAX_WALLCLOCK_SECONDS=1 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-27_HelixRecur_v3/train_gpt.py > helixrecur3_eval_smoke.out 2>&1`
- Solo quick comparison:
  - `env RUN_ID=helixrecur3-quickcmp SEED=1337 MAX_WALLCLOCK_SECONDS=180 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-27_HelixRecur_v3/train_gpt.py > helixrecur3_quickcmp.out 2>&1`
- Longer non-record pass:
  - `env RUN_ID=helixrecur3-long SEED=1337 MAX_WALLCLOCK_SECONDS=600 EVAL_SEQ_LEN=64 TRAIN_LOG_EVERY=1000 VAL_LOSS_EVERY=4000 DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model torchrun --standalone --nproc_per_node=1 records/track_non_record_16mb/2026-03-27_HelixRecur_v3/train_gpt.py > helixrecur3_long.out 2>&1`

### Sanity Results

- Compile sanity passed
- Instantiate sanity:
  - `v3_model_params 15187044`
  - `v3_depth_condition_params 44`
  - `v3_amplitude_params 4`
  - `v3_added_vs_v2 4`
  - `v3_shared_num_layers 6`
  - `v3_virtual_schedule 0,1,2,3,4,5,4,3,2,1,0`
  - initial amplitudes: `0.05,0.05,0.05,0.05`

### Smoke Results

- Train smoke:
  - stop `45.526s`, `step 67`, `step_avg 679.49ms`
  - `val_loss 6.0885`, `val_bpb 3.6059`
  - post-EMA `val_loss 6.0704`, `val_bpb 3.5952`
  - compressed `2792884`, total `2863772`
- Eval smoke:
  - stop `1.384s`, `step 2`
  - `val_loss 8.7316`, `val_bpb 5.1714`
  - post-EMA `val_loss 6.9062`, `val_bpb 4.0902`
  - compressed `2623182`, total `2694070`

### v2 vs v3 Quick Comparison

- HelixRecur v2 quick reference from prior ledger entry:
  - `val_loss 7.54165596`, `val_bpb 4.46659346`, compressed `3042658`, total `3113435`, `step_avg 675.97ms`
- HelixRecur v3 quick:
  - stop `180.019s`, `step 266`, `step_avg 676.76ms`
  - pre-roundtrip stop metric: `val_loss 3.7825`, `val_bpb 2.2402`
  - final roundtrip exact: `val_loss 7.52554692`, `val_bpb 4.45705278`
  - compressed `3114457`, total `3185345`
- Delta vs v2 quick:
  - `val_loss`: `-0.01610904`
  - `val_bpb`: `-0.00954068`
  - `step_avg`: `+0.79ms` (`+0.12%`)
  - compressed bytes: `+71799`
  - total bytes: `+71910`

### Longer Non-record Pass

- HelixRecur v2 long reference from prior ledger entry:
  - `val_loss 4.63764717`, `val_bpb 2.74667588`, compressed `4224324`, total `4295101`, `step_avg 676.24ms`
- HelixRecur v3 long:
  - stop `600.276s`, `step 885`, `step_avg 678.28ms`
  - stop metric: `val_loss 2.4200`, `val_bpb 1.4333`
  - post-EMA `val_loss 2.6336`, `val_bpb 1.5598`
  - final roundtrip exact: `val_loss 4.74046053`, `val_bpb 2.80756774`
  - compressed `4212560`, total `4283448`
- Delta vs v2 long:
  - `val_loss`: `+0.10281336`
  - `val_bpb`: `+0.06089186`
  - `step_avg`: `+2.04ms`
  - compressed bytes: `-11764`
  - total bytes: `-11653`

### Judgment

- v3 improved the quick comparison slightly, with essentially unchanged step time.
- That gain did not hold in the longer pass, where v3 trailed v2 materially on quality.
- The implementation remains small and within the byte-efficiency story, but the result is not strong enough to replace v2.
- Stop stacking recurrence rescue changes in this session; `HelixRecur_v2` remains the active recurrence line.
