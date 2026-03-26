# AGENTS.md

## Repository Purpose

- This repo is for the OpenAI Parameter Golf challenge: train the best language model that fits in a 16,000,000-byte artifact and optimize the official `val_bpb` / held-out loss on the fixed FineWeb validation set.
- Treat the root `README.md`, its FAQ and submission sections, the leaderboard table, and tracked folders under `records/` as the source of truth.

## Hard Rules

- The counted artifact budget is decimal `16,000,000` bytes total: counted code bytes plus compressed model bytes.
- Final leaderboard candidates must train reproducibly in under 10 minutes on `8xH100 SXM`.
- Evaluation has its own separate under-10-minute ceiling.
- No external downloads, network calls, or training-data access are allowed during evaluation.
- No validation leakage during training. Any adaptive or test-time method must remain strictly causal and may only use already-scored validation tokens.
- Do not touch the tokenizer or dataset unless explicitly authorized.
- Record pursuit is only justified when evidence suggests beating the current accepted SOTA by at least `0.005` nats with enough logs for `p < 0.01`.

## Where New Work Goes

- New model work lives under a new `records/...` folder, not in the root training script.
- Start novel work in `records/track_non_record_16mb/...` until there is real evidence that a record attempt is justified.
- Keep counted code compact. Prefer almost all counted code inside the submission folder's `train_gpt.py`.
- Keep one major hypothesis per revision. Do not bundle multiple new ideas before each one has an isolated ablation.

## Smoke And Baseline Commands

Run the feasible checks in this order before broader changes. If the environment cannot run one of them, record the exact blocker instead of guessing.

```bash
python -m py_compile train_gpt.py
```

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

## Experiment Reporting

- Every meaningful experiment must report `val_loss`, `val_bpb`, compressed model size, total artifact bytes, train wallclock, eval wallclock, and the key config.
- Always report metric delta vs the chosen baseline, plus size delta and runtime delta.
- Record the exact command, seed, commit, record-folder path, and outcome.
- Any record claim must include sufficient logs. In practice that usually means multiple seeds and enough evidence to support the required significance threshold.

## Checks After Changes

- Re-run the relevant syntax checks after code changes, for example:

```bash
python -m py_compile records/.../train_gpt.py
```

- Re-run the smallest feasible smoke command before moving to longer runs.
- Do not claim a baseline, donor, or ablation result unless the command actually ran in the current environment.

## Session Exit Criteria

- Leave the worktree clean relative to the work started in the session.
- Do not leave behind stray scratch files, abandoned record folders, or half-updated submission metadata.

## Session Note

- 2026-03-26 donor reproduction work stayed within the existing repo guardrails: root `train_gpt.py`, tokenizer logic, dataset logic, and `records/track_10min_16mb/2026-03-25_HelixGenomeLM` remained untouched.
