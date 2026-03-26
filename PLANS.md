# PLANS.md

## Objective

- Establish a disciplined baseline-first path for a DNA-biomimetic submission without changing the tokenizer, dataset, or root training code.
- Optimize official `val_bpb` on the fixed FineWeb validation set while preserving reproducibility, size discipline, and the 10-minute `8xH100 SXM` train/eval limits.
- Default first implementation path: create a new folder under `records/track_non_record_16mb/` and stage ablations there until record evidence exists.

## Current official constraints

- Source of truth is the root `README.md`, its FAQ/submission sections, and the tracked `records/` folders.
- Counted artifact budget is `16,000,000` bytes total: counted code plus compressed model.
- Final leaderboard candidates must train reproducibly in under 10 minutes on `8xH100 SXM`.
- Evaluation has a separate under-10-minute ceiling.
- No external downloads, network calls, or training-data access are allowed during evaluation.
- No validation leakage is allowed during training. Any adaptive or test-time method must be strictly causal and may only use already-scored validation tokens.
- Official submissions are PRs that add a new folder under the correct `records` track and include `README.md`, `submission.json`, train log(s), `train_gpt.py`, and any needed dependencies/setup notes.

## Current accepted SOTA and date

- Current accepted SOTA in this repo snapshot is `1.1194 val_bpb`, `LeakyReLU^2 + Legal Score-First TTT + Parallel Muon`, dated `2026-03-23`.
- The FAQ says record acceptance is chronological by PR creation time and leaderboard updates may lag, so future targeting must consider the true active PR target, not just the visible table.
- For this snapshot, the accepted record gate is still `<= 1.1144` with enough logs to show `p < 0.01`.
- The root README states the baseline should land around `~1.2 val_bpb` under 16MB. The tracked naive baseline reports `1.2243657` post-quant roundtrip, `1.2172` pre-quant, `15,863,489` total bytes, and `600038ms` train time.

## Baseline reproduction plan

1. Download the published `sp1024` dataset and tokenizer:

   ```bash
   python3 data/cached_challenge_fineweb.py --variant sp1024
   ```

2. Run the smallest meaningful smoke baseline on a `1xH100` machine:

   ```bash
   RUN_ID=baseline_sp1024 \
   DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
   TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
   VOCAB_SIZE=1024 \
   torchrun --standalone --nproc_per_node=1 train_gpt.py
   ```

3. Run the official `8xH100` baseline command used by the tracked naive baseline:

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

4. Accept the baseline as reproduced only after logging exact command, runtime, `val_loss`, `val_bpb`, compressed model size, total bytes, and any environment notes.
5. If the local environment cannot run the baseline, record the precise blockers and keep the tracked naive baseline metrics labeled as repo-documented reference only.

## Primary v0 architecture for future implementation

- Base donor for all first-pass work: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`.
- Preserve the donor stack for Revision 1: tied embeddings, BigramHash, SmearGate, shared value embedding, Partial RoPE, LN scale, XSA-on-late-layers, EMA, tight SWA, GPTQ-lite, late QAT, and the existing compression path.
- First real implementation revision changes only one core hypothesis: replace the donor's unique-depth stack with a fixed shared-depth recurrent core using `6` shared blocks over `11` virtual passes on schedule `0,1,2,3,4,5,4,3,2,1,0`.
- Revision 1 forbids routing, tokenizer changes, dataset changes, and evaluation changes. The comparison target is the donor stack with the same tokenizer family and compression path.
- Intended end-state `v0` remains simple: recurrent/shared-depth core, tied embeddings, a compact motif/codon-style hashed local feature branch, and a tiny regulatory lane. No heavy promoter/router system belongs in `v0`.
- Reusable ideas from tracked winners that are compatible with this direction:
  - tied embeddings are byte-efficient but quantization-sensitive;
  - BigramHash + SmearGate is the strongest cheap local motif donor on the fixed tokenizer;
  - EMA + tight SWA + GPTQ-lite / late QAT is the strongest same-tokenizer compression donor before TTT;
  - shared value embedding with tiny per-layer scales is a good template for a small regulatory lane;
  - legal score-first TTT is a deferred eval-only lever, not a `v0` dependency.

## Fallback architecture

- Use the same March 22 donor and compression stack.
- Implement recurrent shared blocks plus motif features only.
- Keep routing disabled.
- Skip the regulatory lane unless the recurrence-only and motif-only ablations both show clean gains.
- Prioritize simplicity, stability, step speed, and byte efficiency over biological decoration.

## Artifact budget table

| Bucket | Target bytes | Hard stop bytes | Notes |
|---|---:|---:|---|
| Counted code | 80,000 | 120,000 | Keep counted code compact and concentrated in the submission `train_gpt.py`. |
| Compressed model | 15,820,000 | 15,880,000 | The exact ceiling falls if counted code grows. |
| Total artifact | 15,900,000 | 16,000,000 | Leave slack for measurement variance and final packaging. |

## Runtime budget table

| Phase | Target | Hard stop | Notes |
|---|---:|---:|---|
| `1xH100` smoke | 3 minutes | 5 minutes | Fast blocker detection only. |
| `8xH100` candidate train | 540 seconds | 600 seconds | Leave headroom for noisy cluster timing. |
| Standard eval | 240 seconds | 600 seconds | Keep base eval comfortably under the cap before trying anything adaptive. |
| Optional legal TTT eval-only branch | 540 seconds total eval | 600 seconds | Only attempt after the base model already has strong evidence. |

## Ablation order

1. Baseline reproduction.
2. Donor reproduction.
3. Shared-depth recurrence only.
4. Motif/codon hash only.
5. Tiny regulatory lane only.
6. Compression retune only.
7. Optional legal TTT eval-only branch last.

## Record vs non-record decision gates

- Default early DNA work to `records/track_non_record_16mb/`.
- Only switch into record pursuit if compliant `8xH100` evidence suggests a real path to `<= 1.1144` and enough logs can support `p < 0.01`.
- If a result is novel and compliant but not record-worthy, keep the non-record path. Do not force extra complexity just to chase the leaderboard.
- Reject or simplify any revision that worsens `val_bpb` per counted byte, breaks runtime headroom, or adds complexity without a measurable gain.

## Submission packaging checklist

- Add only a new folder under the correct `records` track.
- Include `README.md`, `submission.json`, train log(s), `train_gpt.py`, and any required dependencies/setup notes.
- Ensure the script compiles and runs from within the record folder.
- Report real measured `val_loss`, `val_bpb`, bytes, wallclock, and config from actual runs, not estimates.
- Keep tokenizer and dataset unchanged unless explicitly authorized and validated with extreme care.

## Risks and open questions

- No accepted tracked recurrence donor exists in this snapshot. The only repo evidence is negative or disabled: `depth recurrence` was called too slow in `2026-03-18_FP16Embed_WD3600`, and recurrence flags are `0` in the `2026-03-24` ternary record. Shared-depth recurrence is therefore a fresh hypothesis.
- Recurrence may save bytes but still lose on wallclock if repeated passes cost too many steps.
- A codon-style hash branch may duplicate the signal already covered by BigramHash + SmearGate unless it is cheaper or measurably stronger per counted byte.
- A tiny regulatory lane can become decorative quickly. Keep it small and make it earn its bytes.
- Leaderboard chronology and PR lag can move the record target even if the repo table has not updated yet.
- Local implementation work is confirmed blocked as of `2026-03-26`: `Python 3.14.3` with no importable `torch`/`numpy`/`sentencepiece`/`tqdm`, no visible `nvidia-smi`, and no published `sp1024` dataset/tokenizer assets in this checkout.
- Defaults chosen unless later evidence changes them: donor is the March 22 record, the initial track is non-record, the first shared-depth schedule is `0,1,2,3,4,5,4,3,2,1,0`, and routing stays disabled in `v0`.

## Not Yet

- full promoter/router system
- tokenizer changes
- dataset changes
- risky test-time training variants
- per-token routing unless later evidence justifies it
