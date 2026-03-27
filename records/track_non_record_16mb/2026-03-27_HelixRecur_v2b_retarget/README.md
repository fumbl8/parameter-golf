## Non-record: HelixRecur v2b retarget

HelixRecur v2b keeps the same HelixRecur v2 recurrence stack and the same tiny `11 x 4` conditioning budget, but retargets that budget to an attention-focused control set instead of splitting it across both attention and MLP pathways.

### What changed vs v2

- Base is `records/track_non_record_16mb/2026-03-26_HelixRecur_v2`.
- Exact added parameter count vs v2: `0`.
- The same `11 x 4` virtual-depth table is kept.
- Retargeting choice: attention-focused only.
- The four table channels now drive only attention-side controls: a grouped LN-scale multiplier, attention output scale multiplier, and attention `q_gain` multiplier. The MLP scale path is left unconditioned at `1.0`.

### Why this is still the same hypothesis

This is still the same recurrence-family test as v2. The shared blocks, virtual schedule, donor local features, optimizer family, compression path, and eval path are unchanged. The only question is whether v2's depth signal is more useful when concentrated on attention-side controls rather than spread across both attention and MLP scalar sites.

### Tournament results

Quick tournament attempt at `MAX_WALLCLOCK_SECONDS=180`, `EVAL_SEQ_LEN=64`, seed `1337`:

- comparable final `val_loss` / `val_bpb`: not produced
- artifact bytes: not produced
- runtime status: aborted after the direct log stopped progressing past the early training section while the process stayed live
- early visible step average before abort: 679.51ms

### Judgment

`v2b` is a no-go for this tournament. The same 11x4 conditioning budget was retargeted to attention-side controls only. In practice it violated the runtime discipline before producing a fair quick result, so it does not justify further GPU time in this session.
