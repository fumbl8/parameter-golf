## Non-record: HelixRecur v2a rank8

HelixRecur v2a keeps the HelixRecur v2 recurrence stack exactly intact: the same 6 shared blocks, the same virtual schedule `0,1,2,3,4,5,4,3,2,1,0`, and the same donor local-feature, optimizer, compression, and eval path. The only change is to widen the virtual-depth conditioning table from `11 x 4` to `11 x 8` while still emitting the same four scalar modulations.

### What changed vs v2

- Base is `records/track_non_record_16mb/2026-03-26_HelixRecur_v2`.
- The virtual-depth conditioning table is widened from `11 x 4` to `11 x 8`.
- Exact added parameter count vs v2: `44`.
- The wider table is still compile-friendly: each conditioned scalar reads the mean of a dedicated 2-channel slice.
- Conditioned outputs remain the same v2 targets: LN scale multiplier, attention output scale multiplier, MLP output scale multiplier, and attention `q_gain` multiplier.

### Why this is still the same hypothesis

This is still the same recurrence-family test as v2. There are still `6` shared blocks reused across `11` virtual passes, with no routing, no new branch, and no optimizer, tokenizer, dataset, compression, or eval redesign. The only question is whether v2's depth-conditioning table was capacity-starved.

### Tournament results

Quick proxy at `MAX_WALLCLOCK_SECONDS=180`, `EVAL_SEQ_LEN=64`, seed `1337`:

- `val_loss 7.50140432`
- `val_bpb 4.44275417`
- `step_avg 679.19ms`
- compressed bytes `3085937`
- total bytes `3156750`
- delta vs v2 quick: `val_bpb -0.02383929`, total bytes `+432315`, step time `+3.22ms`

Longer pass at `MAX_WALLCLOCK_SECONDS=600`:

- `val_loss 4.71987996`
- `val_bpb 2.79537877`
- `step_avg 678.98ms`
- compressed bytes `4212788`
- total bytes `4283601`
- delta vs v2 long: `val_bpb +0.04870289`, total bytes `-11500`, step time `+2.74ms`

### Judgment

`v2a` won the quick proxy but lost the longer pass against the v2 long anchor. Keep `HelixRecur_v2` as champion; `v2a` is useful as evidence that extra conditioning capacity can improve the short proxy without improving the longer trajectory.
