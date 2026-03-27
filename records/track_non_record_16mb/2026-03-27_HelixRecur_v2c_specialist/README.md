## Non-record: HelixRecur v2c specialist

HelixRecur v2c keeps the HelixRecur v2 shared recurrence core unchanged, then adds a tiny edge-depth specialist mechanism so the earliest and latest virtual passes can deviate slightly from the fully shared behavior.

### What changed vs v2

- Base is `records/track_non_record_16mb/2026-03-26_HelixRecur_v2`.
- Exact added parameter count vs v2: `4`.
- The same v2 `11 x 4` virtual-depth table is kept unchanged.
- Added a tiny edge-depth residual specialist scale for virtual depths `0`, `1`, `9`, and `10`.
- Each selected depth rescales only the shared-block residual delta with `1 + 0.05 * tanh(param)`.

### Why this is still the same hypothesis

This is still the same recurrence-family test as v2. The model still uses the same `6` shared blocks over the same `11`-pass schedule, with no routing, no new branch, and no broader relaxation of sharing. The only question is whether full sharing is slightly too aggressive at the outer virtual depths.

### Tournament results

Quick tournament attempt at `MAX_WALLCLOCK_SECONDS=180`, `EVAL_SEQ_LEN=64`, seed `1337`:

- comparable final `val_loss` / `val_bpb`: not produced
- artifact bytes: not produced
- runtime status: aborted after the direct log stopped progressing past the early training section while the process stayed live
- early visible step average before abort: 688.71ms

### Judgment

`v2c` is a no-go for this tournament. A tiny edge-depth residual specialist scale was added for virtual depths 0, 1, 9, and 10. In practice it violated the runtime discipline before producing a fair quick result, so it does not justify further GPU time in this session.
