## Non-record: HelixGeneDelta v1 (shared-depth recurrence with gene-coded rank-2 MLP specialization)

Single-hypothesis follow-up to `HelixRecur_v2`: keep the same 6 shared recurrent blocks and virtual schedule `0,1,2,3,4,5,4,3,2,1,0`, but replace the tiny scalar-only depth conditioning with a compile-friendly gene-coded rank-2 specialization that touches only the MLP gate path.

### What changed vs v2

- Base is `records/track_non_record_16mb/2026-03-26_HelixRecur_v2` only.
- Shared-depth recurrence stays unchanged: `6` shared blocks over `11` virtual passes.
- Removed the scalar depth-conditioning table entirely.
- Added one explicit low-rank specialization path on the MLP gate activations only:
  - `11 x 2` virtual-depth gene-code table = `22` params
  - shared rank-2 MLP gate basis = `2 x 1536 = 3072` params
  - total added gene path = `3094` params
- The modulation remains bounded and tiny: `1 + 0.05 * tanh(gene @ basis)`.
- No routing, no TTT, no tokenizer/data changes, no optimizer redesign, and no donor/v3/tournament edits.

### Byte story

- HelixRecur v2 quick total bytes: `3,113,435`
- HelixGeneDelta v1 quick total bytes: `3,103,252`
- HelixGeneDelta v1 quick compressed bytes: `3,032,219`
- Code bytes: `71,033`
- Versus v2 quick, bytes improved slightly while quality regressed.

### Quick comparison

Solo `1xH100` comparison at `MAX_WALLCLOCK_SECONDS=180`, `EVAL_SEQ_LEN=64`, seed `1337`:

| Model | val_loss | val_bpb | compressed bytes | total bytes | step_avg |
|---|---:|---:|---:|---:|---:|
| HelixRecur v2 quick | `7.54165596` | `4.46659346` | `3,042,658` | `3,113,435` | `675.97ms` |
| HelixGeneDelta v1 quick | `7.55187900` | `4.47264812` | `3,032,219` | `3,103,252` | `675.48ms` |

### Smoke results

`1xH100` smoke checks with seed `1337`:

| Run | stop | stop metric | post-EMA | final roundtrip | compressed bytes | total bytes |
|---|---:|---|---|---|---:|---:|
| Train smoke | `45.447s`, step `67` | `val_loss 6.0832`, `val_bpb 3.6028` | `val_loss 6.0700`, `val_bpb 3.5950` | `val_loss 6.09674746`, `val_bpb 3.61083726` | `2,802,415` | `2,873,448` |
| Eval smoke | `1.377s`, step `2` | `val_loss 8.7284`, `val_bpb 5.1695` | `val_loss 6.9062`, `val_bpb 4.0902` | `val_loss 6.91002561`, `val_bpb 4.09250639` | `2,642,146` | `2,713,179` |

### Assessment

- The gene-coded rank-2 MLP specialization stayed compile-friendly and runtime-neutral.
- It did not beat the v2 champion on the quick proxy: `+0.00605466 val_bpb` worse than v2.
- The small byte win does not justify the quality regression.
- This direction is a no-go from the current recurrence line unless a future grant explicitly wants byte-saving negative results or a broader MLP-only specialization study.
