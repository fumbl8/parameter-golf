## Non-record: HelixRecur v2 (shared-depth recurrence with tiny virtual-depth conditioning)

Minimal follow-up to HelixRecur v1: keep the same 6 shared recurrent blocks and schedule `0,1,2,3,4,5,4,3,2,1,0`, then add a tiny virtual-depth conditioning table so repeated passes can recover a small amount of depth-specific behavior without giving up the recurrence byte win.

### What changed vs v1

- Base is `records/track_non_record_16mb/2026-03-26_HelixRecur_v1`, not the donor.
- Shared-depth recurrence stays unchanged: `6` shared blocks over `11` virtual passes.
- Added exactly `44` new trainable parameters: an `11 x 4` virtual-depth conditioning table.
- The conditioning only modulates existing donor/v1 scalars:
  - LN scale multiplier
  - attention output scale multiplier
  - MLP output scale multiplier
  - attention `q_gain` multiplier
- The modulation range is intentionally tiny: each multiplier is `1 + 0.05 * tanh(param)`.
- Donor embeddings, BigramHash, SmearGate, value embeddings, optimizer family, quantization/compression path, and eval path stay intact.

### Byte story

- v1 quick total bytes: `3,150,613`
- v2 quick total bytes: `3,113,435`
- v2 quick compressed bytes: `3,042,658`
- Code bytes rise from `69,074` to `70,777`, but compressed model bytes still improve slightly and the total artifact shrinks.

### Quick comparison

Abbreviated `1xH100` comparison at `MAX_WALLCLOCK_SECONDS=180`, `EVAL_SEQ_LEN=64`, seed `1337`:

| Model | val_loss | val_bpb | compressed bytes | total bytes | step_avg |
|---|---:|---:|---:|---:|---:|
| Donor quick | `7.55493163` | `4.47445606` | `5,019,273` | `5,086,876` | `668.25ms` |
| HelixRecur v1 quick | `7.85509273` | `4.65222837` | `3,081,539` | `3,150,613` | `655.32ms` |
| HelixRecur v2 quick | `7.54165596` | `4.46659346` | `3,042,658` | `3,113,435` | `675.97ms` |

### Longer non-record pass

Solo `1xH100` run at `MAX_WALLCLOCK_SECONDS=600`, `EVAL_SEQ_LEN=64`, seed `1337`:

| Model | val_loss | val_bpb | compressed bytes | total bytes | train stop | step_avg |
|---|---:|---:|---:|---:|---:|---:|
| HelixRecur v2 long | `4.63764717` | `2.74667588` | `4,224,324` | `4,295,101` | `600.504s` | `676.24ms` |

### Assessment

- v2 materially recovers quality relative to v1 on the same quick setting.
- The byte profile remains clearly favorable versus both donor and v1.
- Runtime stays operationally simple and close to v1 once measured in a solo run.
- This makes v2 worth keeping as the first recurrence variant with an honest quality/byte tradeoff.

### Next step

Keep the recurrence structure and byte discipline, but if iterating again, move the same depth-conditioning idea into a more compile-friendly and explicit formulation before trying any broader novelty.
