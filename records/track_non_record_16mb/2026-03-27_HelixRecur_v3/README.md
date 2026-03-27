## Non-record: HelixRecur v3 (v2 conditioning with learnable channel amplitudes)

HelixRecur v3 keeps the same recurrence hypothesis as v2: `6` shared blocks over `11` virtual passes on schedule `0,1,2,3,4,5,4,3,2,1,0`, with the same `11 x 4` virtual-depth conditioning table. The only change is to replace v2's fixed `0.05` modulation strength with tiny bounded learnable amplitudes, one per conditioned channel.

### What changed vs v2

- Base is `records/track_non_record_16mb/2026-03-26_HelixRecur_v2`.
- The same four conditioned concepts remain:
  - LN scale multiplier
  - attention output scale multiplier
  - MLP output scale multiplier
  - attention `q_gain` multiplier
- Added exactly `4` trainable parameters: one learnable amplitude logit per conditioned channel.
- Amplitudes are bounded and initialized to reproduce v2's effective strength:
  - `amp = 0.08 * sigmoid(logit)`
  - initial amplitude per channel = `0.05`
- This is still the same recurrence hypothesis, not a new architecture: the shared blocks, virtual schedule, donor local features, optimizer family, compression path, and eval path are unchanged.

### Parameter and byte effect

- v2 params: `15,187,040`
- v3 params: `15,187,044`
- Added params vs v2: `4`
- Code bytes: v2 `70,777`, v3 `70,888`

### Quick comparison

Solo `1xH100` comparison at `MAX_WALLCLOCK_SECONDS=180`, `EVAL_SEQ_LEN=64`, seed `1337`:

| Model | val_loss | val_bpb | compressed bytes | total bytes | step_avg |
|---|---:|---:|---:|---:|---:|
| HelixRecur v2 quick | `7.54165596` | `4.46659346` | `3,042,658` | `3,113,435` | `675.97ms` |
| HelixRecur v3 quick | `7.52554692` | `4.45705278` | `3,114,457` | `3,185,345` | `676.76ms` |

### Longer non-record pass

Solo `1xH100` run at `MAX_WALLCLOCK_SECONDS=600`, `EVAL_SEQ_LEN=64`, seed `1337`:

| Model | val_loss | val_bpb | compressed bytes | total bytes | train stop | step_avg |
|---|---:|---:|---:|---:|---:|---:|
| HelixRecur v2 long | `4.63764717` | `2.74667588` | `4,224,324` | `4,295,101` | `600.504s` | `676.24ms` |
| HelixRecur v3 long | `4.74046053` | `2.80756774` | `4,212,560` | `4,283,448` | `600.276s` | `678.28ms` |

### Assessment

- The quick comparison improved modestly over v2 with nearly unchanged step time.
- That quick gain did not hold in the longer pass: v3's longer trajectory was worse than v2 on quality.
- Byte efficiency remains strong, but the quality result does not justify replacing v2 with v3.

### Recommendation

Keep `HelixRecur_v2` as the active recurrence line. Treat v3 as a useful negative result showing that tiny learnable amplitude freedom can improve the short comparison without improving the longer trajectory.
