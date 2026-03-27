## Non-record: HelixRecur v1 (shared-depth recurrence on March 22 donor)

First isolated DNA-inspired hypothesis on top of the March 22 donor: replace the donor's 11 distinct transformer blocks with 6 shared recurrent blocks using schedule `0,1,2,3,4,5,4,3,2,1,0` while preserving donor embeddings, BigramHash, SmearGate, value embeddings, quantization/compression, and eval.

### Intended byte story

- Donor reproduced artifact: `16,140,640` total bytes, above the `16,000,000` cap.
- HelixRecur v1 reduces parameter count from `26,993,756` to `15,186,996`.
- Code bytes rise modestly from `67,603` to `69,074`.
- Quick runs show compressed model bytes dropping substantially, so the recurrence idea is strong on byte reuse.

### What changed

- Shared-depth recurrence only.
- Shared block schedule: `0,1,2,3,4,5,4,3,2,1,0`.
- Donor local features kept intact: tied embeddings, BigramHash, SmearGate, shared value embeddings, GPTQ-lite path, zstd compression path, donor eval path.
- No routing, no TTT, no tokenizer changes, no dataset changes, no optimizer redesign.

### Quick outcome

Abbreviated donor-vs-recurrence comparison at `MAX_WALLCLOCK_SECONDS=180`, `EVAL_SEQ_LEN=64`, `1xH100`, seed `1337`:

| Model | val_loss | val_bpb | compressed bytes | total bytes | train stop |
|---|---:|---:|---:|---:|---:|
| Donor quick | `7.55493163` | `4.47445606` | `5,019,273` | `5,086,876` | `180.427s` |
| HelixRecur v1 quick | `7.85509273` | `4.65222837` | `3,081,539` | `3,150,613` | `180.212s` |

### Assessment

- Byte profile improves materially.
- Wallclock is slightly better than donor on the same abbreviated setting.
- Quality is much worse: `val_bpb` degrades by `+0.17777231` on the quick comparison.
- This fails the session's kill criteria. HelixRecur v1 is not worth a longer pass in its current form.

### Recommendation

Keep the idea as a byte-efficiency reference only. Any next recurrence revision should preserve the byte advantage but add a minimal depth-conditioning mechanism or a more donor-faithful way to preserve late-depth specialization before spending more long-run budget.
