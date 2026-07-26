# Floor sweep: Qwen3-0.6B

The top-K cache stores only the teacher's K most likely tokens per position and
models everything below them as a uniform tail. That approximation has a cost.
Even a student identical to its teacher scores a small nonzero KL divergence,
and that value is the measurement floor. `floor-sweep` measures it directly by
scoring a model against its own reconstruction at several K values.

## The measurement

```bash
mlx-kld floor-sweep Qwen/Qwen3-0.6B \
    --k-values 512,2048,8192,32768,65536,131072
```

Default protocol (512 sequences x 512 tokens of wikitext-103, seed 123, scoring
the second half of each sequence), vocabulary 151,936, 131,072 tokens scored.
About two minutes on an M-series laptop.

| K | mean floor (nats) | worst position (nats) |
|---:|---:|---:|
| 512 | 0.125820 | 2.288454 |
| 2,048 | 0.035892 | 1.038573 |
| 8,192 | 0.006944 | 0.299802 |
| **32,768** | **0.002974** | 0.053274 |
| 65,536 | 0.003374 | 0.066123 |
| 131,072 | 0.005570 | 0.109890 |

## The floor is U-shaped in K

The intuitive reading is that caching more of the distribution can only help.
It does not. The floor bottoms out near K=32,768 and climbs again above it.

A second sweep at a different seed and sample count agrees closely, so the
shape is not sampling noise:

```bash
mlx-kld floor-sweep Qwen/Qwen3-0.6B --num-samples 64 --seed 777 \
    --k-values 8192,16384,32768,49152,65536,98304,131072
```

| K | mean floor (nats) |
|---:|---:|
| 8,192 | 0.006908 |
| 16,384 | 0.003653 |
| **32,768** | **0.002975** |
| 49,152 | 0.003103 |
| 65,536 | 0.003374 |
| 98,304 | 0.004153 |
| 131,072 | 0.005553 |

Two errors move in opposite directions as K grows.

1. The uniform-tail approximation gets better. Below the minimum this dominates,
   which is why the floor drops steeply from K=512 to K=8,192.
2. Every cached log-probability is stored as bfloat16, so a larger K rounds more
   numbers. Above the minimum this dominates.

There is also a third effect. The closed-form KL expansion carries a
self-entropy error and a cross-term error of opposite sign that partially
cancel, and the cancellation is least favourable when the tail is very small.
See the comment in `kld_math.kld_from_topk`.

## What to take from this

Raising `--top-k` past the minimum costs disk and accuracy at the same time. A
default-protocol entry is about 6 bytes per position per slot, so K=131,072
writes roughly 206 GB against 51.5 GB at K=32,768, and measures slightly worse.

The minimum's location depends on the vocabulary and on how peaked the model is,
so it is not a universal constant. Measure it for your own teacher before
moving off the default. Its magnitude varies far more than its location: this
0.6B model floors at 0.0030 nats where a 27B floors at 0.0018 nats under the
same protocol, both at K=32,768.
