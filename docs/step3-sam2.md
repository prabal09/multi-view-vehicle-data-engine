# Step 3 — SAM2: from a box to a *tracked* mask

Theory notes for the segmentation-and-tracking stage. SAM2 takes the single box
Grounding DINO (Step 2) found on one frame and turns it into (a) a precise
pixel mask and (b) that same mask **propagated across every following frame** —
i.e. it tracks the vehicle. This doc goes one level below the "how to run" and
opens up *why* SAM2 can remember an object without ever being re-prompted.

Everything here is anchored to the three SAM2 calls in
[sanity/sam2_tracking_sanity.py](../sanity/sam2_tracking_sanity.py):
`init_state`, `add_new_points_or_box`, `propagate_in_video`.

---

## A. The core idea: an image segmenter wrapped in a memory loop

The original **SAM** (v1) is an *image* model: image + prompt (point/box) → mask.
It has **no notion of "the same object later."** Run it frame-by-frame on
`c0_5.mp4` and each frame is segmented from scratch — the car's mask flickers,
drifts, and loses identity the moment it changes pose or ducks behind a pole.

**Tracking is fundamentally a memory problem.** To segment *this* car in frame
`t`, the model must remember what the car looked like in frames `0…t-1`. That is
the entire reason SAM2 exists:

> **SAM2 = SAM's image segmenter + a read-then-write memory loop with bounded
> state.** Every frame *reads* the accumulated memory of the target to condition
> its perception, produces a mask, and *writes* that mask back as new memory.

In the sanity file this is stark: the **only** external information SAM2 ever
receives is the single `add_new_points_or_box` on `ann_idx=0`. Every one of the
other 119 frames is segmented purely from **memory**, not from a new box.

## B. The name / lineage, decoded

- **SAM** = *Segment Anything Model* — the promptable image segmenter (image
  encoder → prompt encoder → mask decoder), trained to turn a point/box/mask
  prompt into a mask for *anything*, open-vocabulary at the mask level.
- **SAM 2** = SAM extended to **video** by adding a **streaming memory**. Same
  three original blocks, plus three new ones (memory attention, memory encoder,
  memory bank) that form the loop.
- **"Streaming"** is the key word: it processes video like a stream, one frame
  in at a time, carrying a **fixed-size** memory forward. Cost per frame is
  *constant* regardless of clip length — the design target was real-time on
  arbitrarily long video.
- **Hiera** = the *Hierarchical* Vision Transformer used as SAM2's image
  encoder — a plain, fast multi-scale ViT (this is the block you swap when you
  choose `hiera_large` vs `hiera_tiny` in the sanity file's comments).

## C. The architecture, component by component

Six blocks. The top three are inherited from SAM1; the bottom three are the
memory subsystem that makes it video.

```text
                    ┌──────────────────────────────────────────────┐
                    │                 MEMORY BANK                   │
                    │  spatial memories (FIFO)  +  object pointers  │
                    └──────▲────────────────────────────┬──────────┘
                    read   │ (keys/values)              │ write
                           │                            │
  frame_t ─► Image ─────► Memory ─────► Mask ─────► mask_t ─► Memory
             Encoder      Attention     Decoder               Encoder
             (Hiera)         ▲            │
                             │            └─► occlusion score (is it even here?)
                         prompt encoder
                    (only on prompted frames)
```

1. **Image encoder — Hiera (runs once per frame).** Streams the video frame by
   frame and produces plain, *unconditioned* multi-scale visual features. It
   does **not** know about the target yet. This is the heavy compute and the
   part that scales with model size.
2. **Prompt encoder (SAM1, unchanged).** Encodes a point/box/mask prompt into
   tokens. In our pipeline it fires **once** — the Grounding DINO box on frame 0.
3. **Memory attention — the heart of "remembering."** Conditions the *current*
   frame's features on the past (Section E). Its output is a feature map that
   already "knows" where the tracked car probably is. This is what **replaces
   re-prompting** on every frame.
4. **Mask decoder (SAM1 + extras).** Emits the mask + an IoU/quality score, and
   critically a new **occlusion score** ("is the object even present?"). Still
   multi-mask-capable to resolve ambiguity.
5. **Memory encoder — writes the new memory.** Compresses `(frame features +
   the mask just predicted)` into a compact memory embedding and pushes it into
   the bank (Section F).
6. **Memory bank — the stored state** (Section D).

Blocks 3→4→5 are the **loop**: read memory, decode a mask, write memory. That
loop running once per frame *is* `propagate_in_video`.

## D. The memory bank — what is actually stored

The single most important design insight is that memory is **two different
things**, because they fail differently and together they're robust:

| Kind | What it is (shape, typical defaults) | Preserves | Strength |
|---|---|---|---|
| **Spatial memory** | A FIFO of the last *N* frames' feature maps fused with their masks. Each ≈ `64×64×64` (a stride-16 grid, projected to 64 channels) | *Where* & appearance/shape — pixel-level detail | Short-term boundary precision |
| **Object pointers** | A small list of `256`-d vectors distilled from the mask-decoder output tokens, one per frame kept | *What* the object is — semantic identity | Survives occlusion & big pose change |
| **Prompt memory** | The memories from frames you explicitly prompted (frame 0 here) | Human ground truth | Anchors the track; weighted heavily |

Why two types instead of one:

- **Spatial memory** gives crisp masks but **degrades under occlusion** — if the
  car hides behind a pole for 10 frames, the recent spatial memories are of the
  *pole*, not the car.
- **Object pointers** are compact semantic anchors that **survive** that
  occlusion. When the car reappears, the pointer says "same object" even though
  recent spatial memory was polluted. This split is *why* SAM2 re-acquires an
  object after occlusion instead of latching onto whatever's in front.

**Bounded state (the "streaming" promise).** Spatial memory is a **sliding
window** of the last *N* frames (a common default is 7 slots: the current frame
plus ~6 recent), *plus* the prompted frames which are kept regardless of age.
Object pointers are likewise capped (≈16). Because both are fixed-size, the
memory-attention cost per frame is **constant**, not growing with the 120-frame
(or 12,000-frame) clip. That is the whole point of calling it *streaming
memory*.

## E. Deep dive — the tensor shapes flowing through memory attention

This is the mechanism that "remembers." It is a small transformer stack (a
common default is **L = 4** blocks), and each block does, in order:

```text
  current-frame tokens ──► self-attention ──► cross-attention ──► FFN ──►
        (queries Q)                             ▲   ▲
                                    keys K ─────┘   └───── values V
                                    (both come from the MEMORY BANK)
```

Concrete shapes, for a `1024×1024` input and the stride-16 feature map SAM2
conditions on (numbers are the typical defaults; exact values are
config-dependent):

- **Queries `Q`** — the current frame's tokens from Hiera:
  `64 × 64 = 4096` spatial tokens, each `C = 256`-d → `Q ∈ ℝ^{4096 × 256}`.
  *Self-attention* first lets the frame reason about itself; then those tokens
  become the queries into memory.
- **Keys/Values `K, V`** — the concatenation of everything in the bank:
  - each of the `N` spatial memories flattened `64×64 = 4096` tokens →
    up to `~7 × 4096 ≈ 28,672` memory tokens,
  - **plus** the object-pointer vectors (≈`16 × 256`),
  - all projected to the attention width and concatenated along the token axis.
- **Cross-attention** = for each of the 4096 current-frame locations, attend
  over those ~28k+ memory tokens and pull in what's relevant. Output stays
  `4096 × 256` — same shape as `Q`, but now **memory-conditioned**.

Two details that matter:

- **Positional encoding is spatio-*temporal*.** Memory tokens carry not just
  their `(x, y)` position but a **temporal embedding** encoding *how many frames
  ago* the memory is. This is how the model knows "this memory is recent / this
  one is the original prompt" — order is information, and without it the bank
  would be an unordered bag.
- **Why condition the *features*, not post-process the masks?** Because tracking
  errors compound. Injecting memory *before* the decoder biases the model's very
  *perception* of the frame toward the target — the past shapes what it "sees,"
  not merely what it outputs. Independent per-frame masks + smoothing can't
  recover a frame that was perceived wrong.

## F. Deep dive — occlusion scoring and how it gates the memory write

Video breaks a hidden SAM1 assumption: **the object might not be in the frame at
all** (it drove off-screen, or is fully behind the pole). If SAM2 blindly wrote
a memory every frame, an occluded frame would write a memory *of the occluder*,
poison the bank, and the track would jump onto the pole.

So the mask decoder emits, alongside the mask, an **occlusion / "object
present" score** (an extra learned output head):

```text
memory attention ─► mask decoder ─┬─► mask logits  (thresholded: logits > 0.0)
                                  ├─► IoU / quality score
                                  └─► occlusion score  ── low ──► "not here"
```

That score **gates the loop**:

- **Present (high score):** decode the mask, and the **memory encoder writes**
  this frame's `(features + mask)` into the spatial FIFO and appends an object
  pointer. Normal case.
- **Absent (low score):** emit an empty/low-confidence mask, and **do not write
  a polluting spatial memory** (the frame contributes at most a "was occluded"
  signal, not a confident mask of the wrong thing). The privileged **object
  pointers stay intact**, so when the car re-emerges, memory attention re-locks
  onto it.

This gate is the concrete reason SAM2 tolerates occlusion. In the sanity file it
also explains the printed `tracked mask present in {covered}/{n} frames`: frames
where the vehicle is genuinely gone/occluded can legitimately show **no mask**,
and that is the occlusion head doing its job — not a bug.

**Why prompted-frame memory is privileged.** Memories generated from the model's
*own* predicted masks can drift (self-reinforcing error — a slightly-off mask
teaches a slightly-off memory, forever). The frame-0 box is the only
human-verified truth in the whole sequence, so its memory is kept and weighted
heavily, anchoring the track to a trusted origin. This is also why adding a
*correction* point mid-clip (SAM2 allows `add_new_points_or_box` on any frame)
snaps the track back hard — that new prompt memory dominates the bank.

## G. Deep dive — multiple objects: how `obj_id` streams stay separate

In the sanity file we seed exactly one object:

```python
predictor.add_new_points_or_box(inference_state=state,
                                frame_idx=args.ann_idx, obj_id=1, box=...)
```

`obj_id` is the **key that partitions the memory bank**. Each object gets its
**own independent memory stream** — its own spatial FIFO, its own object
pointers, its own occlusion state. Conceptually the bank is
`{obj_id → memory}`, and memory attention for object *k* attends **only** to
object *k*'s memories. Two cars can cross paths and their tracks don't merge,
because their banks never share tokens.

The efficiency trick: the **Hiera image features are computed once per frame and
shared** across all objects (that heavy compute is object-independent). Only the
lightweight memory-attention + decoder + memory-encoder passes run per object.
That is why `propagate_in_video` yields `_obj_ids` alongside the masks — it is
handing back a mask **per tracked object** for that frame:

```python
for out_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
    #          ^^^^^^^^  one entry per obj_id; mask_logits[i] is object i's mask
```

To track every vehicle in the scene you would call `add_new_points_or_box` once
per detected box with a distinct `obj_id` (`1, 2, 3, …`) **before** propagating;
one propagation pass then advances all their streams together.

## H. Mapping it back to the three calls

```python
state = predictor.init_state(video_path=frames_dir)
```
Allocates the **inference state**: an empty memory bank, and prepares the
**Hiera image encoder** over the frames folder. Nothing is tracked yet — "load
the film into the projector." *This is why the video predictor wants a **frames
folder**, not a raw mp4*: it indexes frames for random-access memory reads.

```python
predictor.add_new_points_or_box(..., frame_idx=0, obj_id=1, box=np.array(box))
```
Runs **prompt encoder → mask decoder** on frame 0 to get the seed mask, then the
**memory encoder writes the first (privileged) memory** under `obj_id=1`. The
only external information the tracker ever gets.

```python
for out_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
    m = (mask_logits[0] > 0.0).cpu().numpy()
```
The **memory loop itself**, once per frame in order: Hiera features → **memory
attention reads the bank** → mask decoder emits `mask_logits` (thresholded at
`> 0.0`) + occlusion score → **memory encoder writes** this frame's memory back.
That write is why frame 50 stays locked on a car it hasn't been prompted about
in 50 frames — every frame both reads from *and* contributes to the accumulated
memory of that one vehicle.

## I. The practical shape (for implementation)

- **Checkpoints (the Hiera size knob):** `hiera_tiny` → `hiera_small` →
  `hiera_base_plus` → `hiera_large`. Bigger = better masks under motion/occlusion
  but more VRAM and slower. The sanity file defaults to **large** and documents
  the **tiny** fallback for tight VRAM — start large for quality on the clear
  roadside vehicles, drop to tiny only if the T4 complains.
- **Config must match the checkpoint:** `sam2.1_hiera_l.yaml` ↔
  `sam2.1_hiera_large.pt`, `..._t.yaml` ↔ `..._tiny.pt`. Mismatched config/weights
  is the most common load error.
- **Image vs video predictor:** we use `build_sam2_video_predictor` (the one with
  the memory loop). There is also an image-only predictor (no memory) — wrong
  tool for tracking.
- **Precision:** run under `torch.autocast(..., bfloat16)` on CUDA (as the sanity
  file does) — meaningful speed/VRAM win with negligible mask-quality cost.
- **First run:** seed with the Grounding DINO box on frame 0 of `c0_5`, propagate
  over a short window, and *watch the overlay* (green mask + yellow seed box)
  before scaling up. Prove the lock-on on one clip, then grow.

## J. Where it sits in the pipeline

```text
GDINO box (Step 2) ─► SAM2 seed ─► memory loop over frames ─► per-frame masks + track
   "what & where"      (frame 0)     read↔write memory          "exact pixels, followed over time"
```

Grounding DINO answers *what & where* on one frame; SAM2 answers *exact pixels &
identity over time*. Its per-object masks + tracks are the raw material the ETL
stage (Step 4+) turns into the results the DB records.

---

## Open threads to go deeper on later

- Hiera internals — how the hierarchical ViT builds its multi-scale features and
  why it's faster than a plain ViT backbone.
- The memory encoder's fusion op — exactly *how* mask + frame features are
  combined and downsampled to the `64`-d memory.
- Training: how SAM2 was trained on video with simulated interactive prompts
  (the data engine), and how the occlusion head is supervised.
- Failure modes on *our* footage — thin poles, mirrored/glass vehicles, two
  identical cars crossing — and when a mid-clip correction prompt is worth it.
- Turning per-frame masks into a stable per-vehicle record for the DB (identity,
  first/last seen, mask→attributes).
```
