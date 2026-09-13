# Step 2 — Grounding DINO: detecting vehicles *by words*

Theory notes for the detection stage of the pipeline. Grounding DINO is the
prompt-driven front door: given a sampled video frame and a text prompt, it
returns bounding boxes for the things the words describe. Those boxes become the
seeds SAM2 (Step 3) turns into precise masks and temporal tracks.

---

## A. The core idea: open-vocabulary vs a closed-set detector

The project's older `MRCNN.pt` (Mask R-CNN) is a **closed-set** detector: it was
trained on a fixed list of classes, and its output layer can only emit *those*
classes. A new category means collecting labels and retraining — the vocabulary
is baked into the weights.

**Grounding DINO is open-vocabulary.** At *inference* you hand it a **text
prompt**, and it detects whatever the words describe:

```text
prompt: "car . suv . pickup truck . van . bus"   ->  boxes for those things
prompt: "traffic cone . pothole"                 ->  boxes for those instead
```

No retraining — change the words, change what it finds. For this project that is
exactly right: one prompt covers all vehicle types, and the granularity can be
dialed (generic `"vehicle"` for max recall, or specific types to help populate
the "Vehicle Type" column) just by editing a string.

## B. The name, decoded — it tells you the architecture

- **DINO** here is a **DETR-family Transformer detector** — the lineage that
  treats detection as *set prediction* (predict a fixed set of boxes directly,
  no anchor boxes, no non-max-suppression). Strong closed-set detector on its own.
- **Grounding** = **phrase grounding**: linking *phrases in a sentence* to
  *regions in an image*.
- **Grounding DINO** = take that strong DINO detector and **fuse a language model
  into it** so every detection is *conditioned on the text*. It is the successor
  to GLIP, with a better detector backbone.

So the whole model is "a Transformer detector taught to align image regions with
words."

## C. How it works inside

Six stages, two input streams (image + text) fused early:

```text
image --> [Swin Transformer] --> image features ┐
                                                 ├─> [Feature Enhancer] --> [Language-guided --> [Cross-Modality --> boxes +
text  --> [BERT] -------------> text features   ┘    (cross-attention:       Query Selection]     Decoder]          per-word
                                                      image<->text fuse)                                            match scores
```

1. **Image backbone (Swin Transformer)** -> multi-scale visual features.
2. **Text backbone (BERT)** -> token features for the prompt.
3. **Feature Enhancer — the key to "open-vocab."** Stacked cross-attention where
   **image features attend to the text and text attends to the image.** After
   this, the image representation is *text-aware* — regions that look like "a
   pickup truck" get pulled toward the "pickup truck" tokens. This early fusion
   is what makes novel prompts work.
4. **Language-guided query selection** — pick the image locations most similar to
   the prompt to seed candidate detections (the "queries"). Language steers
   *where* to look.
5. **Cross-modality decoder** — each query iteratively refines into a box while
   attending to both image and text, and emits an **alignment score against every
   text token**.
6. **Output** — N boxes, each with (a) coordinates and (b) a per-token similarity
   vector saying *which words it matches and how strongly*.

Mental model: **detection score isn't "class #12 = 0.9"; it's "this box aligns
with the tokens *pickup truck* at 0.9."** That contrastive
image-region-to-text-token alignment *is* the open-vocabulary mechanism.

## D. How the prompt actually behaves

- Categories are lowercase and **separated by `.`**: `"car . suv . truck"`. The
  dots delimit phrases so their tokens don't bleed together via attention masks.
- Two thresholds control the output:
  - **`box_threshold`** — keep only boxes whose confidence clears it (typical ~0.35).
  - **`text_threshold`** — how strongly tokens must match to count as "this
    phrase" (typical ~0.25).
- **Prompt engineering is a real lever**: too generic (`"vehicle"`) maximizes
  recall but gives no type; too specific risks missing an oddball. Likely
  approach: a broad vehicle prompt for detection, with fine type handled
  separately.

## E. What it gives us — and what it does *not*

- **Gives:** boxes + a coarse category (whatever was prompted) + confidence. Raw
  material for detections and the "Vehicle Type" column.
- **Does *not* give:** fine make/model like *"Kia Forte"* or *"Cadillac
  Escalade."* Open-vocab detection is reliable at the *type* level, not the *trim*
  level. Make/model would need a **separate fine-grained classifier or a
  vision-language model** run on each vehicle crop. This is why the sample
  spreadsheet has the model column filled only sometimes — it is a genuinely
  harder, separate problem.
- **It is single-frame.** Grounding DINO has no memory, no identity across
  frames. It answers "what/where in *this* image." Turning that into *tracks over
  time* is Step 3's job (SAM2).

## F. Where it sits in the pipeline

```text
sampled frame --> Grounding DINO --> vehicle boxes --> (each box seeds) --> SAM2 --> mask + track
                  "what & where"                                           "exact pixels & follow over time"
```

Grounding DINO is the **prompt-driven front door**; its boxes become the seeds
SAM2 turns into precise masks and temporal tracks.

## G. The practical shape (for implementation)

- **Checkpoints:** Swin-T ("tiny", fast, ~700 MB) vs Swin-B ("base", more
  accurate, heavier). Start with **tiny** — plenty for clear roadside vehicles,
  runs in a few GB of VRAM (comfortable on the T4's 16 GB).
- **Source:** cleanest path is HuggingFace `transformers`
  (`IDEA-Research/grounding-dino-tiny` with its processor); the original
  IDEA-Research repo is the alternative.
- **First run:** point it at **one frame of `c0_5`** (the Cadillac clip) with a
  vehicle prompt and *look at the boxes* before scaling to anything. Prove it on
  one image, then grow.

---

## H. Swin forward pass, concretely (Swin-T)

Config for **Swin-Tiny** (the `-tiny` checkpoint from section G): embed dim
`C = 96`, blocks per stage `[2, 2, 6, 2]`, attention heads `[3, 6, 12, 24]`
(head dim stays 32 throughout: 96/3 = 192/6 = … = 32), window `M = 7`, MLP
expansion ratio 4. ~28 M parameters. Trace a `224×224` image through it:

```text
input image                         1 × 3 × 224 × 224
  │  Patch Partition (4×4) + Linear Embedding
  │    each 4×4×3 = 48-dim patch  ->  C=96
  ▼
tokens                              1 × (56·56) × 96      = 1 × 3136 × 96   (stride 4)
  │  Stage 1: 2 Swin blocks  (dim 96,  56×56)
  ▼                                 1 × 3136 × 96
  │  Patch Merging (2×2 -> 1, C·2)
  ▼                                 1 × 784  × 192        (28×28, stride 8)
  │  Stage 2: 2 Swin blocks  (dim 192, 28×28)
  ▼                                 1 × 784  × 192
  │  Patch Merging
  ▼                                 1 × 196  × 384        (14×14, stride 16)
  │  Stage 3: 6 Swin blocks  (dim 384, 14×14)
  ▼                                 1 × 196  × 384
  │  Patch Merging
  ▼                                 1 × 49   × 768        (7×7, stride 32)
  │  Stage 4: 2 Swin blocks  (dim 768, 7×7)
  ▼
final feature map                   1 × 49 × 768
```

**Patch dim:** unlike ViT's 16×16, Swin uses a **4×4** patch, so each token
starts from a `4×4×3 = 48`-value vector, linearly projected to `C = 96`.

**Inside one Swin block** (the repeated unit) — note the attention type
*alternates* between consecutive blocks:

```text
x ─► LayerNorm ─► (S)W-MSA ─► + ─► LayerNorm ─► MLP(GELU, 4×) ─► +  ─► out
│                             ▲                                  ▲
└───── residual ──────────────┘                └── residual ─────┘

block 2k   : W-MSA   (regular 7×7 windows)
block 2k+1 : SW-MSA  (windows shifted by 3, with a mask so the cyclic-shift
                      wrap-around regions don't attend to each other)
```

**Patch Merging** is the downsampler: concatenate each `2×2` neighborhood
(`4C` channels), then a linear layer to `2C`. Resolution halves, channels
double — this is what builds the pyramid.

**Final output — two modes:**

- *Classification:* global-average-pool the `7×7×768` map → `768`-vector →
  linear head. (Not what we use.)
- *Detection (Grounding DINO):* export the **per-stage** maps (stages 2/3/4 →
  strides 8/16/32, i.e. `28×28×192`, `14×14×384`, `7×7×768`), each `1×1`-conv'd
  to the transformer width `d = 256`, then flattened and concatenated into one
  sequence of image tokens. That multi-scale set is what feeds the fusion.

> For our real `1600×900` frames the token counts are far larger (stage-1 grid
> is `400×225 = 90,000` tokens), which is exactly why the *linear* cost below
> matters.

## I. Why window attention is O(N)

Let the feature map have `N = h·w` tokens with channel width `C`, window side
`M`. From the Swin paper, the cost of a self-attention layer is:

```text
Global MSA (ViT):   Ω = 4·N·C²  +  2·N²·C
Window MSA (Swin):  Ω = 4·N·C²  +  2·M²·N·C
```

Both share the `4NC²` term (the Q/K/V and output linear projections). The
difference is the **attention itself**:

- Global: `2·N²·C` — **quadratic** in `N`. Double the image side → 4× tokens →
  16× this term.
- Windowed: `2·M²·N·C` — `M` is a **fixed constant** (7), so this is **linear**
  in `N`.

Concrete, at stage 1 of a `224²` image (`N = 3136, C = 96, M = 7`):

```text
global attention term : 2·N²·C = 2·3136²·96      ≈ 1.89 ×10⁹
window attention term : 2·M²·N·C = 2·49·3136·96   ≈ 2.95 ×10⁷
ratio ≈ N / M² = 3136 / 49 = 64× cheaper
```

At detection resolution (`1600×900` → `N = 90,000`) the ratio is
`N/M² ≈ 1837×` — the gap *widens* with image size, which is the whole point:
window attention keeps high-resolution frames tractable. The shifted windows
(section G's "Move 2") restore global information flow *without* reintroducing
the `N²` term.

## J. BERT, and how it fuses with Swin (the combined model)

Grounding DINO runs the whole transformer at a common width **`d = 256`**, so
the first job is to bring both modalities to that width.

**Text side (BERT).** The prompt `"car . truck"` is tokenized (WordPiece) into
`[CLS] car . truck [SEP]`, and BERT (`bert-base-uncased`, 12 layers, hidden
768) encodes it → `1 × L × 768` (here `L = 6`). A **sub-sentence attention
mask** stops tokens of *different* phrases (split by `.`) from attending to each
other, so `"pickup truck"` stays one concept and doesn't bleed into `"car"`.
A linear layer projects `768 → 256`.

**Image side (Swin).** The multi-scale maps from section H are projected to
`256` and flattened → `1 × N_img × 256` (e.g. `N_img = 784+196+49 = 1029`).

**Feature Enhancer (the fusion core).** Stacked layers that interleave:

```text
image tokens ─► deformable self-attention (image only)
text  tokens ─► self-attention (text only)
        └──► image→text cross-attention  ┐  bidirectional fusion:
        ┌──► text→image cross-attention  ┘  image becomes text-aware,
                                            text becomes image-aware
```

**Language-guided query selection** then picks the top-`k` image tokens most
similar to the text (default `k = 900`) to initialize decoder queries. The
**cross-modality decoder** refines each query into a box while attending to both
image and text, and — crucially — emits an **alignment score against every text
token** rather than a fixed class id. Combined forward shapes:

```text
image  1×3×224×224 ─Swin─► 1×1029×256 ┐
                                       ├─ Enhancer ─► queries 1×900×256
text   "car . truck" ─BERT─► 1×6×256  ┘        │
                                               ├─► Decoder ─► boxes  1×900×4
                                               └─►           align  1×900×6
```

That `align 1×900×6` is the heart of it: for each of 900 candidate boxes, a
similarity to each of the 6 text tokens — "*this box matches the `car` token at
0.9*."

## K. Worked example — forward *and* backward through the combined model

**Input.** Image `1×3×224×224` containing one car; prompt `"car . truck"`.

**Forward** (shapes as above) yields, say, query #37 as the best box, with
alignment logits over the 6 tokens. Focus on the two content tokens:

```text
q      = decoder feature of query #37            (256-d)
t_car  = BERT/enhancer feature of "car"          (256-d)
t_truck= BERT/enhancer feature of "truck"        (256-d)

alignment logits:  s_car   = q · t_car
                   s_truck = q · t_truck
```

**Loss (training only).** Grounding DINO has no fixed classes, so:

1. **Hungarian matching** pairs each ground-truth box (with its phrase, here
   "car") to one query — say query #37 wins.
2. The total loss on the match is
   `L = λ_cls·L_contrastive + λ_L1·‖b̂ − b‖₁ + λ_giou·L_giou(b̂, b)`,
   where `L_contrastive` is a **sigmoid focal loss** over the per-token
   alignment logits: target = 1 for the `car` token, 0 for `truck`.

**Backward — the scalar illustration.** With `p_j = σ(s_j)` and targets
`y_car = 1, y_truck = 0`, the alignment-loss gradient is simply `p_j − y_j`:

```text
∂L/∂s_car   = p_car   − 1     (< 0  → push s_car UP)
∂L/∂s_truck = p_truck − 0     (> 0  → push s_truck DOWN)

∂L/∂q       = (p_car−1)·t_car + p_truck·t_truck   → move q  toward t_car, away from t_truck
∂L/∂t_car   = (p_car−1)·q                          → move t_car toward q
```

**Where those gradients go — the key point of a *combined* model:**

```text
        L (contrastive + box)
          │
          ▼
     Cross-modality Decoder
          │
          ▼
      Feature Enhancer  ──────────────┐  (fusion couples the two streams)
        ╱               ╲             │
   ∂L/∂(image feats)   ∂L/∂(text feats)
        │                   │
        ▼                   ▼
      Swin  ◄── updated     BERT ◄── updated
   (image backbone)      (text backbone)
```

Because fusion happens **early** (the enhancer) and the loss is on
**image↔text alignment**, gradients flow into *both* backbones in one pass:
Swin learns visual features that land near BERT's word vectors, and BERT nudges
`"car"`/`"truck"` embeddings toward their visual evidence. (In practice BERT
often gets a smaller learning rate, but both are fine-tuned jointly.) This
mutual pull is *why* a novel prompt at inference can localize an object the
detector never saw as a fixed class.

> We only ever run the **forward** pass (inference) in this project — the
> backward pass above is the *training* mechanism, included so the fusion makes
> sense. The pretrained weights already encode this alignment.

---

## Open threads to go deeper on later

- DETR's set-prediction idea — why no anchors / no NMS.
- Deformable attention specifics inside the enhancer/decoder.
- Prompt strategy for type vs recall, and thresholds tuning.
- Fine make/model: separate classifier vs VLM on crops.
