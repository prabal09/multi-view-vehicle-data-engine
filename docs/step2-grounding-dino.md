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

## Open threads to go deeper on later

- DETR's set-prediction idea — why no anchors / no NMS.
- How the contrastive image<->text alignment is actually trained.
- Attention mechanics of the feature enhancer.
- Prompt strategy for type vs recall, and thresholds tuning.
- Fine make/model: separate classifier vs VLM on crops.
