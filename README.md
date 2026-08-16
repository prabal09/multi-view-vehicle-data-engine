# Multi-View Vehicle Data Engine

An open-vocabulary, multi-view perception pipeline that finds vehicles in
roadside video from **5 calibrated cameras**, localizes each vehicle into one
shared **bird's-eye-view (BEV)** frame, associates the same physical vehicle
across cameras, and stores the result in a **queryable database**.

The headline query it answers:

> *"A vehicle (say a Kia Forte) appears at frame **N** in camera **c0**. What are
> the corresponding tracks and frame ranges for that same vehicle in
> **c1, c2, c3, c4**?"*

---

## The two halves of this project

```text
                         ┌────────────────────────────────────────────┐
   raw video (5 views)   │            PERCEPTION PIPELINE             │
   c0..c4  MP4 clips ───▶ │                                            │
                         │  1. sample frames                          │
                         │  2. Grounding DINO  → vehicle boxes (open- │
                         │        vocabulary: "car . suv . truck …")  │
                         │  3. SAM2            → masks + per-camera   │
                         │        tracking through time               │
                         │  4. homography      → project each vehicle │
                         │        into the shared BEV frame  ◀── reuses│
                         │        existing frameCalibration.yml       │
                         │  5. cross-view association: fuse BEV       │
                         │        position + appearance → one         │
                         │        identity per physical vehicle       │
                         └───────────────────────┬────────────────────┘
                                                 │ per-vehicle multi-view
                                                 │ frame stacks (like
                                                 │ VehicleSyncData.xlsx)
                                                 ▼
                         ┌────────────────────────────────────────────┐
                         │            RESULTS DATABASE                │
                         │  PostgreSQL on EC2                          │
                         │  vehicles / tracks / detections / cameras  │
                         │  + cross-view query functions              │
                         └────────────────────────────────────────────┘
```

**Half 1 — Perception.** Replaces an older Mask R-CNN detector with a modern
open-vocabulary stack (Grounding DINO for *what/where*, SAM2 for *exact pixels
and tracking over time*), then reuses the project's existing homography-based
BEV geometry to place vehicles in one common ground plane and associate them
across views.

**Half 2 — Storage & query.** A PostgreSQL database that stores the associated
vehicles and their per-camera tracks, and answers the cross-view lookup above.
`VehicleSyncData.xlsx` is a hand-authored sample of the intended output shape.

---

## What already exists and is reused (not rebuilt)

Prior work in `…/AvaCar/AvaCAR_code/localization/` provides the **BEV geometry**,
which still holds even though the detector is being modernized:

| Asset | Role | Status |
|---|---|---|
| `frameCalibration.yml` | Per-camera **homography** `h_c0..h_c4` mapping each image plane → shared top-view (BEV), plus vanishing points & horizon lines | ✅ reuse |
| `framecalibration.py` | Builds the calibration above from hand-annotated point correspondences | ✅ reuse |
| `point_of_contact.py`, `tangent_seg.py`, `frameLoc.py` | Compute a vehicle's **ground-contact point** (what gets projected through the homography into BEV) | ✅ adapt |
| `MRCNN.pt` (Mask R-CNN) | Old detector/segmenter | ⛔ replaced by Grounding DINO + SAM2 |

Input data: MP4 clips per camera (`c0_0.MP4 … c0_11.MP4`, ~11–12 per view) in
`…/AvaCar/OpenCV Practice/roadside_video/`. This is a subset; the remainder is
in Google Drive. The full target is ~8 hours × 5 views.

---

## Infrastructure & cost model

Runs on an **AWS EC2 GPU instance** (region `us-east-2`).

- **Instance:** develop on a cheap **`g4dn.xlarge`** (NVIDIA T4, 16 GB VRAM,
  ~$0.53/hr); use a **`g5.xlarge`** (A10G, 24 GB) only for a heavy full-dataset
  pass.
- **Base image:** an AWS **Deep Learning AMI** (NVIDIA driver + CUDA + cuDNN +
  PyTorch pre-verified together), so we manage only the Python layer.
- **GPU quota:** "Running On-Demand G and VT instances" is a *separate* quota
  from Standard vCPUs and may be **0** on a first GPU launch — request an
  increase (≥ 4 vCPUs) before anything else.
- **Cost discipline:** **stop** (not terminate) the instance when idle — compute
  billing stops; only the small EBS disk charge remains. Save an **AMI** once set
  up so the environment can never be lost.

---

## Data model (key insight)

Camera-local track labels such as `c0-5` are **not globally unique** — the same
label is reused by different vehicles with different frame ranges. So tracks are
keyed on a surrogate id, and cross-view lookups resolve by `(camera, frame)`
against the track's frame range. Per-frame *precise* cross-view sync is deferred
(it needs a global BEV time alignment); for now association is at the **track**
level.

---

## Build roadmap

| # | Step | What it does | Status |
|---|---|---|---|
| 1 | **Environment** | EC2 GPU instance, Deep Learning AMI, Python env, models loading | 🔨 in progress |
| 2 | **Detection** | Grounding DINO open-vocabulary vehicle boxes on sampled frames | ⬜ |
| 3 | **Segmentation + tracking** | SAM2 masks + per-camera tracks over time | ⬜ |
| 4 | **BEV localization** | Ground-contact point → homography → shared BEV (reuse existing calibration) | ⬜ |
| 5 | **Cross-view association** | Fuse BEV position + appearance → one identity per vehicle | ⬜ |
| 6 | **Storage + query** | PostgreSQL schema, loader, cross-view query | 🟡 prototype exists |

---

## Repository layout

```text
multi-view-vehicle-data-engine/
├── README.md                 ← this file
├── VehicleSyncData.xlsx      ← sample of the intended output
├── db/                       ← PostgreSQL storage layer (prototype)
│   ├── schema.sql            ← tables: cameras, vehicles, tracks, detections
│   ├── views.sql             ← cross-view query views + functions
│   └── config.py             ← DB connection helper
├── etl/                      ← load pipeline output into the DB
│   ├── parse.py              ← parse VehicleSyncData.xlsx → records
│   └── load_xlsx.py          ← insert records into PostgreSQL
├── query/
│   └── query.py              ← CLI for the cross-view lookup
├── docker-compose.yml        ← local PostgreSQL for development
└── requirements.txt
```

> The `db/`, `etl/`, and `query/` code is an early **prototype of Half 2** built
> from the sample spreadsheet. It will be revisited and walked through once the
> perception pipeline (Half 1) is producing real output.

---

## Status

Early build, developed step by step. Step 1 (environment) is underway; the
perception stages are next, followed by wiring real pipeline output into the
database layer.
