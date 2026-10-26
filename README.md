# Multi-View Vehicle Data Engine

An open-vocabulary, multi-view perception pipeline that finds vehicles in
roadside video from **5 calibrated cameras**, localizes each vehicle into one
shared **bird's-eye-view (BEV)** frame, associates the same physical vehicle
across cameras, and stores the result in a **queryable PostgreSQL database**
(local via Docker, or AWS RDS in the cloud).

The headline query it answers:

> *"A vehicle appears at frame **N** in camera **c0** (clip `c0_5`). What are the
> corresponding tracks and frame ranges for that same vehicle in
> **c1, c2, c3, c4**?"* — and the reverse: *"give me all the SUVs."*

**Status: end-to-end working.** Raw video → detect → track → BEV-localize →
associate → Postgres → cross-view query, deployed to AWS RDS and returning the
correct per-camera frame ranges.

---

## The two halves of this project

```text
                         ┌────────────────────────────────────────────┐
   raw video (5 views)   │            PERCEPTION PIPELINE  (GPU/EC2)   │
   c0..c4  MP4 clips ───▶ │                                            │
                         │  1. sample frames                          │
                         │  2. Grounding DINO → vehicle boxes (open-  │
                         │        vocabulary: "car. suv. truck. …")   │
                         │  3. SAM2 (windowed) → masks + per-camera   │
                         │        tracking through time               │
                         │  4. homography → project each vehicle into │
                         │        the shared BEV frame (reuses the    │
                         │        existing frameCalibration.yml)      │
                         │  5. cross-view association: fuse BEV       │
                         │        position + CLIP appearance → one    │
                         │        identity per physical vehicle       │
                         │  5b. coarse vehicle_type (CLIP vote)       │
                         └───────────────────────┬────────────────────┘
                                                 │ runs/*_tracks.json
                                                 │ (boxes + embeddings)
                                                 ▼
                         ┌────────────────────────────────────────────┐
                         │        RESULTS DATABASE  (CPU, decoupled)  │
                         │  PostgreSQL — Docker (dev) / AWS RDS (prod)│
                         │  cameras / vehicles / tracks               │
                         │  ETL loader + cross-view query CLI         │
                         └────────────────────────────────────────────┘
```

**Half 1 — Perception (GPU).** A modern open-vocabulary stack: **Grounding DINO**
for *what/where* (detects any vehicle class from a text prompt, no retraining),
**SAM2** for *exact pixels and tracking over time*, then the project's existing
homography-based BEV geometry to place vehicles in one common ground plane and
associate them across views by **appearance + a per-camera uniqueness
constraint** (no common clock required).

**Half 2 — Storage & query (CPU).** A PostgreSQL database, deliberately
**decoupled** from the GPU side: the pipeline emits `runs/*_tracks.json` (boxes +
appearance embeddings), and a pure-CPU ETL job localizes, associates, and loads
them. The same SQL runs on local Docker Postgres and on AWS RDS.

---

## Build roadmap — all steps complete

| # | Step | What it does | Code |
|---|---|---|---|
| 1 | **Environment** | EC2 `g4dn.xlarge` + Deep Learning AMI (saved as AMI v2), Python env, checkpoints | `scripts/` |
| 2 | **Detection** | Grounding DINO open-vocab boxes; score filter + label-unify + NMS | `pipeline/detection.py`, `pipeline/detect_clip.py`, `pipeline/sampling.py` |
| 3 | **Tracking** | SAM2 **windowed** tracking (bounded memory) + pre-link seeds + post-merge across windows | `pipeline/tracking.py`, `pipeline/sam2_engine.py`, `pipeline/track_clip.py` |
| 4 | **BEV localization** | Ground-contact point → homography → shared BEV; horizon/ROI/speed filters | `pipeline/bev.py` |
| 5 | **Cross-view association** | Fuse CLIP appearance + per-camera uniqueness → one identity per vehicle | `pipeline/association.py`, `pipeline/appearance.py` |
| 5b | **Vehicle type** | CLIP zero-shot coarse type `{suv, car, truck, bus}`, voted per track (Colab) | `sanity/classify_types.py`, `sanity/extract_crops.py` |
| 6 | **Storage + query** | Postgres schema, ETL loader, cross-view query CLI; deployed to RDS | `db/`, `etl/load_runs.py`, `query/query.py` |

---

## Repository layout

```text
multi-view-vehicle-data-engine/
├── README.md
├── VehicleSyncData.xlsx          ← hand-authored sample of the target output shape
├── docker-compose.yml            ← local Postgres 16 for dev
├── requirements.txt              ← CPU side only (no torch)
│
├── pipeline/                     ← Half 1: perception (runs on EC2 GPU)
│   ├── sampling.py               ← frame sampling / extraction
│   ├── detection.py              ← clean_detections: score filter + NMS
│   ├── detect_clip.py            ← Grounding DINO detector + per-clip driver
│   ├── tracking.py               ← association brain: IoU, seed planning, window merge
│   ├── sam2_engine.py            ← SAM2 model registry + windowed track engine
│   ├── track_clip.py             ← windowing + full run_clip() driver
│   ├── bev.py                    ← Step 4: homography → BEV + validity filters
│   ├── appearance.py             ← CLIP appearance embeddings per track
│   └── association.py            ← Step 5: constrained agglomerative clustering
│
├── db/                           ← Half 2: database
│   ├── schema.sql                ← cameras / vehicles / tracks (normalized)
│   └── config.py                 ← DATABASE_URL (RDS) or localhost default
├── etl/
│   └── load_runs.py              ← Extract+Transform+Load runs/*.json → Postgres
├── query/
│   └── query.py                  ← cross-view query CLI (frame / vehicle / type / list)
│
├── scripts/                      ← EC2 batch runners
│   ├── run_one_clip.py
│   └── run_all_clips.sh
├── sanity/                       ← Colab validation experiments + crop/type tooling
├── tools/downscale_clips.ps1     ← 4K → 1600×900 H.264 data prep
└── docs/                         ← step-by-step theory notes (GDINO, SAM2)
```

---

## Data & addressing model (key insight)

Addressing is **(camera, clip, local-frame)** — e.g. `c0` / `c0_5` / frame `590`.
Clips are **not** merged (merging risks track-continuity errors), and a
camera-local track label like `c0-5` is **not** globally unique. So:

- **Tracks** are keyed on a surrogate `track_id`; cross-view lookups resolve by
  `(camera, clip, frame)` against each track's `frame_start..frame_end` range.
- **Vehicles** are association clusters. A physical vehicle has up to one track
  per camera (the per-camera uniqueness constraint in Step 5).
- `vehicle_id` on a track is **nullable**: a track can be ingested before
  association runs, then updated once it's clustered.
- Frame-*precise* cross-view sync (one global clock) is deferred; association is
  at the **track** level, which needs no common clock.

### Schema (`db/schema.sql`)

```text
cameras(camera_id PK, name, fps)
vehicles(vehicle_id PK, vehicle_type, color [reserved], natural_key UNIQUE, created_at)
tracks(track_id PK, vehicle_id FK→vehicles (nullable, ON DELETE SET NULL),
       camera_id FK→cameras, clip, frame_start, frame_end, time_start, time_end,
       UNIQUE(camera_id, clip, frame_start))
```

Indexes: `tracks(camera_id, clip, frame_start, frame_end)` for range lookup,
`tracks(vehicle_id)` for the cross-view answer, `vehicles(vehicle_type)` for
"all SUVs".

---

## Running it

### Half 1 — perception (EC2 GPU)

```bash
# one clip-window → runs/<clip>_tracks.json (boxes + embeddings)
python scripts/run_one_clip.py --clip c0_5 --start 0 --end 600 \
    --sam-model large --window 120 --overlap 20
# or batch all clips
bash scripts/run_all_clips.sh
```

Key flags: `--window/--overlap` bound SAM2 memory (fresh state per window,
overlap-stitched); `--offload` moves SAM2 state to CPU under VRAM pressure;
`--no-appearance` skips CLIP embedding. **Stop (don't terminate)** the instance
when idle; the environment is saved as an AMI.

### Half 2 — database & query (CPU, local or RDS)

```bash
# local Postgres for dev
docker compose up -d

# load the run outputs (BEV-filter → associate → insert). --init applies schema.
python -m etl.load_runs --init --runs runs --types runs/track_types.json

# the headline cross-view query
python -m query.query frame c0 c0_5 590     # → that vehicle's (clip, range) in every camera
python -m query.query type suv              # → all SUVs
python -m query.query vehicle 2             # → one vehicle's per-camera tracks
python -m query.query list                  # → every vehicle
```

**Local vs RDS** is controlled entirely by `DATABASE_URL` in `.env`
(gitignored). Unset → local Docker Postgres; set to the RDS endpoint (with
`sslmode=require`) → cloud. The AWS Console "Query Editor" is **Aurora-only**;
for this standard RDS Postgres instance, query via `query.py`, `psql`, or a GUI
client (pgAdmin / DBeaver / TablePlus).

---

## What is reused (not rebuilt)

Prior AvaCar work in `…/AvaCar/AvaCAR_code/localization/` supplies the **BEV
geometry**, which still holds after modernizing the detector:

| Asset | Role |
|---|---|
| `frameCalibration.yml` | Per-camera homographies `h_c0..h_c4` (image plane → shared BEV, ~1px) + horizon lines. Clips are 1600×900, matching the calibration — no rescaling. |
| `framecalibration.py` | Builds the calibration from annotated point correspondences. |
| `point_of_contact.py` etc. | Ground-contact point logic (adapted into `bev.py`). |
| `MRCNN.pt` (Mask R-CNN) | ⛔ Replaced by Grounding DINO + SAM2. |

---

## Infrastructure & cost

- **Compute:** EC2 `g4dn.xlarge` (T4, 16 GB) on a Deep Learning AMI, region
  `us-east-2`; **stop** when idle, environment saved as an AMI. GPU quota
  ("Running On-Demand G and VT instances") is separate from standard vCPUs and
  may be 0 on a first launch — request an increase first.
- **Database:** AWS RDS PostgreSQL, **Single-AZ**, `db.t3.micro`, gp3 20 GB
  (~$15/mo). Security group restricts port 5432 to specific IPs (never
  `0.0.0.0/0`); connections use `sslmode=require`. Credentials live only in
  `.env` (gitignored).

---

## Known limitations / backlog

- **c4_2 Cadillac** is a single-view, too-small/too-far track (max box ~69×41,
  below the crop floor) — it stays a singleton and has no `vehicle_type`. A known
  data-quality limit of that view, not a bug.
- Backlog: direct psycopg logging during the GPU run (skip the JSON hop),
  stream/incremental ingestion, **color** queries ("red SUV" — column reserved),
  GPU-skip registry (detect → compare to DB → skip SAM2), frame-precise sync,
  appearance ReID / viewpoint-aware embeddings, c4 BEV-fusion, a read-only
  `viewer` DB role for sharing.