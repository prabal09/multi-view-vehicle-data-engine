# Prototype (from the sample spreadsheet) — ARCHIVED REFERENCE

This folder is an **early prototype of the results-database layer (Half 2)**,
written from the hand-authored `VehicleSyncData.xlsx` *before* the real
perception pipeline existed. It is kept as a **reference only** — not live code.

We will rebuild the ETL and database **for real, together, incrementally**,
starting once the pipeline produces genuine output (after Step 3, per-camera
tracks), and evolving the schema as Steps 4–5 add BEV position and cross-view
identity. When we get there we'll mine this folder for ideas, but write each
line deliberately rather than inheriting this.

Contents:

- `db/schema.sql`, `db/views.sql` — PostgreSQL tables + cross-view query functions
- `etl/parse.py`, `etl/load_xlsx.py` — parse the sample xlsx and load it
- `query/query.py` — CLI for the cross-view lookup
- `docker-compose.yml`, `requirements.txt`, `.env.example` — local Postgres dev

Key idea worth carrying forward: camera-local track labels like `c0-5` are
**not globally unique**, so tracks are keyed on a surrogate id and resolved by
`(camera, frame)` range.
