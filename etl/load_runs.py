"""ETL: load perception run outputs (runs/*_tracks.json) into Postgres.

  Extract   : read the per-camera track JSONs
  Transform : BEV-filter (localize_tracks) -> associate across cameras -> rows
  Load      : (optional schema init) + truncate + insert cameras/vehicles/tracks

Idempotent: a full truncate-and-reload, so re-running yields the same DB state.
All CPU -- no torch (uses the embeddings already baked into the JSONs).

Usage (repo root, local Postgres running):
    python -m etl.load_runs --init
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

from db.config import connect
from pipeline.association import associate
from pipeline.bev import load_homographies, load_horizons, localize_tracks

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 29.97
CAMERAS = ["c0", "c1", "c2", "c3", "c4"]
DEFAULT_YML = (r"C:/Users/praba/PycharmProjects/AvaCar/AvaCAR_code/"
               r"localization/frameCalibration.yml")


# ------------------------------ Extract + Transform --------------------------
def extract_and_transform(runs_dir: str, yml: str):
    """Return (tracks, key->vehicle_index, n_vehicles).

    tracks: list of {key, camera, clip, frame_start, frame_end, embedding}
    """
    H = load_homographies(yml)
    hz = load_horizons(yml)

    tracks = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "*_tracks.json"))):
        data = json.load(open(f))
        by_id = {t["track_id"]: t for t in data}
        for tid, traj in localize_tracks(data, H, hz).items():     # BEV filter
            t = by_id[tid]
            frames = sorted(traj)
            tracks.append({
                "key": f"{t['camera']}#{t['clip']}#{tid}",
                "camera": t["camera"], "clip": t["clip"],
                "frame_start": frames[0], "frame_end": frames[-1],
                "embedding": t.get("embedding"),
            })

    # associate the tracks that have an appearance embedding
    usable = [t for t in tracks if t["embedding"]]
    clusters = associate([t["key"] for t in usable],
                         [t["camera"] for t in usable],
                         np.array([t["embedding"] for t in usable]),
                         sim_thresh=0.80)
    key_to_v = {k: vi for vi, cl in enumerate(clusters) for k in cl}

    # any track without an embedding becomes its own (single-view) vehicle
    n = len(clusters)
    for t in tracks:
        if t["key"] not in key_to_v:
            key_to_v[t["key"]] = n
            n += 1
    return tracks, key_to_v, n


def vote_vehicle_types(tracks, key_to_v, n_vehicles, types_path):
    """Majority-vote a coarse vehicle_type per vehicle from Step 5b's per-track
    types ({camera#clip#track: type}). Returns {vehicle_index: type|None}."""
    track_type = {}
    if types_path and os.path.exists(types_path):
        track_type = json.load(open(types_path, encoding="utf-8"))

    votes = defaultdict(list)
    for t in tracks:
        ty = track_type.get(t["key"])
        if ty:
            votes[key_to_v[t["key"]]].append(ty)

    return {vi: (Counter(votes[vi]).most_common(1)[0][0] if votes.get(vi) else None)
            for vi in range(n_vehicles)}


# ---------------------------------- Load -------------------------------------
def apply_schema(cur):
    # Execute the whole file in one call: Postgres parses the statements (and
    # the -- comments, which may contain ';') correctly. Splitting on ';'
    # ourselves breaks on a semicolon inside a comment or string literal.
    sql = open(os.path.join(REPO, "db", "schema.sql"), encoding="utf-8").read()
    cur.execute(sql)


def load(tracks, key_to_v, n_vehicles, vehicle_types, init: bool):
    with connect() as conn:
        with conn.cursor() as cur:
            if init:
                apply_schema(cur)
            cur.execute("TRUNCATE tracks, vehicles RESTART IDENTITY CASCADE")

            for c in CAMERAS:
                cur.execute("INSERT INTO cameras (camera_id, fps) VALUES (%s, %s) "
                            "ON CONFLICT (camera_id) DO UPDATE SET fps = EXCLUDED.fps",
                            (c, FPS))

            vids = {}
            for vi in range(n_vehicles):
                cur.execute("INSERT INTO vehicles (vehicle_type) VALUES (%s) RETURNING vehicle_id",
                            (vehicle_types.get(vi),))
                vids[vi] = cur.fetchone()[0]

            for t in tracks:
                cur.execute(
                    "INSERT INTO tracks (vehicle_id, camera_id, clip, frame_start, "
                    "frame_end, time_start, time_end) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (vids[key_to_v[t["key"]]], t["camera"], t["clip"],
                     t["frame_start"], t["frame_end"],
                     t["frame_start"] / FPS, t["frame_end"] / FPS))
        conn.commit()


def main():
    ap = argparse.ArgumentParser(description="load runs/*.json into Postgres")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--yml", default=DEFAULT_YML)
    ap.add_argument("--types", default="runs/track_types.json",
                    help="per-track types from Step 5b (optional; sets vehicle_type)")
    ap.add_argument("--init", action="store_true", help="apply db/schema.sql first")
    args = ap.parse_args()

    tracks, key_to_v, n_vehicles = extract_and_transform(args.runs, args.yml)
    vehicle_types = vote_vehicle_types(tracks, key_to_v, n_vehicles, args.types)
    print(f"transformed {len(tracks)} tracks -> {n_vehicles} vehicles")
    print(f"vehicle types: {vehicle_types}")
    load(tracks, key_to_v, n_vehicles, vehicle_types, args.init)
    print(f"loaded {len(tracks)} tracks and {n_vehicles} vehicles into Postgres")


if __name__ == "__main__":
    main()
