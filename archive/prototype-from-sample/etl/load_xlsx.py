"""Load VehicleSyncData.xlsx into PostgreSQL.

Usage (from the repo root):

    # one-shot: (re)create schema + load the sample
    python -m etl.load_xlsx --xlsx VehicleSyncData.xlsx --init

    # load only (schema already applied)
    python -m etl.load_xlsx --xlsx VehicleSyncData.xlsx

Flags:
    --init      apply db/schema.sql and db/views.sql before loading
    --truncate  delete existing vehicles/tracks first (idempotent reload)
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as a script (python etl/load_xlsx.py) as well as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.config import connect                         # noqa: E402
from etl.parse import (                               # noqa: E402
    CameraExtrinsic,
    VehicleRecord,
    nonempty_vehicles,
    parse_extrinsics,
    parse_vehicles,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMERAS = ["c0", "c1", "c2", "c3", "c4"]


def _apply_sql_file(cur, path: str) -> None:
    with open(path, "r", encoding="utf-8") as fh:
        cur.execute(fh.read())


def init_schema(conn) -> None:
    with conn.cursor() as cur:
        _apply_sql_file(cur, os.path.join(REPO_ROOT, "db", "schema.sql"))
        _apply_sql_file(cur, os.path.join(REPO_ROOT, "db", "views.sql"))
    conn.commit()
    print("Applied db/schema.sql and db/views.sql")


def ensure_cameras(conn, extrinsics: list[CameraExtrinsic]) -> None:
    """Insert the c0..c4 camera rows (idempotent). Extrinsics from Sheet2 are
    merged in where present."""
    ext_by_cam = {e.camera: e for e in extrinsics}
    with conn.cursor() as cur:
        for cam in CAMERAS:
            e = ext_by_cam.get(cam)
            cur.execute(
                """
                INSERT INTO cameras (camera_id, tx, ty, tz, rotation)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (camera_id) DO NOTHING
                """,
                (
                    cam,
                    e.tx if e else None,
                    e.ty if e else None,
                    e.tz if e else None,
                    e.rotation if e else None,
                ),
            )
    conn.commit()


def truncate(conn) -> None:
    with conn.cursor() as cur:
        # tracks + detections cascade from vehicles; cameras are kept.
        cur.execute("TRUNCATE vehicles RESTART IDENTITY CASCADE")
    conn.commit()
    print("Truncated vehicles/tracks/detections")


def load_vehicles(conn, vehicles: list[VehicleRecord]) -> tuple[int, int]:
    n_vehicles = n_tracks = 0
    with conn.cursor() as cur:
        for v in vehicles:
            cur.execute(
                """
                INSERT INTO vehicles (sheet_index, vehicle_type, vehicle_model)
                VALUES (%s, %s, %s)
                RETURNING vehicle_id
                """,
                (v.index, v.vehicle_type, v.vehicle_model),
            )
            vehicle_id = cur.fetchone()[0]
            n_vehicles += 1

            seen_cams: set[str] = set()
            for t in v.tracks:
                if t.camera not in CAMERAS or t.camera in seen_cams:
                    # Skip duplicates/unknown cameras (UNIQUE(vehicle,camera)).
                    continue
                seen_cams.add(t.camera)
                cur.execute(
                    """
                    INSERT INTO tracks
                        (vehicle_id, camera_id, label,
                         frame_start, frame_end, time_start, time_end)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        vehicle_id, t.camera, t.label,
                        t.frame_start, t.frame_end, t.time_start, t.time_end,
                    ),
                )
                n_tracks += 1
    conn.commit()
    return n_vehicles, n_tracks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load VehicleSyncData.xlsx into PostgreSQL")
    ap.add_argument("--xlsx", default="VehicleSyncData.xlsx", help="path to the workbook")
    ap.add_argument("--init", action="store_true", help="apply schema.sql + views.sql first")
    ap.add_argument("--truncate", action="store_true", help="clear existing rows before load")
    args = ap.parse_args(argv)

    vehicles = nonempty_vehicles(parse_vehicles(args.xlsx))
    extrinsics = parse_extrinsics(args.xlsx)
    print(f"Parsed {len(vehicles)} vehicles from {args.xlsx}")

    with connect() as conn:
        if args.init:
            init_schema(conn)
        if args.truncate:
            truncate(conn)
        ensure_cameras(conn, extrinsics)
        nv, nt = load_vehicles(conn, vehicles)

    print(f"Loaded {nv} vehicles and {nt} tracks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
