"""Cross-view vehicle lookup CLI.

Answers: given a vehicle seen in one camera (by frame or by track label),
what are the corresponding tracks / frame ranges in the other cameras?

Examples (from the repo root):

    # By camera + frame -- "who is at frame 592 in c0, and where in c1..c4?"
    python -m query.query frame c0 592

    # By camera-local track label (labels repeat, so add a frame to disambiguate)
    python -m query.query label c0-5
    python -m query.query label c0-5 --frame 592

    # By vehicle id
    python -m query.query vehicle 1

    # List everything (the spreadsheet-style matrix)
    python -m query.query list

Output is a table by default; pass --json for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.config import connect                          # noqa: E402

CAMERAS = ["c0", "c1", "c2", "c3", "c4"]

# Ordered column set returned by the cross_view_* functions.
_COLS = [
    "vehicle_id", "vehicle_type", "vehicle_model",
    "camera_id", "label", "frame_start", "frame_end", "time_start", "time_end",
]


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(zip(_COLS, r)) for r in rows]


def run_frame(conn, camera: str, frame: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM cross_view_by_frame(%s, %s)", (camera, frame))
        return _rows_to_dicts(cur.fetchall())


def run_label(conn, label: str, frame: int | None) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM cross_view_by_label(%s, %s)", (label, frame))
        return _rows_to_dicts(cur.fetchall())


def run_vehicle(conn, vehicle_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM cross_view_for_vehicle(%s)", (vehicle_id,))
        return _rows_to_dicts(cur.fetchall())


def run_list(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM v_vehicle_tracks")
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _fmt_range(a, b) -> str:
    if a is None and b is None:
        return "-"
    return f"{'' if a is None else a}..{'' if b is None else b}"


def print_grouped(rows: list[dict]) -> None:
    """Print one block per vehicle, one line per camera -- mirrors the sheet."""
    if not rows:
        print("No match.")
        return
    by_vehicle: dict = defaultdict(list)
    for r in rows:
        by_vehicle[r["vehicle_id"]].append(r)

    for vid, tracks in by_vehicle.items():
        head = tracks[0]
        label = " / ".join(x for x in (head.get("vehicle_type"),
                                       head.get("vehicle_model")) if x) or "(unlabelled)"
        print(f"\nvehicle #{vid}  {label}")
        print(f"  {'cam':<4} {'track':<8} {'frames':<14} {'time (s)':<14}")
        print(f"  {'-'*4} {'-'*8} {'-'*14} {'-'*14}")
        for t in sorted(tracks, key=lambda x: x["camera_id"]):
            print(f"  {t['camera_id']:<4} {str(t['label'] or '-'):<8} "
                  f"{_fmt_range(t['frame_start'], t['frame_end']):<14} "
                  f"{_fmt_range(t['time_start'], t['time_end']):<14}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-view vehicle lookup")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_frame = sub.add_parser("frame", help="lookup by camera + frame")
    p_frame.add_argument("camera", choices=CAMERAS)
    p_frame.add_argument("frame", type=int)

    p_label = sub.add_parser("label", help="lookup by camera-local track label")
    p_label.add_argument("label")
    p_label.add_argument("--frame", type=int, default=None,
                         help="disambiguate reused labels")

    p_veh = sub.add_parser("vehicle", help="lookup by vehicle id")
    p_veh.add_argument("vehicle_id", type=int)

    sub.add_parser("list", help="list all vehicles/tracks")

    args = ap.parse_args(argv)

    with connect() as conn:
        if args.cmd == "frame":
            rows = run_frame(conn, args.camera, args.frame)
        elif args.cmd == "label":
            rows = run_label(conn, args.label, args.frame)
        elif args.cmd == "vehicle":
            rows = run_vehicle(conn, args.vehicle_id)
        else:
            rows = run_list(conn)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        print_grouped(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
