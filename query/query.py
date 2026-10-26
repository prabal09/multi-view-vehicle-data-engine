"""Cross-view vehicle query CLI (reads the Postgres DB loaded by etl.load_runs).

  frame  <camera> <clip> <frame>   a vehicle at that frame -> its (clip, range) in every camera
  vehicle <id>                     all tracks of one vehicle
  type   <vehicle_type>            all vehicles of a body-type (needs Step 5b to populate type)
  list                             every vehicle with its per-camera tracks

Example (the project's headline query):
  python -m query.query frame c0 c0_5 590
"""

from __future__ import annotations

import argparse

from db.config import connect


def _print_tracks(rows):
    for camera, clip, f0, f1, t0, t1 in rows:
        ts = f"{t0:.1f}-{t1:.1f}s" if t0 is not None else "-"
        print(f"    {camera}  {clip:<6} frames {f0}-{f1:<6} {ts}")


def tracks_of_vehicle(cur, vid):
    cur.execute("SELECT camera_id, clip, frame_start, frame_end, time_start, time_end "
                "FROM tracks WHERE vehicle_id = %s ORDER BY camera_id", (vid,))
    return cur.fetchall()


def cmd_frame(cur, camera, clip, frame):
    cur.execute("SELECT vehicle_id FROM tracks "
                "WHERE camera_id = %s AND clip = %s "
                "AND %s BETWEEN frame_start AND frame_end", (camera, clip, frame))
    row = cur.fetchone()
    if not row or row[0] is None:
        print(f"No associated vehicle at {camera}/{clip} frame {frame}.")
        return
    vid = row[0]
    cur.execute("SELECT vehicle_type FROM vehicles WHERE vehicle_id = %s", (vid,))
    vtype = cur.fetchone()[0]
    print(f"vehicle #{vid} (type: {vtype or 'unknown'}) at {camera}/{clip}:{frame} "
          f"-> seen in:")
    _print_tracks(tracks_of_vehicle(cur, vid))


def cmd_vehicle(cur, vid):
    print(f"vehicle #{vid}:")
    _print_tracks(tracks_of_vehicle(cur, vid))


def cmd_type(cur, vtype):
    cur.execute("SELECT vehicle_id FROM vehicles WHERE vehicle_type = %s "
                "ORDER BY vehicle_id", (vtype,))
    vids = [r[0] for r in cur.fetchall()]
    if not vids:
        print(f"No vehicles of type '{vtype}'. (type is populated by Step 5b)")
        return
    for vid in vids:
        print(f"vehicle #{vid} ({vtype}):")
        _print_tracks(tracks_of_vehicle(cur, vid))


def cmd_list(cur):
    cur.execute("SELECT vehicle_id, COALESCE(vehicle_type, 'unknown') "
                "FROM vehicles ORDER BY vehicle_id")
    for vid, vtype in cur.fetchall():
        print(f"vehicle #{vid} ({vtype}):")
        _print_tracks(tracks_of_vehicle(cur, vid))


def main():
    ap = argparse.ArgumentParser(description="cross-view vehicle query")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frame"); p.add_argument("camera"); p.add_argument("clip"); p.add_argument("frame", type=int)
    p = sub.add_parser("vehicle"); p.add_argument("vehicle_id", type=int)
    p = sub.add_parser("type"); p.add_argument("vehicle_type")
    sub.add_parser("list")
    args = ap.parse_args()

    with connect() as conn, conn.cursor() as cur:
        if args.cmd == "frame":
            cmd_frame(cur, args.camera, args.clip, args.frame)
        elif args.cmd == "vehicle":
            cmd_vehicle(cur, args.vehicle_id)
        elif args.cmd == "type":
            cmd_type(cur, args.vehicle_type)
        else:
            cmd_list(cur)


if __name__ == "__main__":
    main()
