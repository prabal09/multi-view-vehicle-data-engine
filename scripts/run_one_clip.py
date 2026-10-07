"""Run the per-clip detect+track pipeline on ONE clip and summarize. (EC2)

Example (on the GPU box, inside the pytorch venv):
    python scripts/run_one_clip.py \
        --clip ~/data/clips/c0/c0_5.mp4 --camera c0 --clip-name c0_5 \
        --out c0_5_tracks.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.track_clip import run_clip   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="detect+track one clip")
    ap.add_argument("--clip", required=True, help="path to the 1600x900 clip")
    ap.add_argument("--camera", required=True, help="e.g. c0")
    ap.add_argument("--clip-name", required=True, help="e.g. c0_5")
    ap.add_argument("--frames-dir", default="sam2_frames",
                    help="scratch dir for extracted frames (root disk, not /tmp)")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--end", type=int, default=None,
                    help="cap processing to frames [0, end) for a quick validation")
    ap.add_argument("--out", default="tracks.json")
    args = ap.parse_args()

    tracks = run_clip(args.clip, camera=args.camera, clip=args.clip_name,
                      frames_dir=args.frames_dir, step=args.step, end=args.end)

    print(f"\n=== {len(tracks)} tracks ===")
    for t in sorted(tracks, key=lambda t: t.frame_start):
        print(f"  track {t.track_id:>3}: frames {t.frame_start:>5}..{t.frame_end:<5} "
              f"({len(t.boxes)} frames)")

    data = [{"track_id": t.track_id, "camera": t.camera, "clip": t.clip,
             "seed_frame": t.seed_frame,
             "frame_start": t.frame_start, "frame_end": t.frame_end,
             "boxes": {str(f): list(b) for f, b in t.boxes.items()}}
            for t in tracks]
    with open(args.out, "w") as fh:
        json.dump(data, fh)
    print(f"\nwrote {args.out}  ({len(tracks)} tracks)")


if __name__ == "__main__":
    main()
