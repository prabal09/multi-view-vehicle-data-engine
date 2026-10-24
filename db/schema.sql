-- ============================================================================
-- Multi-view vehicle data engine -- query database schema (clip-based model)
-- ============================================================================
-- Stores the perception pipeline's output so it can be queried:
--   "a vehicle at frame N in camera c0 (clip c0_5) -> its (clip, frame-range)
--    in c1..c4", and "give me all SUVs".
--
-- Model (normalized): one physical VEHICLE has up to five TRACKS (one per
-- camera), each identified by (camera, clip, frame-range). This mirrors the
-- (camera, clip, local-frame) addressing used throughout the pipeline.
--
-- Target: PostgreSQL. Identical SQL on a local Postgres (dev) and AWS RDS (prod).
-- Idempotent: all CREATE ... IF NOT EXISTS, safe to re-apply.
-- ============================================================================

-- The 5 calibrated viewpoints. Small reference table; also holds fps so we can
-- convert between frame index and seconds.
CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,          -- 'c0'..'c4'
    name      TEXT,
    fps       DOUBLE PRECISION           -- frame <-> time
);

-- One row per PHYSICAL vehicle (an association cluster from Step 5).
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    vehicle_type TEXT,                   -- coarse body-type: suv/car/truck/bus (CLIP vote, Step 5b)
    color        TEXT,                   -- reserved for future "red suv" queries (nullable)
    natural_key  TEXT UNIQUE,            -- deterministic cluster identity -> idempotent upsert
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per camera-track. vehicle_id is NULLABLE: a track is ingested as soon
-- as its camera is processed ("unassociated"), then UPDATEd when Step 5 runs.
CREATE TABLE IF NOT EXISTS tracks (
    track_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    vehicle_id  BIGINT REFERENCES vehicles(vehicle_id) ON DELETE SET NULL,
    camera_id   TEXT NOT NULL REFERENCES cameras(camera_id),
    clip        TEXT NOT NULL,           -- 'c0_5'
    frame_start INTEGER NOT NULL,
    frame_end   INTEGER NOT NULL,
    time_start  DOUBLE PRECISION,        -- seconds (frame_start / fps)
    time_end    DOUBLE PRECISION,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- natural key: lets re-ingesting a clip UPSERT instead of duplicate.
    UNIQUE (camera_id, clip, frame_start)
);

-- "which track in camera X / clip Y contains frame N" (range lookup)
CREATE INDEX IF NOT EXISTS tracks_lookup_idx ON tracks (camera_id, clip, frame_start, frame_end);
-- "all tracks of this vehicle" (the cross-view answer)
CREATE INDEX IF NOT EXISTS tracks_vehicle_idx ON tracks (vehicle_id);
-- "all SUVs"
CREATE INDEX IF NOT EXISTS vehicles_type_idx  ON vehicles (vehicle_type);
