-- ============================================================================
-- Multi-view vehicle data engine -- PostgreSQL schema
-- ============================================================================
-- Stores the output of the open-vocabulary multi-view perception pipeline:
-- vehicles that have been associated across 5 calibrated viewpoints, the
-- per-camera tracks that make up each vehicle, and (for future frame-precise
-- work) per-frame detections plus camera calibration.
--
-- Target: AWS RDS for PostgreSQL 14+. Runs unchanged on a local Postgres.
-- The pgvector extension is OPTIONAL and only used for appearance embeddings;
-- if it is unavailable the embedding column degrades to real[] (see below).
-- ============================================================================

BEGIN;

-- Optional: appearance-embedding vector type. Comment this out (and switch the
-- detections.appearance_embedding column type) if pgvector is not installed.
CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
-- cameras: the 5 calibrated viewpoints (c0..c4) and their calibration.
-- ----------------------------------------------------------------------------
-- Sheet2 of the sample holds, per camera, the translation (x,y,z) and rotation
-- of the camera relative to the vehicle, plus a homography to the shared
-- bird's-eye-view (BEV) frame used for cross-view association. Values are
-- nullable because the sample only carries the headers.
CREATE TABLE cameras (
    camera_id     text PRIMARY KEY,                 -- 'c0'..'c4'
    name          text,
    -- extrinsics relative to the vehicle / rig origin
    tx            double precision,
    ty            double precision,
    tz            double precision,
    rotation      double precision[][],             -- 3x3 rotation matrix
    -- homography mapping this camera's image plane -> shared BEV plane
    homography    double precision[][],             -- 3x3
    fps           double precision,                 -- frame rate (frame<->time)
    notes         text
);

-- ----------------------------------------------------------------------------
-- vehicles: one physical vehicle, associated across views by the pipeline.
-- ----------------------------------------------------------------------------
-- This is the "identity" a query resolves to. Type/model come from the
-- open-vocabulary detector (Grounding DINO) and may be null/uncertain.
CREATE TABLE vehicles (
    vehicle_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sheet_index   integer,                          -- order in the source sheet (provenance)
    vehicle_type  text,                             -- 'SUV','Sedan','Pick Up',...
    vehicle_model text,                             -- 'Cadillac Escalade',...
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- ----------------------------------------------------------------------------
-- tracks: one camera's view of a vehicle over a frame/time range.
-- ----------------------------------------------------------------------------
-- IMPORTANT: the camera-local label (e.g. 'c0-5') is NOT globally unique --
-- the same label is reused by different vehicles with different frame ranges.
-- So the primary key is a surrogate; (camera_id, label) is intentionally NOT
-- unique. Cross-view lookups resolve by (camera_id, frame) against the range.
CREATE TABLE tracks (
    track_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    vehicle_id    bigint NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE,
    camera_id     text   NOT NULL REFERENCES cameras(camera_id),
    label         text,                             -- camera-local label, e.g. 'c1-2'
    frame_start   integer,
    frame_end     integer,
    time_start    double precision,                 -- seconds
    time_end      double precision,
    -- Note: no CHECK (frame_end >= frame_start) -- real exports contain
    -- occasional inverted ranges; validation is a reporting concern, not a
    -- load-blocking constraint.
    UNIQUE (vehicle_id, camera_id)                  -- one track per camera per vehicle
);

CREATE INDEX tracks_vehicle_idx     ON tracks (vehicle_id);
CREATE INDEX tracks_camera_label_idx ON tracks (camera_id, label);
-- Range lookup "which track in camera X contains frame N":
CREATE INDEX tracks_camera_frame_idx ON tracks (camera_id, frame_start, frame_end);

-- ----------------------------------------------------------------------------
-- detections: per-frame data. EMPTY for now -- reserved for the frame-precise
-- association built on a global BEV. Populated later from the pipeline.
-- ----------------------------------------------------------------------------
CREATE TABLE detections (
    detection_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    track_id      bigint NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
    frame         integer NOT NULL,
    time_sec      double precision,
    -- image-space bounding box (pixels)
    bbox_x        double precision,
    bbox_y        double precision,
    bbox_w        double precision,
    bbox_h        double precision,
    -- position in the shared bird's-eye-view frame (homography output)
    bev_x         double precision,
    bev_y         double precision,
    -- appearance embedding used for cross-view appearance fusion.
    -- If pgvector is unavailable, change to: appearance_embedding real[]
    appearance_embedding vector(512),
    UNIQUE (track_id, frame)
);

CREATE INDEX detections_track_frame_idx ON detections (track_id, frame);
CREATE INDEX detections_frame_idx       ON detections (frame);

COMMIT;
