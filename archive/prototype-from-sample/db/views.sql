-- ============================================================================
-- Cross-view association: views + functions
-- ============================================================================
-- The core question this DB answers:
--   "A vehicle appears at frame N in camera X (e.g. a Kia Forte in c0). What
--    are the corresponding tracks / frame ranges in c1..c4?"
--
-- Resolution strategy (track-level, per the current pipeline stage):
--   1. Find the track in camera X whose [frame_start, frame_end] contains N
--      -- or match by camera-local label directly.
--   2. That track belongs to exactly one vehicle.
--   3. Return every track of that vehicle (all cameras), which gives the
--      corresponding frame/time ranges in the other views.
-- ============================================================================

BEGIN;

-- Flat, human-readable view of every vehicle/track pair.
CREATE OR REPLACE VIEW v_vehicle_tracks AS
SELECT
    v.vehicle_id,
    v.vehicle_type,
    v.vehicle_model,
    t.track_id,
    t.camera_id,
    t.label,
    t.frame_start,
    t.frame_end,
    t.time_start,
    t.time_end
FROM vehicles v
JOIN tracks   t ON t.vehicle_id = v.vehicle_id
ORDER BY v.vehicle_id, t.camera_id;

-- One row per vehicle with its per-camera tracks pivoted into columns --
-- mirrors the layout of the source spreadsheet.
CREATE OR REPLACE VIEW v_vehicle_matrix AS
SELECT
    v.vehicle_id,
    v.vehicle_type,
    v.vehicle_model,
    MAX(t.label) FILTER (WHERE t.camera_id = 'c0') AS c0_label,
    MAX(t.label) FILTER (WHERE t.camera_id = 'c1') AS c1_label,
    MAX(t.label) FILTER (WHERE t.camera_id = 'c2') AS c2_label,
    MAX(t.label) FILTER (WHERE t.camera_id = 'c3') AS c3_label,
    MAX(t.label) FILTER (WHERE t.camera_id = 'c4') AS c4_label
FROM vehicles v
LEFT JOIN tracks t ON t.vehicle_id = v.vehicle_id
GROUP BY v.vehicle_id, v.vehicle_type, v.vehicle_model
ORDER BY v.vehicle_id;

-- Return all tracks belonging to a given vehicle (the cross-view answer).
CREATE OR REPLACE FUNCTION cross_view_for_vehicle(p_vehicle_id bigint)
RETURNS TABLE (
    vehicle_id    bigint,
    vehicle_type  text,
    vehicle_model text,
    camera_id     text,
    label         text,
    frame_start   integer,
    frame_end     integer,
    time_start    double precision,
    time_end      double precision
) AS $$
    SELECT v.vehicle_id, v.vehicle_type, v.vehicle_model,
           t.camera_id, t.label, t.frame_start, t.frame_end,
           t.time_start, t.time_end
    FROM vehicles v
    JOIN tracks   t ON t.vehicle_id = v.vehicle_id
    WHERE v.vehicle_id = p_vehicle_id
    ORDER BY t.camera_id;
$$ LANGUAGE sql STABLE;

-- Resolve by (camera, frame): find the track in that camera whose range
-- contains the frame, then return the whole vehicle's cross-view tracks.
-- If several tracks in the camera overlap the frame, the vehicle(s) are all
-- returned (disambiguate with the label variant below).
CREATE OR REPLACE FUNCTION cross_view_by_frame(p_camera text, p_frame integer)
RETURNS TABLE (
    vehicle_id    bigint,
    vehicle_type  text,
    vehicle_model text,
    camera_id     text,
    label         text,
    frame_start   integer,
    frame_end     integer,
    time_start    double precision,
    time_end      double precision
) AS $$
    SELECT cv.*
    FROM (
        SELECT DISTINCT t.vehicle_id
        FROM tracks t
        WHERE t.camera_id = p_camera
          AND t.frame_start IS NOT NULL AND t.frame_end IS NOT NULL
          AND p_frame BETWEEN LEAST(t.frame_start, t.frame_end)
                          AND GREATEST(t.frame_start, t.frame_end)
    ) hit
    CROSS JOIN LATERAL cross_view_for_vehicle(hit.vehicle_id) cv
    ORDER BY cv.vehicle_id, cv.camera_id;
$$ LANGUAGE sql STABLE;

-- Resolve by camera-local label (e.g. 'c0-5'). Because labels are reused,
-- pass an optional frame to disambiguate; when null, every vehicle carrying
-- that label is returned.
CREATE OR REPLACE FUNCTION cross_view_by_label(p_label text, p_frame integer DEFAULT NULL)
RETURNS TABLE (
    vehicle_id    bigint,
    vehicle_type  text,
    vehicle_model text,
    camera_id     text,
    label         text,
    frame_start   integer,
    frame_end     integer,
    time_start    double precision,
    time_end      double precision
) AS $$
    SELECT cv.*
    FROM (
        SELECT DISTINCT t.vehicle_id
        FROM tracks t
        WHERE t.label = p_label
          AND (
                p_frame IS NULL
             OR (t.frame_start IS NOT NULL AND t.frame_end IS NOT NULL
                 AND p_frame BETWEEN LEAST(t.frame_start, t.frame_end)
                                 AND GREATEST(t.frame_start, t.frame_end))
              )
    ) hit
    CROSS JOIN LATERAL cross_view_for_vehicle(hit.vehicle_id) cv
    ORDER BY cv.vehicle_id, cv.camera_id;
$$ LANGUAGE sql STABLE;

COMMIT;
