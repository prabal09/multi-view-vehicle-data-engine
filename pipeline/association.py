"""Step 5: cross-view association -- group per-camera tracks into physical vehicles.

Fuses appearance (embedding cosine) with a hard constraint: a physical vehicle
appears AT MOST ONCE per camera. Agglomerative average-linkage: repeatedly merge
the most-similar pair of clusters that don't already share a camera, until the
best similarity drops below a threshold.

The per-camera constraint is what makes this robust despite imperfect appearance:
two different vehicles that each span the same set of cameras can never merge
(they'd put two tracks from one camera in one vehicle), so the clustering
naturally stops at one cluster per physical vehicle.

Pure NumPy -- runs on the embeddings already stored in the track JSONs.
"""

from __future__ import annotations

import numpy as np


def associate(track_keys, cameras, embeddings, sim_thresh: float = 0.5):
    """Cluster tracks into physical vehicles.

    track_keys:  list of hashable ids (one per track)
    cameras:     camera of each track (same order)
    embeddings:  (N, D) array, one appearance vector per track
    returns:     list of clusters, each a list of track_keys
    """
    n = len(track_keys)
    E = np.asarray(embeddings, dtype=float)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    S = E @ E.T                                   # cosine similarity

    clusters = [[i] for i in range(n)]            # start: every track alone
    cams = [{cameras[i]} for i in range(n)]

    def linkage(a, b):                            # average-linkage similarity
        return float(np.mean([S[i, j] for i in clusters[a] for j in clusters[b]]))

    while True:
        best, best_sim = None, sim_thresh
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if cams[a] & cams[b]:             # already share a camera -> skip
                    continue
                s = linkage(a, b)
                if s > best_sim:
                    best, best_sim = (a, b), s
        if best is None:
            break
        a, b = best
        clusters[a] += clusters[b]
        cams[a] |= cams[b]
        del clusters[b], cams[b]

    return [[track_keys[i] for i in c] for c in clusters]
