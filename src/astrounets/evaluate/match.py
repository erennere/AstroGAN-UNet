"""One-to-one source matching."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def match(xy_true: np.ndarray, xy_found: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices (i_true, i_found) of matched pairs: each source is used at most once, pairs are chosen
    to minimise the total distance, and no pair is farther apart than `radius` pixels."""
    xy_true, xy_found = np.asarray(xy_true, float).reshape(-1, 2), np.asarray(xy_found, float).reshape(-1, 2)
    if len(xy_true) == 0 or len(xy_found) == 0:
        return np.array([], int), np.array([], int)
    dist = cdist(xy_true, xy_found)
    cost = np.where(dist <= radius, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    keep = dist[rows, cols] <= radius
    return rows[keep], cols[keep]
