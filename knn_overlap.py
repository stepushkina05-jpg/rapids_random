#!/usr/bin/env python3

from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp


# --------------------------------------------------
# PATHS
# --------------------------------------------------

GRAPH_A = Path(
    "test_results/knn_random/pbmc3k_seed_20_knn.h5"
)

GRAPH_B = Path(
    "test_results/knn_random/pbmc3k_seed_32_knn.h5"
)


# --------------------------------------------------
# LOAD GRAPH
# --------------------------------------------------

def load_distances(path: Path):
    with h5py.File(path, "r") as h5:
        g = h5["distances"]

        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]

        if "shape" in g:
            shape = tuple(g["shape"][:])
        else:
            n = len(h5["cell_ids"])
            shape = (n, n)

        cell_ids = h5["cell_ids"][:].astype(str)

    matrix = sp.csr_matrix(
        (data, indices, indptr),
        shape=shape,
    )

    return matrix, cell_ids


A, ids_a = load_distances(GRAPH_A)
B, ids_b = load_distances(GRAPH_B)


# --------------------------------------------------
# CHECK CELL ORDER
# --------------------------------------------------

if not np.array_equal(ids_a, ids_b):
    raise ValueError(
        "Cell IDs or cell order differ between graphs."
    )


# --------------------------------------------------
# COMPARE NEIGHBORS
# --------------------------------------------------

shared_fraction = []
jaccard_scores = []

for i in range(A.shape[0]):

    neighbors_a = set(
        A.indices[
            A.indptr[i]:
            A.indptr[i + 1]
        ]
    )

    neighbors_b = set(
        B.indices[
            B.indptr[i]:
            B.indptr[i + 1]
        ]
    )

    intersection = (
        neighbors_a
        & neighbors_b
    )

    union = (
        neighbors_a
        | neighbors_b
    )

    shared_fraction.append(
        len(intersection)
        / len(neighbors_a)
    )

    jaccard_scores.append(
        len(intersection)
        / len(union)
    )


shared_fraction = np.asarray(
    shared_fraction
)

jaccard_scores = np.asarray(
    jaccard_scores
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print()
print("kNN overlap summary")
print("-------------------")

print(
    "Mean shared-neighbor fraction:",
    shared_fraction.mean(),
)

print(
    "Median shared-neighbor fraction:",
    np.median(shared_fraction),
)

print(
    "Minimum shared-neighbor fraction:",
    shared_fraction.min(),
)

print()

print(
    "Mean Jaccard:",
    jaccard_scores.mean(),
)

print(
    "Median Jaccard:",
    np.median(jaccard_scores),
)

print(
    "Minimum Jaccard:",
    jaccard_scores.min(),
)