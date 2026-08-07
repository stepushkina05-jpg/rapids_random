#!/usr/bin/env python3
"""
PCA module (rapids-singlecell-backed) for omnibenchmark.

Output
------
File: {output_dir}/{name}_pcas.tsv

Tab-separated, with header row:
  PC1  PC2  ...  PC{n_components}

One row per cell, prefixed by cell barcode (so R's read.table(..., header=TRUE)
auto-promotes column 1 to row.names). Values are float64.

Implementation notes
--------------------
- Genes are always centered/scaled before PCA (rsc.pp.scale, zero_center=True).
  Mirrors the scanpy module's invariant. If alternative scaling is needed
  later, expose it as a new --pca_type variant rather than as an independent
  flag (see src/cli.py for the rapids-* solver-token convention).
- ``--solver`` is a single opaque token ``rapids`` here. cuML's internal
  svd_solver knob (auto/full/jacobi) is left at its default. To compare
  cuML algorithms head-to-head, add new solver tokens (rapids-jacobi,
  rapids-full, rapids-truncated) in src/cli.py — do not expose svd_solver
  as a free-form flag.
- RMM is reinitialized with a non-managed, non-pooled allocator. This makes
  GPU memory accounting predictable for benchmark runs (a pool allocator
  would mask the true working-set cost).
"""

import sys
from pathlib import Path

import numpy as np
import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_pca_parser  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import load_matrix  # noqa: E402
from phases import phase  # noqa: E402
from writers import Embedding, write_embeddings  # noqa: E402


import hashlib
import json
import scipy.sparse as sp

def write_input_summary(
    X,
    cell_ids,
    gene_ids,
    output_file: Path,
):
    """
    Standardized PCA input summary.

    Canonical matrix for hashing:
    - cells x genes
    - float64
    - little-endian
    - row-major
    """

    if sp.issparse(X):
        X_dense = X.toarray()
        original_storage = type(X).__name__
        nnz = int(X.nnz)
    else:
        X_dense = np.asarray(X)
        original_storage = type(X).__name__
        nnz = int(np.count_nonzero(X_dense))

    X_canonical = np.asarray(
        X_dense,
        dtype="<f8",
        order="C",
    )

    cell_ids = [
        str(x) for x in cell_ids
    ]

    gene_ids = [
        str(x) for x in gene_ids
    ]

    if X_canonical.shape != (
        len(cell_ids),
        len(gene_ids),
    ):
        raise ValueError(
            "Matrix dimensions do not match IDs: "
            f"{X_canonical.shape}"
        )

    # Matrix values only.
    # Shape is checked separately in the JSON.
    matrix_sha256 = hashlib.sha256(
        X_canonical.tobytes(order="C")
    ).hexdigest()

    cell_bytes = (
        b"\0".join(
            x.encode("utf-8")
            for x in cell_ids
        )
        + b"\0"
    )

    gene_bytes = (
        b"\0".join(
            x.encode("utf-8")
            for x in gene_ids
        )
        + b"\0"
    )

    summary = {
        "schema_version": "pca_input_summary_v1",
        "orientation": "cells_x_genes",
        "canonical_dtype": "float64",
        "byte_order": "little_endian",
        "storage_order": "row_major",
        "original_storage": original_storage,

        "shape": [
            int(X_canonical.shape[0]),
            int(X_canonical.shape[1]),
        ],

        "n_cells": int(
            X_canonical.shape[0]
        ),

        "n_genes": int(
            X_canonical.shape[1]
        ),

        "nnz": nnz,

        "density": float(
            nnz / X_canonical.size
        ),

        "sum": float(
            np.sum(
                X_canonical,
                dtype=np.float64,
            )
        ),

        "sum_of_squares": float(
            np.sum(
                X_canonical * X_canonical,
                dtype=np.float64,
            )
        ),

        "minimum": float(
            np.min(X_canonical)
        ),

        "maximum": float(
            np.max(X_canonical)
        ),

        "matrix_sha256": matrix_sha256,

        "cell_ids_sha256": hashlib.sha256(
            cell_bytes
        ).hexdigest(),

        "gene_ids_sha256": hashlib.sha256(
            gene_bytes
        ).hexdigest(),
    }

    with Path(output_file).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    print(
        f"  input summary: {output_file}"
    )
    print(
        f"  matrix SHA256: {matrix_sha256}"
    )

    return summary

def run_pca(adata, args):
    """GPU randomized PCA without additional per-gene scaling."""
    rsc.pp.pca(
        adata,
        n_comps=args.n_components,
        zero_center=True,
        svd_solver="randomized",
        random_state=args.random_seed,
        n_oversamples=10,
        n_iter=2,
        dtype="float64",
        chunked=False,
    )


def main():
    args = build_pca_parser().parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "input_h5", "solver", "n_components", "random_seed"):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(args.output_dir)

    setup_gpu()

    with phase("load") as attrs:
        adata = load_matrix(args.input_h5)
        cell_ids = np.array(adata.obs_names)
        gene_ids = np.array(adata.var_names)
        attrs["n_cells"], attrs["n_genes"] = adata.shape
        print(f"  matrix (cells x genes): {adata.shape}")
        write_input_summary(
        X=adata.X,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        output_file=(
            Path(args.output_dir)
            / f"{args.name}_input_summary.json"
        ),)

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)

    with phase("compute") as attrs:
        run_pca(adata, args)
        attrs["n_components"] = args.n_components

    with phase("gpu_download"):
        rsc.get.anndata_to_CPU(adata, convert_all=True)

    with phase("write") as attrs:
        embedding = np.asarray(adata.obsm["X_pca"].get(), dtype=np.float64)
        col_names = [f"PC{i + 1}" for i in range(embedding.shape[1])]
        out = Path(args.output_dir) / f"{args.name}_pcas.tsv"
        write_embeddings(Embedding(embedding, list(cell_ids), col_names), out)
        attrs["path"] = str(out)
        print(f"  embedding: {embedding.shape}")
        print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
