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

import cupy as cp
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
from obkit.logger import init_logger

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cli import build_pca_parser  # noqa: E402
from gpu import setup_gpu  # noqa: E402
from loaders import load_matrix  # noqa: E402
from phases import phase  # noqa: E402
from writers import Embedding, write_embeddings  # noqa: E402


def run_pca(adata, args):
    """GPU randomized PCA without additional per-gene scaling."""
    if args.solver == "rapids-randomized":    
        rsc.pp.pca(adata, n_comps=args.n_components, zero_center=True, svd_solver="randomized", random_state=args.random_seed, n_iter=args.n_iter, n_oversamples=args.n_oversamples,)
    elif args.solver == "rapids-exact":
        rsc.pp.pca(adata, n_comps=args.n_components, zero_center=True, svd_solver="covariance_eigh",)
    else:
        raise ValueError(f"Unknown solver: {args.solver}")

def main():
    args = build_pca_parser().parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    for k in ("output_dir", "name", "input_h5", "solver", "n_components", "random_seed", "n_iter", "n_oversamples",):
        print(f"  {k}: {getattr(args, k)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    init_logger(args.output_dir)

    setup_gpu()

    with phase("load") as attrs:
        adata = load_matrix(args.input_h5)
        cell_ids = np.array(adata.obs_names)
        attrs["n_cells"], attrs["n_genes"] = adata.shape
        print(f"  matrix (cells x genes): {adata.shape}")

    with phase("gpu_upload"):
        rsc.get.anndata_to_GPU(adata)
    
    cp.cuda.runtime.deviceSynchronize()
    with phase("compute") as attrs:
        run_pca(adata, args)
        cp.cuda.runtime.deviceSynchronize() 

        attrs["n_components"] = args.n_components
        attrs["solver"] = args.solver
        attrs["random_seed"] = args.random_seed

        if args.solver == "rapids-randomized":
            attrs["n_iter"] = args.n_iter
            attrs["n_oversamples"] = args.n_oversamples

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

    with phase("write_loadings") as attrs:
        loadings = np.asarray(adata.varm["PCs"].get(), dtype=np.float64)
        out_loadings = Path(args.output_dir) / f"{args.name}_loadings.tsv"
        loadings_df = pd.DataFrame(loadings, index=gene_ids, columns=col_names)
        loadings_df.index.name = "gene"
        loadings_df.to_csv(out_loadings, sep="\t")
        attrs["path"] = str(out_loadings)
        print(f"  loadings: {loadings.shape}")
        print(f"  wrote: {out_loadings}")

if __name__ == "__main__":
    main()




