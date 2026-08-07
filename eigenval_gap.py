#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

Exact_FILE = Path(
    "test_results/reference/pbmc3k_full_pcas.tsv"
)

CORRELATION_FILE = Path(
    "test_results/pca_comparison/"
    "randomized_vs_EXACT_correlations.tsv"
)

OUTPUT_DIR = Path(
    "test_results/eigenvalue_gaps"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# LOAD PCA SCORES
# --------------------------------------------------

def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    table = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
    )

    table.index = table.index.astype(str)

    return table.apply(
        pd.to_numeric,
        errors="raise",
    )


scores = load_scores(
    Exact_FILE
)


# --------------------------------------------------
# CALCULATE REFERENCE EIGENVALUES
# --------------------------------------------------

# For PCA scores, the sample variance of PC j
# equals the corresponding PCA eigenvalue.
eigenvalues = scores.var(
    axis=0,
    ddof=1,
).to_numpy(dtype=np.float64)

pc_numbers = np.arange(
    1,
    len(eigenvalues) + 1,
)

# Gap after PC j:
# lambda_j - lambda_{j+1}
absolute_gaps = (
    eigenvalues[:-1]
    - eigenvalues[1:]
)

relative_gaps = (
    absolute_gaps
    / eigenvalues[:-1]
)

gap_table = pd.DataFrame(
    {
        "pc": pc_numbers,
        "eigenvalue": eigenvalues,
        "absolute_gap_after_pc": np.append(
            absolute_gaps,
            np.nan,
        ),
        "relative_gap_after_pc": np.append(
            relative_gaps,
            np.nan,
        ),
    }
)


# --------------------------------------------------
# ADD RANDOMIZED-PCA STABILITY
# --------------------------------------------------

if CORRELATION_FILE.exists():
    correlations = pd.read_csv(
        CORRELATION_FILE,
        sep="\t",
    )

    seed_columns = [
        column
        for column in correlations.columns
        if column.startswith("seed_")
    ]

    correlation_matrix = correlations[
        seed_columns
    ].to_numpy(dtype=np.float64)

    gap_table[
        "median_absolute_correlation"
    ] = np.median(
        correlation_matrix,
        axis=1,
    )

    gap_table[
        "minimum_absolute_correlation"
    ] = np.min(
        correlation_matrix,
        axis=1,
    )

    gap_table[
        "correlation_instability"
    ] = (
        1.0
        - gap_table[
            "median_absolute_correlation"
        ]
    )


# --------------------------------------------------
# SAVE TABLE
# --------------------------------------------------

output_table = (
    OUTPUT_DIR
    / "Exact_eigenvalue_gaps.tsv"
)

gap_table.to_csv(
    output_table,
    sep="\t",
    index=False,
)


# --------------------------------------------------
# PLOT 1: EIGENVALUE SPECTRUM
# --------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    pc_numbers,
    eigenvalues,
    marker="o",
    markersize=3,
)

plt.xlabel(
    "Principal component"
)

plt.ylabel(
    "Exact eigenvalue"
)

plt.title(
    "Exact PCA eigenvalue spectrum"
)

plt.xticks(
    np.arange(
        1,
        len(eigenvalues) + 1,
        5,
    )
)

plt.xlim(
    1,
    len(eigenvalues),
)

plt.tight_layout()

spectrum_plot = (
    OUTPUT_DIR
    / "Exact_eigenvalue_spectrum.png"
)

plt.savefig(
    spectrum_plot,
    dpi=300,
)

plt.close()


# --------------------------------------------------
# PLOT 2: RELATIVE EIGENVALUE GAPS
# --------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    pc_numbers[:-1],
    relative_gaps,
    marker="o",
    markersize=3,
)

plt.xlabel(
    "Gap after principal component"
)

plt.ylabel(
    "Relative eigenvalue gap"
)

plt.title(
    "Relative gaps in the Exact eigenvalue spectrum"
)

plt.xticks(
    np.arange(
        1,
        len(eigenvalues),
        5,
    )
)

plt.xlim(
    1,
    len(eigenvalues) - 1,
)

plt.tight_layout()

gap_plot = (
    OUTPUT_DIR
    / "Exact_relative_eigenvalue_gaps.png"
)

plt.savefig(
    gap_plot,
    dpi=300,
)

plt.close()


# --------------------------------------------------
# PLOT 3: GAP VERSUS INSTABILITY
# --------------------------------------------------

scatter_plot = None

if "correlation_instability" in gap_table:
    usable = gap_table.iloc[:-1].copy()

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        usable["relative_gap_after_pc"],
        usable["correlation_instability"],
    )

    for _, row in usable.iterrows():
        pc = int(row["pc"])

        # Label later PCs and especially unstable PCs.
        if (
            pc >= 25
            or row["correlation_instability"] > 0.1
        ):
            plt.annotate(
                str(pc),
                (
                    row["relative_gap_after_pc"],
                    row["correlation_instability"],
                ),
                fontsize=8,
            )

    plt.xlabel(
        "Relative eigenvalue gap after PC"
    )

    plt.ylabel(
        "Instability: 1 - median absolute correlation"
    )

    plt.title(
        "Eigenvalue gaps versus randomized-PCA instability"
    )

    plt.tight_layout()

    scatter_plot = (
        OUTPUT_DIR
        / "gap_vs_correlation_instability.png"
    )

    plt.savefig(
        scatter_plot,
        dpi=300,
    )

    plt.close()


# --------------------------------------------------
# NUMERICAL ASSOCIATION
# --------------------------------------------------

if "correlation_instability" in gap_table:
    usable = gap_table.iloc[:-1].dropna(
        subset=[
            "relative_gap_after_pc",
            "correlation_instability",
        ]
    )

    pearson = usable[
        [
            "relative_gap_after_pc",
            "correlation_instability",
        ]
    ].corr(
        method="pearson"
    ).iloc[0, 1]

    spearman = usable[
        [
            "relative_gap_after_pc",
            "correlation_instability",
        ]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    print(
        "\nAssociation between eigenvalue gap "
        "and instability:"
    )

    print(
        f"Pearson correlation: {pearson:.4f}"
    )

    print(
        f"Spearman correlation: {spearman:.4f}"
    )


# --------------------------------------------------
# PRINT SMALLEST GAPS
# --------------------------------------------------

smallest_gaps = (
    gap_table.iloc[:-1]
    .sort_values(
        "relative_gap_after_pc"
    )
    .head(10)
)

print(
    "\nTen smallest relative eigenvalue gaps:"
)

columns_to_print = [
    "pc",
    "eigenvalue",
    "relative_gap_after_pc",
]

if "median_absolute_correlation" in smallest_gaps:
    columns_to_print.append(
        "median_absolute_correlation"
    )

print(
    smallest_gaps[
        columns_to_print
    ].to_string(
        index=False
    )
)

print(
    f"\nSaved table: {output_table}"
)

print(
    f"Saved spectrum plot: {spectrum_plot}"
)

print(
    f"Saved gap plot: {gap_plot}"
)

if scatter_plot is not None:
    print(
        f"Saved gap-instability plot: "
        f"{scatter_plot}"
    )