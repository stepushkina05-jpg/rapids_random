#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ARPACK_FILE = Path("test_results") / "reference" / "pbmc3k_arpack_pcas.tsv"

RANDOMIZED_FILES = {
    1: Path("test_results") / "pca_random"/ "pbmc3k_seed_1_pcas.tsv",
    2: Path("test_results") / "pca_random"/ "pbmc3k_seed_2_pcas.tsv",
}
OUTPUT_DIR = (
    Path("test_results")
    / "pca_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CORRELATION_THRESHOLD = 0.90


# --------------------------------------------------
# LOAD PCA SCORES
# --------------------------------------------------

def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"PCA file not found: {path.resolve()}"
        )

    table = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
    )

    table.index = table.index.astype(str)
    table.index.name = "cell_id"

    return table.apply(
        pd.to_numeric,
        errors="raise",
    )


reference_scores = load_scores(
    ARPACK_FILE
)


# --------------------------------------------------
# CALCULATE CORRELATIONS
# --------------------------------------------------

correlation_results = {}
summary_records = []

for seed, randomized_file in (
    RANDOMIZED_FILES.items()
):
    randomized_scores = load_scores(
        randomized_file
    )

    if set(reference_scores.index) != set(
        randomized_scores.index
    ):
        raise ValueError(
            f"Seed {seed} and ARPACK contain "
            "different cell IDs."
        )

    # Make cell order identical
    randomized_scores = randomized_scores.loc[
        reference_scores.index
    ]

    if list(reference_scores.columns) != list(
        randomized_scores.columns
    ):
        raise ValueError(
            f"Seed {seed} and ARPACK contain "
            "different PC columns."
        )

    correlations = []

    for pc_name in reference_scores.columns:
        correlation = np.corrcoef(
            reference_scores[
                pc_name
            ].to_numpy(),
            randomized_scores[
                pc_name
            ].to_numpy(),
        )[0, 1]

        correlations.append(
            abs(correlation)
        )

    correlations = np.asarray(
        correlations,
        dtype=np.float64,
    )

    correlation_results[seed] = correlations

    below_threshold = np.flatnonzero(
        correlations
        < CORRELATION_THRESHOLD
    )

    if len(below_threshold) == 0:
        breakdown_pc = None
    else:
        breakdown_pc = int(
            below_threshold[0] + 1
        )

    summary_records.append(
        {
            "seed": seed,
            "mean_absolute_correlation": (
                correlations.mean()
            ),
            "minimum_absolute_correlation": (
                correlations.min()
            ),
            "breakdown_pc": breakdown_pc,
            "threshold": (
                CORRELATION_THRESHOLD
            ),
        }
    )


# --------------------------------------------------
# SAVE CORRELATION TABLE
# --------------------------------------------------

pc_names = list(
    reference_scores.columns
)

correlation_table = pd.DataFrame(
    {
        "pc": pc_names,
        **{
            f"seed_{seed}": correlations
            for seed, correlations
            in correlation_results.items()
        },
    }
)

correlation_table.to_csv(
    OUTPUT_DIR
    / "randomized_vs_arpack_correlations.tsv",
    sep="\t",
    index=False,
)

summary_table = pd.DataFrame(
    summary_records
)

summary_table.to_csv(
    OUTPUT_DIR
    / "randomized_vs_arpack_summary.tsv",
    sep="\t",
    index=False,
)


# --------------------------------------------------
# PLOT
# --------------------------------------------------

pc_numbers = np.arange(
    1,
    len(pc_names) + 1,
)

plt.figure(figsize=(10, 6))

for seed, correlations in (
    correlation_results.items()
):
    plt.plot(
        pc_numbers,
        correlations,
        marker="o",
        markersize=3,
        linewidth=1.3,
        label=f"Seed {seed}",
    )

plt.axhline(
    CORRELATION_THRESHOLD,
    linestyle="--",
    linewidth=1.3,
    label=(
        f"Threshold "
        f"{CORRELATION_THRESHOLD:.2f}"
    ),
)

plt.xlabel("Principal component")
plt.ylabel(
    "Absolute correlation with ARPACK"
)
plt.title(
    "Randomized PCA compared with ARPACK reference"
)

plt.ylim(0, 1.02)
plt.xlim(1, len(pc_names))

plt.xticks(
    np.arange(
        1,
        len(pc_names) + 1,
        5,
    )
)

plt.legend()
plt.tight_layout()

plot_file = (
    OUTPUT_DIR
    / "randomized_vs_arpack_correlations.png"
)

plt.savefig(
    plot_file,
    dpi=300,
)

plt.close()


# --------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------

print("\nCorrelation summary:")
print(
    summary_table.to_string(
        index=False
    )
)

print("\nSaved correlation table:")
print(
    OUTPUT_DIR
    / "randomized_vs_arpack_correlations.tsv"
)

print("\nSaved summary:")
print(
    OUTPUT_DIR
    / "randomized_vs_arpack_summary.tsv"
)

print("\nSaved plot:")
print(plot_file)