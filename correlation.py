#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

EXACT_FILE = Path(
    "test_results/reference/pbmc3k_full_pcas.tsv"
)

SEEDS = range(1, 51)

RANDOMIZED_FILES = {
    seed: (
        Path("test_results")
        / "pca_random"
        / f"pbmc3k_seed_{seed}_pcas.tsv"
    )
    for seed in SEEDS
}

OUTPUT_DIR = Path(
    "test_results/pca_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CORRELATION_THRESHOLD = 0.85




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
    EXACT_FILE
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
            f"Seed {seed} and EXACT contain "
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
            f"Seed {seed} and EXACT contain "
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
    / "randomized_vs_EXACT_correlations.tsv",
    sep="\t",
    index=False,
)

summary_table = pd.DataFrame(
    summary_records
)

summary_table.to_csv(
    OUTPUT_DIR
    / "randomized_vs_EXACT_summary.tsv",
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

correlation_matrix = np.vstack(
    [
        correlation_results[seed]
        for seed in sorted(correlation_results)
    ]
)

median_correlations = np.median(
    correlation_matrix,
    axis=0,
)

minimum_correlations = np.min(
    correlation_matrix,
    axis=0,
)

maximum_correlations = np.max(
    correlation_matrix,
    axis=0,
)

plt.figure(figsize=(10, 6))

# Individual seed runs
for seed in sorted(correlation_results):
    plt.plot(
        pc_numbers,
        correlation_results[seed],
        linewidth=0.8,
        alpha=0.35,
    )

# Range across seeds
plt.fill_between(
    pc_numbers,
    minimum_correlations,
    maximum_correlations,
    alpha=0.20,
    label="Seed range",
)

# Median across seeds
plt.plot(
    pc_numbers,
    median_correlations,
    marker="o",
    markersize=3,
    linewidth=2.0,
    label="Median across 10 seeds",
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
    "Absolute correlation with EXACT"
)

plt.title(
    "Scanpy randomized PCA across 50 seeds"
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
    / "randomized_50_seeds_vs_EXACT.png"
)

plt.savefig(
    plot_file,
    dpi=300,
)

plt.close()


# --------------------------------------------------
# PLOT BREAKDOWN POINT BY SEED
# --------------------------------------------------

breakdown_plot_data = (
    summary_table
    .dropna(subset=["breakdown_pc"])
    .copy()
)

breakdown_plot_data[
    "breakdown_pc"
] = breakdown_plot_data[
    "breakdown_pc"
].astype(int)

plt.figure(figsize=(12, 6))

plt.bar(
    breakdown_plot_data["seed"],
    breakdown_plot_data["breakdown_pc"],
)

plt.xlabel("Random seed")
plt.ylabel(
    "First PC below correlation threshold"
)

plt.title(
    "Randomized PCA breakdown point by seed"
)

plt.xticks(
    breakdown_plot_data["seed"]
)

plt.ylim(
    0,
    breakdown_plot_data[
        "breakdown_pc"
    ].max() + 3,
)

plt.tight_layout()

breakdown_plot_file = (
    OUTPUT_DIR
    / "breakdown_point_by_seed.png"
)

plt.savefig(
    breakdown_plot_file,
    dpi=300,
)

plt.close()
print("\nSaved breakdown plot:")
print(breakdown_plot_file)