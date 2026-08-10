#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles


# --------------------------------------------------
# PATHS
# --------------------------------------------------

Exact_FILE = Path(
    "test_results/reference/pbmc3k_full_pcas.tsv"
)

RANDOMIZED_DIR = Path(
    "test_results/pca_random"
)

OUTPUT_DIR = Path(
    "test_results/principal_angles"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SEEDS = [20, 32]

K_VALUES = [
    5,
    10,
    20,
    30,
    40,
    50,
]


# --------------------------------------------------
# LOAD SCORES
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
    Exact_FILE
)


# --------------------------------------------------
# CALCULATE PRINCIPAL ANGLES
# --------------------------------------------------

records = []

for seed in SEEDS:
    randomized_file = (
        RANDOMIZED_DIR
        / f"pbmc3k_seed_{seed}_pcas.tsv"
    )

    randomized_scores = load_scores(
        randomized_file
    )

    if set(reference_scores.index) != set(
        randomized_scores.index
    ):
        raise ValueError(
            f"Seed {seed} and Exact contain "
            "different cell IDs."
        )

    randomized_scores = randomized_scores.loc[
        reference_scores.index
    ]

    for k in K_VALUES:
        if k > reference_scores.shape[1]:
            raise ValueError(
                f"k={k} exceeds available PCs."
            )

        reference_subspace = (
            reference_scores.iloc[:, :k]
            .to_numpy(dtype=np.float64)
        )

        randomized_subspace = (
            randomized_scores.iloc[:, :k]
            .to_numpy(dtype=np.float64)
        )

        angles_radians = subspace_angles(
            reference_subspace,
            randomized_subspace,
        )

        angles_degrees = np.degrees(
            angles_radians
        )

        records.append(
            {
                "seed": seed,
                "k": k,
                "maximum_angle_degrees": (
                    float(
                        np.max(
                            angles_degrees
                        )
                    )
                ),
                "mean_angle_degrees": (
                    float(
                        np.mean(
                            angles_degrees
                        )
                    )
                ),
                "median_angle_degrees": (
                    float(
                        np.median(
                            angles_degrees
                        )
                    )
                ),
                "minimum_angle_degrees": (
                    float(
                        np.min(
                            angles_degrees
                        )
                    )
                ),
            }
        )
        
results = pd.DataFrame(
    records
)

print(
    results[
        [
            "seed",
            "k",
            "maximum_angle_degrees",
            "mean_angle_degrees",
            "median_angle_degrees",
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# SAVE TABLES
# --------------------------------------------------

results.to_csv(
    OUTPUT_DIR
    / "principal_angles_by_seed.tsv",
    sep="\t",
    index=False,
)

summary = (
    results.groupby("k")
    .agg(
        mean_max_angle=(
            "maximum_angle_degrees",
            "mean",
        ),
        median_max_angle=(
            "maximum_angle_degrees",
            "median",
        ),
        min_max_angle=(
            "maximum_angle_degrees",
            "min",
        ),
        max_max_angle=(
            "maximum_angle_degrees",
            "max",
        ),
        mean_mean_angle=(
            "mean_angle_degrees",
            "mean",
        ),
    )
    .reset_index()
)   
    

summary.to_csv(
    OUTPUT_DIR
    / "principal_angles_summary.tsv",
    sep="\t",
    index=False,
)

# --------------------------------------------------
# PLOT
# --------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

for seed in SEEDS:
    seed_data = results[
        results["seed"] == seed
    ]

    plt.plot(
        seed_data["k"],
        seed_data[
            "maximum_angle_degrees"
        ],
        marker="o",
        linewidth=1,
        alpha=0.5,
    )

median_curve = (
    results.groupby("k")[
        "maximum_angle_degrees"
    ]
    .median()
)

plt.plot(
    median_curve.index,
    median_curve.values,
    marker="o",
    linewidth=2.5,
    label="Median across 10 seeds",
)

plt.xlabel(
    "Subspace dimension k"
)

plt.ylabel(
    "Largest principal angle (degrees)"
)

plt.title(
    "Principal-angle stability of Rapids randomized PCA"
)

plt.xticks(
    K_VALUES
)

plt.ylim(
    bottom=0
)

plt.legend()
plt.tight_layout()

plot_file = (
    OUTPUT_DIR
    / "principal_angles_rapids.png"
)

plt.savefig(
    plot_file,
    dpi=300,
)

plt.close()


for k in K_VALUES:
    subset = results[results["k"] == k]

    print(
        f"k={k}:",
        "median maximum angle =",
        subset["maximum_angle_degrees"].median(),
        "median mean angle =",
        subset["mean_angle_degrees"].median(),
    )
# --------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------

print(
    "\nPrincipal-angle summary:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nSaved detailed results:"
)

print(
    OUTPUT_DIR
    / "principal_angles_by_seed.tsv"
)

print(
    "\nSaved plot:"
)

print(
    plot_file
)

