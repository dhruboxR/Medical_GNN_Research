
"""
PHASE 3B.25
Geometry-Conditioned Cross-Scanner Graph Stability

Reconstruct Phase 3B.21 connected graphs and compare
observed residual-edge overlap with an exact null
conditional on each scanner's edge-distance counts.

Controls:
    - Frozen 10-region cohort
    - Same 35-edge spatial MST
    - Same residual-edge budgets
    - Same candidate edge pool
    - AT2/GT450 squared-distance edge histograms

Null model:
    Independent uniform sampling within each
    exact squared-distance stratum.

Calculates:
    - Observed shared residual edges
    - Geometry-conditioned expected overlap
    - Exact standard deviation
    - Excess overlap
    - Conditional chance-adjusted overlap
    - Exact one-sided tail probability
    - Paired spatial-minus-global differences

No network calls, trained GNN, or data modifications.
Tail probabilities describe this conditional null,
not independent biological replication.
"""

from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

MANIFEST = BASE / (
    "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

FEATURES = BASE / (
    "outputs/features/"
    "plism_phase3b_dinov2_features.npz"
)

PREVIOUS = BASE / (
    "outputs/tables/"
    "plism_phase3b_connected_graph_per_region.csv"
)

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"

DETAIL = TABLES / (
    "plism_phase3b_geometry_conditioned_per_region.csv"
)

STRATA = TABLES / (
    "plism_phase3b_geometry_distance_strata.csv"
)

SUMMARY = TABLES / (
    "plism_phase3b_geometry_conditioned_summary.csv"
)

PAIRED = TABLES / (
    "plism_phase3b_geometry_conditioned_paired.csv"
)

REPORT = REPORTS / (
    "25_geometry_conditioned_graph_stability.json"
)

EXPECTED_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

BUDGETS = (54, 72, 90)
METHODS = ("global", "spatial")

for output in (DETAIL, STRATA, SUMMARY, PAIRED, REPORT):
    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite previous result: {output}"
        )


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine_matrix(features):
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)

    if np.any(norms <= 1e-12):
        raise RuntimeError("Zero-norm embedding.")

    x = x / norms
    return x @ x.T


def candidate_edges(rows, cols, method):
    result = []

    for a, b in combinations(range(len(rows)), 2):
        dx = abs(int(rows[a]) - int(rows[b]))
        dy = abs(int(cols[a]) - int(cols[b]))

        if method == "spatial" and max(dx, dy) != 1:
            continue

        result.append((a, b))

    return result


def spatial_mst(rows, cols):
    # Exact ordering from Phase 3B.21.
    edges = candidate_edges(rows, cols, "spatial")

    edges.sort(
        key=lambda e: (
            (int(rows[e[0]]) - int(rows[e[1]])) ** 2
            + (int(cols[e[0]]) - int(cols[e[1]])) ** 2,
            e[0],
            e[1],
        )
    )

    parent = list(range(len(rows)))
    sizes = [1] * len(rows)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a = find(a)
        b = find(b)

        if a == b:
            return False

        if sizes[a] < sizes[b]:
            a, b = b, a

        parent[b] = a
        sizes[a] += sizes[b]
        return True

    tree = set()

    for a, b in edges:
        if union(a, b):
            tree.add((a, b))

        if len(tree) == len(rows) - 1:
            break

    if len(tree) != len(rows) - 1:
        raise RuntimeError("Disconnected spanning tree.")

    return tree


def select_residual(similarity, candidates, backbone, count):
    pool = [e for e in candidates if e not in backbone]

    ranked = sorted(
        pool,
        key=lambda e: (
            -float(similarity[e[0], e[1]]),
            e[0],
            e[1],
        ),
    )

    if count > len(ranked):
        raise RuntimeError("Residual budget too large.")

    return set(ranked[:count]), pool


def distance_squared(edge, rows, cols):
    a, b = edge
    dx = int(rows[a]) - int(rows[b])
    dy = int(cols[a]) - int(cols[b])
    return dx * dx + dy * dy


def geometry_conditioned_null(
    pool, selected_a, selected_b, rows, cols
):
    """
    Exact distribution of common selected edges
    given each scanner's counts within every
    squared-distance stratum.

    Within stratum d:
        N = available candidate edges
        a = AT2 selected edges
        b = GT450 selected edges

    Shared count follows Hypergeom(N, a, b).

    Strata are independent under the defined null,
    and their overlap-count distributions convolve.
    """

    pools = defaultdict(set)

    for edge in pool:
        d2 = distance_squared(edge, rows, cols)
        pools[d2].add(edge)

    counts_a = Counter(
        distance_squared(e, rows, cols)
        for e in selected_a
    )

    counts_b = Counter(
        distance_squared(e, rows, cols)
        for e in selected_b
    )

    observed = len(selected_a & selected_b)

    # Probability mass for total shared edge count.
    pmf_total = np.array([1.0], dtype=np.float64)

    expected = 0.0
    maximum = 0

    stratum_records = []

    for d2 in sorted(pools):
        N = len(pools[d2])
        a = counts_a[d2]
        b = counts_b[d2]

        if a > N or b > N:
            raise RuntimeError("Invalid distance counts.")

        observed_d = len(
            (selected_a & pools[d2])
            & (selected_b & pools[d2])
        )

        lower = max(0, a + b - N)
        upper = min(a, b)

        support = np.arange(lower, upper + 1)

        probabilities = hypergeom.pmf(
            support, N, a, b
        )

        # Include zeros for impossible lower counts.
        stratum_pmf = np.zeros(
            upper + 1, dtype=np.float64
        )

        stratum_pmf[support] = probabilities

        pmf_total = np.convolve(
            pmf_total, stratum_pmf
        )

        expected_d = a * b / N

        variance_d = (
            a * b * (N - a) * (N - b)
            / (N * N * (N - 1))
            if N > 1 else 0.0
        )

        expected += expected_d
        maximum += upper

        stratum_records.append({
            "distance_squared": d2,
            "candidate_edges": N,
            "at2_selected": a,
            "gt450_selected": b,
            "observed_shared": observed_d,
            "expected_shared": expected_d,
            "null_variance": variance_d,
        })

    pmf_total = np.maximum(pmf_total, 0.0)
    probability_sum = pmf_total.sum()

    if not np.isclose(
        probability_sum, 1.0, atol=1e-10
    ):
        raise RuntimeError(
            "Conditional null PMF not normalized."
        )

    pmf_total /= probability_sum

    support = np.arange(len(pmf_total))

    exact_mean = float(
        np.sum(support * pmf_total)
    )

    exact_variance = float(
        np.sum(
            (support - exact_mean) ** 2
            * pmf_total
        )
    )

    if not np.isclose(
        expected, exact_mean, atol=1e-7
    ):
        raise RuntimeError(
            "Null expectation mismatch."
        )

    if sum(
        rec["observed_shared"]
        for rec in stratum_records
    ) != observed:
        raise RuntimeError(
            "Stratum overlap mismatch."
        )

    null_sd = float(
        np.sqrt(max(0.0, exact_variance))
    )

    upper_tail_probability = float(
        pmf_total[observed:].sum()
    )

    excess_overlap = observed - expected

    # Normalize by the largest overlap attainable
    # given BOTH scanner-specific distance profiles.
    available_excess = maximum - expected

    if available_excess > 1e-12:
        conditional_adjusted = (
            excess_overlap / available_excess
        )
    else:
        conditional_adjusted = float("nan")

    if null_sd > 1e-12:
        null_z = excess_overlap / null_sd
    else:
        null_z = float("nan")

    return {
        "observed_shared": observed,
        "expected_shared": expected,
        "null_standard_deviation": null_sd,
        "excess_shared": excess_overlap,
        "maximum_conditional_shared": maximum,
        "conditional_adjusted": conditional_adjusted,
        "null_z": null_z,
        "exact_upper_tail_probability":
            upper_tail_probability,
    }, stratum_records


# ============================================================
# STEP 1: VERIFY INPUTS
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B.25 - GEOMETRY-CONDITIONED STABILITY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying inputs...", flush=True)

for path in (MANIFEST, FEATURES, PREVIOUS):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

if file_hash(MANIFEST) != EXPECTED_HASH:
    raise RuntimeError("Frozen manifest hash mismatch.")

manifest = pd.read_csv(
    MANIFEST,
    dtype={"tile_id": str, "region_id": str}
).sort_values(
    ["region_id", "level", "x", "y", "tile_id"]
).reset_index(drop=True)

with np.load(FEATURES, allow_pickle=False) as data:
    at2 = data["at2"].copy()
    gt450 = data["gt450"].copy()
    tile_ids = data["tile_ids"].astype(str)
    region_ids = data["region_ids"].astype(str)
    rows = data["grid_rows"].astype(int)
    cols = data["grid_cols"].astype(int)

if at2.shape != (360, 384) or gt450.shape != (360, 384):
    raise RuntimeError("Unexpected feature dimensions.")

if not np.isfinite(at2).all() or not np.isfinite(gt450).all():
    raise RuntimeError("Non-finite features.")

for actual, expected_values in (
    (tile_ids, manifest["tile_id"].to_numpy(dtype=str)),
    (region_ids, manifest["region_id"].to_numpy(dtype=str)),
    (rows, manifest["x"].to_numpy(dtype=int)),
    (cols, manifest["y"].to_numpy(dtype=int)),
):
    if not np.array_equal(actual, expected_values):
        raise RuntimeError(
            "Feature/manifest alignment mismatch."
        )

previous = pd.read_csv(PREVIOUS)

if len(previous) != 60:
    raise RuntimeError(
        "Phase 3B.21 summary has unexpected size."
    )

if previous.duplicated(
    ["region_id", "graph_type", "edge_budget"]
).any():
    raise RuntimeError(
        "Duplicate Phase 3B.21 experiment keys."
    )

previous_index = previous.set_index(
    ["region_id", "graph_type", "edge_budget"]
)

regions = sorted(set(region_ids))

if len(regions) != 10:
    raise RuntimeError("Expected ten regions.")

print("Frozen manifest: VERIFIED")
print("DINOv2 features: VERIFIED")
print("Previous connected graphs: AVAILABLE")
print("Regions:", len(regions))


# ============================================================
# STEP 2: RECONSTRUCT AND AUDIT CONNECTED GRAPHS
# ============================================================

print("\n[2] Evaluating distance-conditioned overlap...", flush=True)

detail_records = []
stratum_records = []

for region_id in regions:
    indices = np.where(region_ids == region_id)[0]

    if len(indices) != 36:
        raise RuntimeError("Unexpected region size.")

    r = rows[indices]
    c = cols[indices]

    coordinates = set(
        zip(r.tolist(), c.tolist())
    )

    expected_coordinates = {
        (x, y)
        for x in range(int(r.min()), int(r.min()) + 6)
        for y in range(int(c.min()), int(c.min()) + 6)
    }

    if coordinates != expected_coordinates:
        raise RuntimeError(
            f"Invalid region geometry: {region_id}"
        )

    backbone = spatial_mst(r, c)

    sim_a = cosine_matrix(at2[indices])
    sim_b = cosine_matrix(gt450[indices])

    for method in METHODS:
        candidates = candidate_edges(r, c, method)

        for budget in BUDGETS:
            residual_count = budget - len(backbone)

            selected_a, pool = select_residual(
                sim_a, candidates, backbone,
                residual_count
            )

            selected_b, second_pool = select_residual(
                sim_b, candidates, backbone,
                residual_count
            )

            if pool != second_pool:
                raise RuntimeError("Candidate pool mismatch.")

            full_a = backbone | selected_a
            full_b = backbone | selected_b

            if (
                len(full_a) != budget
                or len(full_b) != budget
            ):
                raise RuntimeError("Edge count mismatch.")

            key = (region_id, method, budget)

            if key not in previous_index.index:
                raise RuntimeError(
                    f"Missing previous graph result: {key}"
                )

            old_shared = int(
                previous_index.loc[
                    key, "shared_residual_edges"
                ]
            )

            new_shared = len(
                selected_a & selected_b
            )

            if new_shared != old_shared:
                raise RuntimeError(
                    f"Phase 3B.21 graph mismatch: {key}. "
                    f"Old={old_shared}, New={new_shared}"
                )

            null_result, distance_records = (
                geometry_conditioned_null(
                    pool, selected_a, selected_b, r, c
                )
            )

            detail_records.append({
                "region_id": region_id,
                "graph_type": method,
                "edge_budget": budget,
                "residual_edges": residual_count,
                "residual_pool_size": len(pool),
                "distance_strata": len(distance_records),
                **null_result,
            })

            for record in distance_records:
                stratum_records.append({
                    "region_id": region_id,
                    "graph_type": method,
                    "edge_budget": budget,
                    **record,
                })

    print(
        f"  {region_id}: "
        "connected graphs and null distributions verified",
        flush=True
    )


# ============================================================
# STEP 3: COMPARE METHODS
# ============================================================

print("\n[3] Comparing geometry-conditioned stability...", flush=True)

details = pd.DataFrame(detail_records)
strata_df = pd.DataFrame(stratum_records)

if len(details) != 60:
    raise RuntimeError("Expected 60 graph comparisons.")

summary = (
    details.groupby(
        ["graph_type", "edge_budget"],
        as_index=False
    )
    .agg(
        regions=("region_id", "nunique"),
        mean_observed_shared=("observed_shared", "mean"),
        mean_expected_shared=("expected_shared", "mean"),
        mean_excess_shared=("excess_shared", "mean"),
        mean_conditional_adjusted=(
            "conditional_adjusted", "mean"
        ),
        mean_null_z=("null_z", "mean"),
        positive_excess_regions=(
            "excess_shared",
            lambda x: int((x > 0).sum())
        ),
    )
)

paired_records = []

for (region_id, budget), group in details.groupby(
    ["region_id", "edge_budget"]
):
    g = group.set_index("graph_type")

    global_row = g.loc["global"]
    spatial_row = g.loc["spatial"]

    paired_records.append({
        "region_id": region_id,
        "edge_budget": budget,
        "global_conditional_adjusted":
            global_row.conditional_adjusted,
        "spatial_conditional_adjusted":
            spatial_row.conditional_adjusted,
        "spatial_minus_global_adjusted": (
            spatial_row.conditional_adjusted
            - global_row.conditional_adjusted
        ),
        "global_null_z": global_row.null_z,
        "spatial_null_z": spatial_row.null_z,
        "global_excess_shared":
            global_row.excess_shared,
        "spatial_excess_shared":
            spatial_row.excess_shared,
    })

paired = pd.DataFrame(paired_records)

print("\nGeometry-conditioned summary:")
print(summary.to_string(index=False))

print("\nPaired spatial-minus-global adjusted advantage:")

for budget in BUDGETS:
    subset = paired[
        paired["edge_budget"] == budget
    ]

    differences = subset[
        "spatial_minus_global_adjusted"
    ]

    print(f"\nBudget {budget}")
    print(f"  Mean difference: {differences.mean():.6f}")
    print(f"  Minimum: {differences.min():.6f}")
    print(f"  Maximum: {differences.max():.6f}")
    print(
        f"  Spatial-positive regions: "
        f"{int((differences > 0).sum())}/10"
    )


# ============================================================
# STEP 4: SAVE RESULTS
# ============================================================

print("\n[4] Saving results...", flush=True)

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

details.to_csv(DETAIL, index=False)
strata_df.to_csv(STRATA, index=False)
summary.to_csv(SUMMARY, index=False)
paired.to_csv(PAIRED, index=False)

report = {
    "phase": "3B",
    "step": "3B.25",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": file_hash(MANIFEST),
    "feature_sha256": file_hash(FEATURES),
    "previous_step21_sha256": file_hash(PREVIOUS),
    "regions": len(regions),
    "graph_comparisons": len(details),
    "distance_stratum_records": len(strata_df),
    "graph_reconstruction_verified": True,
    "budgets": list(BUDGETS),
    "methods": list(METHODS),
    "null_model": (
        "Independent uniform residual-edge selection "
        "within exact squared-grid-distance strata, "
        "conditional on scanner-specific stratum counts."
    ),
    "null_distribution": (
        "Exact convolution of hypergeometric overlap "
        "distributions over distance strata."
    ),
    "conditional_adjustment": (
        "(observed_shared - expected_shared) / "
        "(maximum_shared_given_strata - expected_shared)"
    ),
    "exact_tail_probability_interpretation": (
        "Conditional combinatorial null probability, "
        "not population-level biological significance."
    ),
    "frozen_manifest_modified": False,
    "trained_gnn_used": False,
    "biological_independence_verified": False,
    "limitations": [
        "Distance-stratified null does not preserve feature correlations",
        "Distance bins do not fully capture spatial edge topology",
        "Spatial and global graphs have different candidate pools",
        "Some dense configurations may have little null variance",
        "The regions belong to one acquisition pair",
        "No independent patient-level replication",
        "No clinical downstream task has been evaluated",
    ],
}

REPORT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("PHASE 3B.25 - GEOMETRY CONTROL COMPLETED")
print("=" * 70)

print("Regions:", len(regions))
print("Graph comparisons:", len(details))
print("Distance-stratum records:", len(strata_df))
print("Phase 3B.21 graph reproduction: VERIFIED")

print("\nSaved:")
print(DETAIL)
print(STRATA)
print(SUMMARY)
print(PAIRED)
print(REPORT)

print("\nFrozen manifest: UNCHANGED")
print("Original features: UNCHANGED")
print("Trained GNN: NOT USED")
print("Biological independence: UNVERIFIED")
print("\nSTEP 3B.25 COMPLETED")
