
"""
PHASE 4.2.29 - ROBUSTNESS AND XAI SYNTHESIS

Read-only consolidation of completed experiments:
  33 representation preservation
  41 corrected IG reliability
  43 learned edge-mask reliability

Produces:
  44_research_synthesis.csv
  44_research_synthesis_report.json

No model inference or retraining.
No checkpoint selection.
No test evaluation.
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math


ROOT = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

REPRESENTATION = (
    ROOT / "outputs/reports/33_representation_preservation_report.json"
)
REPRESENTATION_TABLE = (
    ROOT / "outputs/tables/33_representation_preservation_summary.csv"
)

IG = (
    ROOT / "outputs/reports/41_corrected_ig_audit_report.json"
)
IG_DETAIL = (
    ROOT / "outputs/xai/41_corrected_ig_degree_audit.csv"
)
IG_REGIONS = (
    ROOT / "outputs/xai/41_corrected_ig_paired_regions.csv"
)

EDGE = (
    ROOT / "outputs/reports/43_edge_mask_reliability_report.json"
)
EDGE_MASKS = (
    ROOT / "outputs/xai/43_edge_mask_diagnostics.csv"
)
EDGE_PAIRS = (
    ROOT / "outputs/xai/43_edge_mask_paired_audit.csv"
)
EDGE_REGIONS = (
    ROOT / "outputs/xai/43_edge_mask_region_comparison.csv"
)

TABLE_OUT = (
    ROOT / "outputs/tables/44_research_synthesis.csv"
)
REPORT_OUT = (
    ROOT / "outputs/reports/44_research_synthesis_report.json"
)

MODELS = (
    "ResidualGate05",
    "TopologyConsistentResidual",
)

SOURCE_FILES = {
    "representation_report": REPRESENTATION,
    "representation_table": REPRESENTATION_TABLE,
    "ig_report": IG,
    "ig_detail": IG_DETAIL,
    "ig_regions": IG_REGIONS,
    "edge_report": EDGE,
    "edge_masks": EDGE_MASKS,
    "edge_pairs": EDGE_PAIRS,
    "edge_regions": EDGE_REGIONS,
}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(
        newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))


def write_csv_new(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def check_hash(report, key, file_path):
    expected = report[key]
    observed = sha256(file_path)

    if expected != observed:
        raise RuntimeError(
            f"Provenance mismatch: {file_path}"
        )


def numeric(value, label):
    number = float(value)

    if not math.isfinite(number):
        raise RuntimeError(
            f"Nonfinite metric: {label}"
        )

    return number


def validate_region_table(path, expected_rows):
    records = read_csv(path)

    if len(records) != expected_rows:
        raise RuntimeError(
            f"Unexpected row count: {path}"
        )

    return records


def get_interval(report, metric):
    return report["paired_region_bootstrap"][
        "results"
    ][metric]


print("=" * 70)
print("PHASE 4.2.29 - ROBUSTNESS AND XAI SYNTHESIS")
print("=" * 70)


# ---------------------------------------------------
# 1. Verify source availability
# ---------------------------------------------------

print("\n[1] Checking historical research artifacts...")

for path in SOURCE_FILES.values():
    if not path.is_file():
        raise RuntimeError(
            f"Missing research artifact: {path}"
        )

for path in (TABLE_OUT, REPORT_OUT):
    if path.exists():
        raise RuntimeError(
            f"Refusing overwrite: {path}"
        )

representation = read_json(REPRESENTATION)
ig = read_json(IG)
edge = read_json(EDGE)

expected_steps = (
    (representation, "4.2.18"),
    (ig, "4.2.26"),
    (edge, "4.2.28"),
)

for report, step in expected_steps:
    if report["step"] != step:
        raise RuntimeError(
            f"Unexpected source experiment: {step}"
        )

    if report["test_evaluated"] is not False:
        raise RuntimeError(
            "Test evaluation recorded in source."
        )

    if report["training_performed"] is not False:
        raise RuntimeError(
            "Unexpected model training."
        )

graph_hash = representation["graph_archive_sha256"]
protocol_hash = representation["protocol_sha256"]

for report in (ig, edge):
    if (
        report["graph_archive_sha256"] != graph_hash
        or report["protocol_sha256"] != protocol_hash
    ):
        raise RuntimeError(
            "Source graph/protocol mismatch."
        )

check_hash(
    representation,
    "summary_table_sha256",
    REPRESENTATION_TABLE,
)

check_hash(ig, "detail_table_sha256", IG_DETAIL)
check_hash(ig, "region_table_sha256", IG_REGIONS)

check_hash(edge, "mask_table_sha256", EDGE_MASKS)
check_hash(edge, "pair_table_sha256", EDGE_PAIRS)
check_hash(edge, "region_table_sha256", EDGE_REGIONS)


# ---------------------------------------------------
# 2. Verify expected experiment sizes
# ---------------------------------------------------

print("\n[2] Verifying experimental coverage...")

if representation["validation_regions"] != 10:
    raise RuntimeError("Incorrect representation regions.")

if (
    ig["regions"] != 10
    or edge["validation_regions"] != 10
):
    raise RuntimeError("Incorrect XAI regions.")

if (
    tuple(ig["seeds"]) != (42, 123, 2026)
    or tuple(edge["seeds"]) != (42, 123, 2026)
):
    raise RuntimeError("Training seed mismatch.")

representation_rows = validate_region_table(
    REPRESENTATION_TABLE, 2
)
validate_region_table(IG_DETAIL, 60)
validate_region_table(IG_REGIONS, 10)
validate_region_table(EDGE_MASKS, 120)
validate_region_table(EDGE_PAIRS, 60)
validate_region_table(EDGE_REGIONS, 10)

representation_index = {
    row["model"]: row
    for row in representation_rows
}

if (
    set(representation_index) != set(MODELS)
    or set(ig["model_summary"]) != set(MODELS)
    or set(edge["model_results"]) != set(MODELS)
):
    raise RuntimeError(
        "Model coverage mismatch."
    )


# ---------------------------------------------------
# 3. Consolidate model-level measurements
# ---------------------------------------------------

print("\n[3] Consolidating retrieval and XAI findings...")

fields = [
    "model",
    "normal_recall_at_1",
    "discordant_recall_at_1",
    "robustness_drop_pp",
    "absolute_ig_excess_rho",
    "signed_ig_excess_rho",
    "at2_signed_ig_fidelity_excess",
    "gt450_signed_ig_fidelity_excess",
    "edge_mask_mean",
    "edge_mask_std",
    "edge_mask_excess_rho",
    "edge_mask_excess_top12_overlap",
    "at2_edge_fidelity_excess",
    "gt450_edge_fidelity_excess",
]

rows = []

for model in MODELS:
    r = representation_index[model]
    i = ig["model_summary"][model]
    e = edge["model_results"][model]

    record = {
        "model": model,
        "normal_recall_at_1": numeric(
            r["mean_normal_recall_at_1"], "normal R@1"
        ),
        "discordant_recall_at_1": numeric(
            r["mean_discordant_recall_at_1"],
            "discordant R@1",
        ),
        "robustness_drop_pp": numeric(
            r["robustness_drop_pp"],
            "robustness drop",
        ),
        "absolute_ig_excess_rho": numeric(
            i["absolute_excess_rho"], "IG rho"
        ),
        "signed_ig_excess_rho": numeric(
            i["signed_excess_rho"], "signed IG rho"
        ),
        "at2_signed_ig_fidelity_excess": numeric(
            i["at2_signed_positive_excess"],
            "AT2 IG fidelity",
        ),
        "gt450_signed_ig_fidelity_excess": numeric(
            i["gt450_signed_positive_excess"],
            "GT450 IG fidelity",
        ),
        "edge_mask_mean": numeric(
            e["mean_mask_weight"], "mask weight"
        ),
        "edge_mask_std": numeric(
            e["mean_mask_std"], "mask dispersion"
        ),
        "edge_mask_excess_rho": numeric(
            e["mean_excess_rho"], "edge rho"
        ),
        "edge_mask_excess_top12_overlap": numeric(
            e["excess_top12_overlap"],
            "edge overlap",
        ),
        "at2_edge_fidelity_excess": numeric(
            e["at2_fidelity_excess"],
            "AT2 edge fidelity",
        ),
        "gt450_edge_fidelity_excess": numeric(
            e["gt450_fidelity_excess"],
            "GT450 edge fidelity",
        ),
    }

    rows.append(record)

    print(
        f"  {model} | "
        f"Normal R@1 {record['normal_recall_at_1']:.4f} | "
        f"Discordant R@1 "
        f"{record['discordant_recall_at_1']:.4f} | "
        f"IG AT2 fidelity "
        f"{record['at2_signed_ig_fidelity_excess']:+.4f} | "
        f"Edge AT2 fidelity "
        f"{record['at2_edge_fidelity_excess']:+.4f}"
    )


# ---------------------------------------------------
# 4. Preserve uncertainty and interpretation limits
# ---------------------------------------------------

print("\n[4] Recording paired uncertainty...")

selected_intervals = {
    "ig_absolute_excess_rho": get_interval(
        ig, "absolute_excess_rho"
    ),
    "ig_signed_excess_rho": get_interval(
        ig, "signed_excess_rho"
    ),
    "ig_at2_signed_fidelity": get_interval(
        ig, "at2_signed_positive_excess"
    ),
    "ig_gt450_signed_fidelity": get_interval(
        ig, "gt450_signed_positive_excess"
    ),
    "edge_excess_rho": get_interval(
        edge, "excess_rho"
    ),
    "edge_excess_top12_overlap": get_interval(
        edge, "excess_top12_overlap"
    ),
    "edge_at2_fidelity": get_interval(
        edge, "at2_fidelity_excess"
    ),
    "edge_gt450_fidelity": get_interval(
        edge, "gt450_fidelity_excess"
    ),
}

robustness_difference = (
    rows[1]["discordant_recall_at_1"]
    - rows[0]["discordant_recall_at_1"]
)

normal_difference = (
    rows[1]["normal_recall_at_1"]
    - rows[0]["normal_recall_at_1"]
)

print(
    f"  Discordant retrieval difference: "
    f"{robustness_difference * 100:+.2f} pp"
)
print(
    f"  Normal retrieval difference: "
    f"{normal_difference * 100:+.2f} pp"
)


# ---------------------------------------------------
# 5. Save synthesis
# ---------------------------------------------------

print("\n[5] Saving final exploratory synthesis...")

write_csv_new(TABLE_OUT, rows, fields)

report = {
    "phase": "4.2",
    "step": "4.2.29",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "type": "exploratory_research_synthesis",
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "source_sha256": {
        name: sha256(path)
        for name, path in SOURCE_FILES.items()
    },
    "models": rows,
    "paired_regularized_minus_baseline": {
        "discordant_recall_difference_pp":
            robustness_difference * 100,
        "normal_recall_difference_pp":
            normal_difference * 100,
        "xai_descriptive_intervals":
            selected_intervals,
    },
    "conclusions": [
        "Consistency improves synthetic scanner-discordant "
        "retrieval under the evaluated perturbations.",
        "Normal-topology retrieval decreases slightly.",
        "Corrected IG feature attributions show positive "
        "deletion-fidelity excess in both models.",
        "The unregularized residual model has stronger "
        "signed IG deletion fidelity.",
        "Learned edge masks show weak cross-scanner "
        "structural attribution agreement.",
        "The unregularized residual model has stronger "
        "historical edge-deletion fidelity.",
        "No independent biological or clinical "
        "generalization is established.",
    ],
    "limitations": [
        "Ten validation regions from shared biological material",
        "Repeated exploratory use of the validation partition",
        "Synthetic graph perturbations are not real scanner shifts",
        "IG explains 384-dimensional features, not image pixels",
        "Edge masks optimized on validation instances",
        "Edge deletion fidelity lacks edge-type-matched controls",
        "Descriptive region bootstrap is not patient-level inference",
        "No pathologist-validated explanation ground truth",
        "This synthesis does not choose a winning model",
    ],
    "synthesis_table_sha256":
        sha256(TABLE_OUT),
    "model_training_performed": False,
    "model_inference_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "recommended_next_stage":
        "Phase 4.3 protocol freeze",
}

write_json_new(REPORT_OUT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.29 - SYNTHESIS COMPLETED")
print("=" * 70)

print("Saved:")
print(TABLE_OUT)
print(REPORT_OUT)

print("\nTraining: NO")
print("Model inference: NO")
print("Checkpoints modified: NO")
print("Test evaluated: NO")
print("Next: PHASE 4.3 - PROTOCOL FREEZE")
