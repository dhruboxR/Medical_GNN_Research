# Phase 2: Research Design and Experimental Protocol

## Research Project

**Cross-Scanner Graph Topology Stability in Computational Histopathology**

This phase develops the experimental framework for investigating how
scanner-related variation affects graph construction, neighborhood
consistency, graph representations, and graph-based learning.

## Research Objectives

1. Quantify graph edge instability across matched scanner acquisitions.
2. Compare feature-based and spatially constrained graph construction.
3. Control graph density, connectivity, and candidate-pool effects.
4. Investigate representation stability and graph coarsening.
5. Evaluate whether stable topology preserves useful tissue information.
6. Assess generalization across appropriately identified independent
   specimens or analysis groups.

## Dataset Strategy

Primary pilot dataset:

- PLISM histopathology images.
- Matched AT2 and GT450 scanner acquisitions.
- GIVH staining condition.
- Registered tile pairs from the Owkin PLISM tile release.

The pilot is intended for feasibility assessment and protocol development.

Larger studies require careful validation of tissue identity, spatial
registration, region overlap, and biological independence.

## Research Design Components

The phase includes:

- Research questions and hypotheses.
- Literature and novelty assessment.
- Dataset selection and sampling criteria.
- Matched-budget graph construction protocols.
- Pretrained feature encoder selection.
- Graph stability metrics and random baselines.
- Specimen-aware statistical analysis plans.
- Graph coarsening and representation evaluation plans.
- Master experimental protocol.

## Planned Graph Comparisons

Candidate baselines include:

- Spatial-only graph.
- Global feature-similarity graph.
- Spatially constrained feature-similarity graph.
- Connectivity-preserving graph variants.

A shared fixed backbone must be evaluated separately from
scanner-dependent feature-selected edges.

## Feature Encoders

The research design considers:

- DINOv2-Small as a general visual feature baseline.
- Pathology-specific pretrained encoders, subject to availability
  and reproducibility checks.

## Statistical Considerations

Graph nodes and overlapping tile regions are not automatically
independent observations.

Inferential comparisons must use defensible biological or
higher-level independent analysis units where available.

## Phase Status

**Status: Research design completed for pilot experiments.**

The design may be refined prospectively before larger studies.
Pilot outcomes should not be used to claim independent validation
or confirmed research novelty.

## Project Folders

- `Docs/`: Research design documentation.
- `literature_review/`: Literature assessment.
- `protocols/`: Experimental procedures and controls.

## Related Phases

- Phase 1: `../dataset_exploration/`
- Phase 3A: `../experimental_research/`
- Phase 3B: `../multi_region_research/`
