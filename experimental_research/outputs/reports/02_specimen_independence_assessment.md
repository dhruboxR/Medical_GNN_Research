# PLISM Biological Specimen Independence Assessment

## Research Context

- Phase: 3A
- Step: 3.2
- Dataset: PLISM
- Scanner pair: AT2 and GT450
- Staining condition: GIVH

## Local Metadata Results

| Metric | Value |
|---|---:|
| Matched tile locations | 90 |
| Complete scanner pairs | 90 |
| Valid cross-scanner pairs | 90 |
| Observed scanner groups | 2 |
| Distinct observed slide IDs | 2 |

## Biological Independence

**Status: NOT VERIFIED**

The available metadata does not establish independently
identified biological specimens.

Slide IDs represent acquisition-related identifiers and
must not be counted as separate independent specimens
without verified provenance.

The dataset documentation describes repeated acquisitions
of shared tissue material across scanner and staining
conditions.

## Specimen Identifier Audit

Candidate explicit specimen fields observed:
None

Any discovered identifier would require verification
before it could be used for statistical grouping.

## Matched Tissue Locations

The pilot includes 90 valid
cross-scanner tile pairs.

This supports paired scanner-comparison experiments,
but not independent specimen-level statistical claims.

## Sampling Implications

1. Keep the Phase 1 pilot as development-only.
2. Do not use tile-level random train/test splitting.
3. Do not interpret scanner-specific slides as
   independent biological specimens.
4. Verify tissue-core mapping before assigning
   independent analysis groups.
5. Consider complementary datasets if independent
   specimen-level validation is not feasible.

## Research Readiness

| Research activity | Decision |
|---|---|
| Paired scanner feature comparison | Suitable |
| Paired graph topology comparison | Suitable |
| Specimen-level significance testing | Not ready |
| Independent held-out specimen testing | Not ready |
| Clinical classification claims | Not supported |

## Next Step

Step 3.3: Tissue-Core Mapping and Sampling Feasibility.

Investigate whether PLISM provides tissue-core annotations
or external metadata that can identify biologically
distinct sampling units.

No additional image downloading is required at this stage.

## References

- PLISM dataset:
  https://huggingface.co/datasets/owkin/plism-dataset-tiles

- Original PLISM publication:
  https://doi.org/10.1038/s41597-024-03122-5
