# Research-Rigor Extension Rules for AI-DLC

## Overview

These rules are MANDATORY cross-cutting constraints that apply when the
AI-DLC workflow is driving **research-oriented software** (ML methods
for science, novel algorithms with intended publication, scientific
data products). They complement the security baseline and the generic
AI-DLC workflow without replacing them.

**Enforcement**: At each applicable stage, the model MUST verify
compliance with these rules before presenting the stage completion
message to the user.

### Blocking Research-Rigor Finding Behaviour

A **blocking research-rigor finding** means:
1. The finding MUST be listed in the stage completion message under a
   "Research Rigor Findings" section with the RESEARCH rule ID and
   description.
2. The stage MUST NOT present the "Continue to Next Stage" option
   until all blocking findings are resolved.
3. The model MUST present only the "Request Changes" option with a
   clear explanation of what needs to change.
4. The finding MUST be logged in `aidlc-docs/audit.md` with the
   RESEARCH rule ID, description and stage context.

If a RESEARCH rule is not applicable to the current project (e.g.
RESEARCH-05 when the project does not compare against prior art), mark
it as **N/A** in the compliance summary — this is not a blocking
finding.

### Default Enforcement

All rules in this document are **blocking** by default.

---

## Rule RESEARCH-01: Literature Review Gate

**Rule**: Before any architectural / algorithmic contribution is
claimed as novel, a literature review artefact MUST exist under
`aidlc-docs/research/literature-review.md` that:
- Enumerates the top-5–10 closest prior-art papers (with arXiv IDs).
- Maps each prior-art paper to the specific piece of the contribution
  it overlaps or precludes.
- Contains an "unclaimed ground" matrix showing which dimensions of
  the contribution are genuinely new.

**Verification**:
- `aidlc-docs/research/literature-review.md` exists and is non-empty.
- Every novelty claim in `novel-contributions.md` has at least one
  cited prior-art row in the literature review.
- The most recent cited paper is ≤ 24 months older than the review
  cut-off, or the review explicitly documents why older prior art is
  still state-of-the-art.

---

## Rule RESEARCH-02: Falsifiable Prediction per Contribution

**Rule**: Every contribution tagged as "novel" MUST declare a
falsifiable prediction — a quantitative, testable statement that, if
not observed empirically, constitutes a negative result for that
contribution. Vague claims ("works better", "is more robust") are not
acceptable.

**Verification**:
- Every `## C<N>.` section in `novel-contributions.md` contains a
  `### Falsifiable prediction` subsection.
- The falsifiable prediction names (a) the metric, (b) the baseline,
  (c) the threshold.
- The threshold is a number, not an adjective.

---

## Rule RESEARCH-03: Matched-Compute Baseline

**Rule**: When a novel method is compared to a baseline, the
comparison MUST be at matched compute (wall-clock or FLOPs within 10%)
AND matched parameter count (within 10%) AND matched data.
Out-of-the-box-better numbers against an underpowered baseline are
not permissible.

**Verification**:
- Every experiment report cites: parameter count, FLOPs or wall-clock,
  dataset size, hardware.
- For every comparison pair, the above fields differ by at most 10%
  between novel and baseline, or the discrepancy is explained and
  justified in the report.

---

## Rule RESEARCH-04: Ablation Matrix Mandatory

**Rule**: A novel method composed of ≥ 2 distinct components
(e.g. "IVAR loss **and** equivariance") MUST ship an ablation matrix
that isolates the marginal contribution of each component.

**Verification**:
- For each multi-component contribution, a `2^k` or Cartesian matrix
  of ablations exists in the reported experiments.
- Each cell of the matrix reports the metric required by the
  falsifiable prediction.

---

## Rule RESEARCH-05: Uncertainty and Calibration

**Rule**: When a contribution produces probabilistic outputs
(photo-z, anomaly scores, classification probabilities), calibration
metrics MUST be reported alongside point estimates.

**Verification**:
- For regression with a density head: CRPS, PIT, 68 / 95% coverage are
  all reported.
- For classification with confidence: ECE (expected calibration error)
  or reliability diagram.
- No paper-grade claim rests on a metric without at least one
  calibration diagnostic.

---

## Rule RESEARCH-06: Seed Stability

**Rule**: Every reported number MUST be averaged over at least 3 seeds
unless the training cost makes this infeasible (in which case this
limitation must be flagged in the report).

**Verification**:
- Results tables report mean ± std (or median + IQR).
- The random seed list is committed in `experiments/seeds.txt` or
  equivalent.

---

## Rule RESEARCH-07: Data Provenance and Licensing

**Rule**: Every dataset used MUST have a provenance entry in
`aidlc-docs/research/data-provenance.md` specifying:
- Source URL / DOI.
- Licence / ToS and whether it permits the intended use.
- Download script + checksum.
- Any derived versions committed under `data/` are noted in
  `.gitignore` and not actually committed unless the licence permits.

**Verification**:
- Every dataset referenced in code or in the config matrix has a
  provenance entry.
- No proprietary or CC-BY-NC data is pushed to a public branch.

---

## Rule RESEARCH-08: Reproducibility Bundle

**Rule**: For every numerical claim in a paper-grade results table the
pipeline MUST be able to produce, from a clean checkout:
- the command line,
- the resolved config (with all overrides expanded),
- the dataset version / hash,
- the git SHA,
- the hardware spec,
- the resulting metrics,
within a single `experiments/<name>/` directory.

**Verification**:
- The experiment-runner CLI writes `config_resolved.yaml`,
  `results.json`, and `env.txt` for every run.
- A CI job can execute the smallest experiment bundle end-to-end and
  produce the expected metrics within tolerance.

---

## Rule RESEARCH-09: Scope-Creep Brake

**Rule**: Research-track projects MUST declare, in
`research-intent.md`, a short "Out of scope" section that is
re-verified at every milestone transition. If a new work item is
proposed that falls in the out-of-scope section, it is a blocking
finding; the finding is resolved either by explicitly moving the item
into scope (with a justification) or by deferring it.

**Verification**:
- `aidlc-docs/research/research-intent.md` has an
  `## Out of scope` section.
- Each completed milestone has either (a) no entries pulled from the
  out-of-scope list, or (b) a justification.

---

## Rule RESEARCH-10: Negative-Result Honesty

**Rule**: Reports MUST present negative results as prominently as
positive ones. If the falsifiable prediction for a contribution is
not met, the report MUST state so plainly in the abstract-equivalent
section. Silently dropping a failing ablation is a blocking finding.

**Verification**:
- Every declared falsifiable prediction has a row in the results
  table, regardless of whether it passes.
- Contributions whose falsifiable prediction fails are explicitly
  flagged as "thesis component not supported by current evidence"
  in the summary.

---

## Compliance Summary Template

At every stage that touches research-track output, the model appends a
summary like:

```
Research Rigor Compliance Summary
- RESEARCH-01 (Literature Review Gate): PASS / FAIL / N/A — <note>
- RESEARCH-02 (Falsifiable Predictions): PASS / FAIL / N/A — <note>
- RESEARCH-03 (Matched-Compute Baseline): PASS / FAIL / N/A — <note>
- RESEARCH-04 (Ablation Matrix): PASS / FAIL / N/A — <note>
- RESEARCH-05 (Uncertainty & Calibration): PASS / FAIL / N/A — <note>
- RESEARCH-06 (Seed Stability): PASS / FAIL / N/A — <note>
- RESEARCH-07 (Data Provenance): PASS / FAIL / N/A — <note>
- RESEARCH-08 (Reproducibility Bundle): PASS / FAIL / N/A — <note>
- RESEARCH-09 (Scope-Creep Brake): PASS / FAIL / N/A — <note>
- RESEARCH-10 (Negative-Result Honesty): PASS / FAIL / N/A — <note>
```
