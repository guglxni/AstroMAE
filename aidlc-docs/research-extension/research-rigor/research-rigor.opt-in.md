# Research-Rigor Extension — Opt-in Prompt

This prompt is presented once, during Requirements Analysis, to decide
whether the Research-Rigor extension rules (RESEARCH-01 … RESEARCH-10)
are loaded for the rest of the workflow.

---

## Prompt

Is this project producing **research-oriented output** where novelty and
scientific defensibility are primary quality goals?

Examples of "yes":
- Building a new ML method with intended publication / preprint.
- Producing a scientific data product (catalogue, foundation-model
  checkpoint) that other researchers will rely on.
- Writing code whose outputs will appear in a paper's results table.

Examples of "no":
- Shipping a product feature where correctness and security matter but
  novelty does not.
- Internal tooling / dashboards.
- Prototyping without any claim beyond "it runs".

### Options

1. **Yes, enforce the Research-Rigor extension.** Literature review is
   required before novelty is claimed; every novel contribution must
   declare a falsifiable prediction; baselines must be matched on
   compute and parameters; ablations, calibration metrics, seed
   stability, data provenance, reproducibility bundles and negative
   results are all blocking requirements.
2. **No, skip this extension.** The workflow proceeds without
   research-rigor gates.

Default: **No** (skip) unless the project intent explicitly states a
research / scientific goal.
