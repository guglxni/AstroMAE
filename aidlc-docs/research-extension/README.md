# AI-DLC Research-Rigor Extension

This folder is a self-contained extension to the upstream
[awslabs/aidlc-workflows](https://github.com/awslabs/aidlc-workflows)
rules set. It follows the extension format described in the AI-DLC
README (see `.aidlc/aidlc-rules/aws-aidlc-rule-details/extensions/*`
for the shipped `security/` and `testing/` examples).

## Why it exists

The AWS-provided AI-DLC rules focus on software engineering rigor
(requirements, design, code generation, security, build and test). They
don't cover the particular failure modes of *research-oriented*
software — novelty inflation, non-reproducible results, uncalibrated
probabilistic outputs, cherry-picked baselines, silent negative
results.

This extension adds 10 blocking rules (RESEARCH-01 … RESEARCH-10) that
enforce standard research hygiene inside the AI-DLC workflow.

## Layout

```
research-extension/
└── research-rigor/
    ├── research-rigor.md          # The blocking rules.
    └── research-rigor.opt-in.md   # Structured multiple-choice prompt
                                   # presented during Requirements Analysis.
```

## How to enable it in a downstream project

The canonical location for AI-DLC extensions is
`.aidlc/aidlc-rules/aws-aidlc-rule-details/extensions/`. If you are
using the AI-DLC release download flow (see the repo-root README),
add this extension by either:

1. **Copy-in (local use)**:

   ```bash
   mkdir -p .aidlc/aidlc-rules/aws-aidlc-rule-details/extensions/research
   cp -R aidlc-docs/research-extension/research-rigor \
         .aidlc/aidlc-rules/aws-aidlc-rule-details/extensions/research/
   ```

2. **Upstream submission**: submit the two files to awslabs/aidlc-workflows
   under `aws-aidlc-rule-details/extensions/research/rigor/`.

Once loaded, the core AI-DLC workflow will discover the opt-in prompt
and present it during Requirements Analysis. Answering "Yes" makes the
10 rules blocking for the rest of the workflow.

## Adoption in AstroMAE

AstroMAE opts in to this extension for any research-track work on
branch names prefixed `research-*`. Enforcement evidence for the
current AstroMAE research upgrade lives under
`aidlc-docs/research/`:

- `research-intent.md` (scope + falsifiable thesis)
- `literature-review.md` (RESEARCH-01 evidence)
- `novel-contributions.md` (RESEARCH-02 evidence — per-contribution
  falsifiable predictions)
- `research-plan.md` (RESEARCH-09 evidence — scope-creep brake)
- `evaluation-protocol.md` (RESEARCH-03 / 05 / 06 / 08 evidence)
- `data-provenance.md` (RESEARCH-07 evidence)
