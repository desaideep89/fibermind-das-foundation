# 0001: One package, one pipeline, stage boundaries as handoff boundaries

Status: accepted, 2026-11-03

## Context

Three generations of code exist: a Colab notebook (January 2026), the `src/fibermind` stack on `main` (June 2026) and the `fibermind-fm` v0.3 archive (August 2026). Each was written to be run by one person at a keyboard. None of them can be handed to a new engineer with the instruction "clone and run". Separately, part of the development will be commissioned as compartmentalised packages from outside contractors, which only works if each package has a stated interface and can be built against a synthetic fixture without the rest of the system.

## Decision

A single importable package, `fibermind`, with one subpackage per concern (`data`, `physics`, `augment`, `models`, `objectives`, `training`, `eval`) and a single pipeline of six stages (ingest, preprocess, dataset, pretrain, evaluate, report), one entry script per stage. Subpackage boundaries are the boundaries at which work can be handed to someone else; each has a README stating its interface. `physics` and `objectives` import nothing from `training` or `scripts`.

## Alternatives considered

- Keep the flat `src/` layout and add scripts as needed. Rejected: it has already produced three incompatible generations and no handoff path.
- Separate repositories per concern. Rejected at this size: it adds release and versioning overhead for a one-person team and makes the end-to-end test harder, not easier. Revisit if a package genuinely needs to ship to an outside party.

## Consequences

The first commit on the new structure is a pure refactor of the June code with no behaviour change. The v0.3 components are brought in afterwards, one pull request each. Outside work is scoped to one subpackage plus a synthetic test fixture. The notebook moves to `docs/history/`.
