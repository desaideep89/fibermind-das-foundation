# 0002: Manifests in git, not a data versioning tool

Status: accepted, 2026-11-03

## Context

Every artefact the company relies on (window arrays, checkpoints, benchmark results) must be regenerable from a named commit plus the artefact store. Tools such as DVC, LakeFS or W&B Artifacts exist for this. The team is one part-time founder, soon one engineer, with a single bucket and a handful of datasets.

## Decision

Small, human-readable JSON manifests, committed to git, at every stage boundary. A manifest names the artefact by store path and SHA-256, and records every parameter and upstream manifest hash that produced it. Nothing downstream consumes an artefact without a manifest. Artefacts themselves are never committed.

## Alternatives considered

- DVC. Rejected for now: it adds a second tool, a second cache and a learning cost for every contributor and work-trial candidate, and the benefit (content-addressed pointers in git) is what the manifests already provide at our scale.
- W&B Artifacts as the only record. Rejected: it ties reproducibility to a third-party account and to runs that were logged correctly, and it is not readable from a clean clone without credentials.

## Consequences

Manifest schemas are defined once in `fibermind/data/manifests.py` and validated on read. Hashes of large arrays are computed once at write time and stored, not recomputed on every load. If the team or the dataset count grows to the point where manifests are unwieldy, a new entry supersedes this one and DVC is the obvious successor; the manifests would be its inputs.
