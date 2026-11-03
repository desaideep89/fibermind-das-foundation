# 0007: Every run is tagged with git commit, config hash and manifest hashes

Status: accepted, 2026-11-03

## Context

The one W&B run checked into the repository (June 2026, validation loss 0.847) cannot be traced to the code that produced it with certainty, because the commit on `main` at the time was a temporal-only model and the cross-channel block was added later with no further logged run. Nor can it be traced to the windows it trained on, because selection was random and unseeded with respect to the manifest.

## Decision

At run start the training entry point records the git commit (and refuses to start on a dirty tree unless an explicit override flag is passed, in which case the diff is saved with the run), the SHA-256 of the fully resolved Hydra config, and the hashes of every manifest the run consumed (windows, split). These are set as W&B tags and written into the run manifest saved next to the checkpoints. A run without them is treated as unreproducible and excluded from any report.

## Alternatives considered

- Rely on W&B's automatic code logging. Rejected as sufficient: it captures the working tree, not the manifest lineage, and it is easy to disable by accident.

## Consequences

Experiment branches are pushed even when they will never be merged, so the commit behind any run is retrievable. The run manifest template in `docs/RUN_MANIFEST_TEMPLATE.json` is replaced by the schema in `fibermind/data/manifests.py`, which covers runs as well as data.
