# 0005: Cloudflare R2 is the artefact store; Google Drive is retired from the pipeline

Status: accepted, 2026-11-03

## Context

Raw data and checkpoints have been kept in a mixture of Google Drive (the June `RcloneCheckpointBackup` callback), the Cloudflare R2 bucket `fibermind-das` (used from August), and the local disk of RunPod pods that no longer exist. One full stack of code was lost in June because it lived only on a pod. Google Drive is where pitch material lives and is not a programmatic store.

## Decision

R2 bucket `fibermind-das` is the only durable home for pipeline artefacts, with prefixes `raw/`, `windows/`, `checkpoints/` and `figures/`. Checkpoints sync to R2 on a schedule during training, not only at the end. The Google Drive callback is removed. Pods are disposable: a bootstrap script clones the repository at a named commit, installs from the lock file, configures the R2 remote and pulls the manifests it needs. A stop script refuses to run with uncommitted or unpushed changes.

## Alternatives considered

- Keep Drive for checkpoints. Rejected: no S3 interface, quota and sharing behaviour unsuited to multi-gigabyte artefacts, and it mixes company data with pitch decks.
- RunPod network volumes as the store. Rejected: tied to one provider and one region, and not readable from a laptop.

## Consequences

rclone remote `r2` (S3 provider, Cloudflare endpoint, `region=auto`; the "provider not known" notice is harmless) is part of the bootstrap. Egress from R2 is free, which matters for pulling arrays to a new pod. Access keys live in the pod's environment and are never committed; the runbook records how they are provisioned.
