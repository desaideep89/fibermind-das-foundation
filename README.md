# FiberMind DAS Foundation

A representation layer for distributed acoustic sensing (DAS) that transfers across interrogators, sites and conditions. Near-term framing: an OEM-agnostic adaptation layer for rail DAS that cuts false alarms and commissioning time. Destination: broken rail detection.

Read `docs/ARCHITECTURE.md` first. Decisions that shaped the code are in `docs/decisions/`.

## Layout

```
fibermind/      the package: physics, data, augment, models, objectives, training, eval
scripts/        one thin entry point per pipeline stage (prepare_data.py, pretrain.py)
configs/        Hydra config
tests/          pytest on synthetic data, CPU, under two minutes
docs/           architecture, decision log, history
```

## Setup

```
pip install -e ".[dev]"
pytest
```

Data, checkpoints and figures live in the artefact store (Cloudflare R2, bucket `fibermind-das`), never in git.

## Status

This is the consolidated layout with the June 2026 code moved in unchanged (forward, loss, gradients and window selection verified identical). The v0.3 components (JEPA mainline, instrument-response augmentation, tube masking, worker-safe loader, probe) follow one pull request at a time; see `docs/ARCHITECTURE.md` section 10.
