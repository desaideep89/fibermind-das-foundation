# 0006: Benchmarks are frozen before the models they evaluate are trained

Status: accepted, 2026-11-03

## Context

The company's evidence is a small number of evaluation figures: a label-efficiency curve for train passage on ADIF HDAS, and the same curve with the probe moved to Silixa iDAS on the same corridor, with and without instrument-response augmentation. Those figures will be shown to investors and used to compare work-trial candidates and the founding engineer's day-90 deliverable. They are worthless if the benchmark drifts between runs or is tuned to a result.

## Decision

A benchmark is a versioned definition in `fibermind/eval/benchmarks/` comprising its data manifest, its labels (including the weak-label threshold and how it was chosen), its splits, its label ladder and its metric. A benchmark is named and frozen before any model is trained against it. Changing any component creates a new benchmark name; results under the old name remain valid and reported as such. The held-out test split is never used for model selection.

## Alternatives considered

- Evaluate ad hoc from notebooks as results are needed. Rejected: this is how unreproducible numbers get into decks.
- Rely on W&B comparison views. Rejected as the record: they compare whatever was logged, which is not the same as holding the benchmark fixed.

## Consequences

Benchmark v1 (train passage, ADIF HDAS, weak labels from energy spread, probe ladder 10 to 1000) is written and reviewed before the first consolidated pretraining run. The report command accepts only evaluation outputs that name a benchmark version. Work-trial candidates receive the benchmark definition as part of the pack and are scored by the same command.
