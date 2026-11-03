# FiberMind DAS Foundation: Architecture

Status: draft for review, November 2026. Owner: Deep Desai. This is the first document a new engineer reads and the reference every other document points back to. Decisions that shaped it are recorded one per file in `docs/decisions/`; change the decision there before changing the text here.

## 1. What this repository is for

FiberMind is building a representation layer for distributed acoustic sensing (DAS) data that transfers across interrogators from different manufacturers, across sites and across conditions. The near-term product framing is an OEM-agnostic adaptation layer for rail DAS that reduces false alarms and commissioning time, running in shadow mode alongside an operator's existing system. The destination is broken rail detection. The near-term deliverable is not the destination; it is the evidence that the representation transfers.

This repository holds everything needed to turn raw DAS recordings into a pretrained encoder and a benchmark result, end to end, from a clean clone. It does not hold data, checkpoints or plots. Those live in the artefact store (section 8) and are referenced from here by manifest.

The claim the company has to be able to demonstrate, and that this pipeline exists to produce, is a pair of numbers: how well an encoder pretrained on one instrument performs on a task when probed on a second instrument recording the same corridor, with and without the instrument-response augmentation. The same-corridor pair we have is the ADIF Aragón test track in February 2023, recorded on an Aragón Photonics HDAS (hour files held by FiberMind through the operator relationship, also available in the PubDAS DAS Month release) and on a Silixa iDAS (PubDAS). Neither dataset is exclusive to us and we do not say otherwise. The edge is the domain knowledge in the preprocessing, the evaluation design and the speed of execution.

## 2. Principles

1. Reproducible from the repository plus the artefact store. Every window array, checkpoint and benchmark figure can be regenerated from a named commit and a named manifest. Nothing the company relies on may depend on a pod that has since been stopped or a command that lives only in someone's terminal history. This has already cost us one full stack of code (June 2026) and is the first thing this architecture designs out.
2. Physics before pixels. A DAS recording is a set of strain-rate time series, one per channel, with time as the primary axis and inter-channel structure (moveout, apparent velocity) as secondary. The preprocessing chain, the patching and the attention layout all follow from that. We do not treat waterfalls as images.
3. Evaluation is designed and frozen before the model that will be evaluated is trained. The benchmark, its labels, its splits and its metric are versioned in this repository and do not move between runs. A run that changes the benchmark is a benchmark change, reviewed as such, not a model improvement.
4. Small vertical slice first. One instrument, one site, one task, one number, running end to end, before any widening. Widening is a config change if the architecture is right.
5. Every stage reads a manifest and writes a manifest. Nothing downstream consumes an artefact that has no manifest.
6. Tests run on a laptop in under two minutes on synthetic data. The GPU is for experiments, not for finding out whether the code runs.

## 3. Repository layout

```
fibermind/              the package; importable as `fibermind`
  data/                 raw readers per instrument, manifest builder, window store, split logic
  physics/              preprocessing chain as pure functions; no I/O, no state
  augment/              instrument-response augmentation (virtual gauge length, resampling,
                        response coloration, gain and fading)
  models/               encoder blocks: temporal patch embed, temporal transformer (Stage A),
                        cross-channel block (Stage B); MAE and JEPA heads
  objectives/           masking strategies and losses, separated from the models that use them
  training/             Lightning module, data module, callbacks (checkpoint sync, run tagging)
  eval/                 linear probe, train-passage benchmark, cross-instrument report
scripts/                one thin entry point per pipeline stage; argument parsing and logging only
configs/                Hydra config groups: data/, model/, objective/, training/, experiment/
tests/                  pytest suite on synthetic data; see section 9
docs/                   this file, DATA.md, RUNBOOK.md, decisions/, history/
```

Rules that keep this honest:

- `fibermind/physics/` and `fibermind/objectives/` import nothing from `fibermind/training/` or `scripts/`. They are the parts most likely to be reused, tested in isolation or handed to an external contributor.
- `scripts/` contain no logic. If a script grows a function, the function moves into the package and gets a test.
- Each top-level package directory has a README stating its interface (inputs, outputs, invariants) in a dozen lines. This is what lets a package be built or reviewed by someone who has not seen the rest, which is how compartmentalised outside work will be commissioned and integrated.
- The January 2026 Colab notebook (`FiberMind_FM_CGT_v1.ipynb`) moves to `docs/history/`. It is a record of how the project started, not a dependency.

The current tree (`src/fibermind/`, `train.py` and `preprocess.py` at the root) is folded into this layout in the consolidation step described in section 10.

## 4. The pipeline

Six stages. Each has one entry point in `scripts/`, one config group in `configs/`, and reads and writes manifests as described. Stage boundaries are also the boundaries at which work can be handed to someone else.

### 4.1 Ingest

Input: raw recordings as delivered by the operator or downloaded from PubDAS, placed unchanged in the artefact store under `raw/<instrument>/<site>/<date>/`.

Output: `manifests/raw/<instrument>_<site>_<date>.json`, one entry per file: path, SHA-256, dataset name inside the file, array shape, sampling rate, channel spacing, gauge length, units of the stored quantity (strain rate, strain, phase), instrument model and firmware where known, source and licence, and any operator notes.

The reader for each instrument lives in `fibermind/data/readers/` and exposes the same interface: open a file, report its shape and metadata, read a `(T, C)` block as float64 in the file's native units. Adding an instrument means adding a reader and a manifest; nothing downstream changes.

### 4.2 Preprocess

Input: a raw manifest and a preprocessing config.

The chain, as validated on the HDAS data and carried unchanged from the June stack:

1. strain rate to strain by cumulative sum divided by the sampling rate;
2. high-pass filter, 8th-order Butterworth at 0.5 Hz, zero-phase (forward and backward), applied along time;
3. the result is dynamic strain;
4. per window, factor into `w_norm` (each channel divided by its own RMS over the window, so unit RMS per channel) and `log_rms` (the natural log of that RMS, one value per channel).

No decibel conversion anywhere in the chain. Amplitude information is carried by `log_rms`, structure by `w_norm`, and the two are fed to the model separately.

Window selection is event-aware: candidate windows are scored by energy spread, the 90th percentile of `log_rms` minus its median, which is high when a band of channels is lit up by a train and flat for ambient noise. A configured fraction of kept windows comes from the top of that ranking and the remainder is random, so the model still sees quiet track. Candidate positions are drawn from a seeded generator and the seed is recorded, so the same selection comes out on every run.

Output: one memory-mapped array per source holding the selected windows, with `w_norm` as `float32` of shape `(n_windows, win_t, win_c)` and `log_rms` of shape `(n_windows, win_c)`, plus a side table of `(t0, c0, spread)` per window, and `manifests/windows/<name>.json` recording the raw manifest hash, every parameter of the chain and the selection, the seed, the output shapes and the SHA of the arrays. One array per source rather than thousands of compressed files: see decision 0003.

### 4.3 Dataset and loader

Input: one or more window manifests and a split config.

The split (train, validation, held-out test) is computed once from the manifest, by window index with a recorded seed, and written to `manifests/splits/<name>.json`. Runs reference the split by name. The held-out set is never used for model selection.

The loader is a memmap window reader. It is worker-safe: each DataLoader worker derives its own random generator from the run seed, the worker id and the item index, so augmentation and any stochastic selection differ across workers. (The August review found all workers forking an identical generator state; the fix is a unit test now, section 9.)

Augmentation is applied here, before normalisation, so that a synthetic gain change is absorbed into `log_rms` in the same way a real change of hardware would be. Augmentation is off by default and switched on by config.

### 4.4 Pretrain

Input: a split, a model config, an objective config, a training config.

Two objectives share one encoder:

- MAE: reconstruct the masked patches of `w_norm`, normalised per patch. This is the ablation floor.
- JEPA: a context encoder sees the visible patches; a small predictor must match the embeddings an exponential-moving-average target encoder produces at the masked positions. No pixel reconstruction, so capacity is not spent memorising fading and speckle. This is the mainline.

Masking is tube masking for any model with a cross-channel block: whole time patches are masked and every channel is kept at the time patches that survive. See decision 0004 and section 6.

Every run is logged to Weights and Biases with the git commit, the resolved config hash and the hashes of the manifests it consumed as tags, so a W&B run can be traced to the exact code, config and data that produced it. Checkpoints are written locally and synced to the artefact store on a schedule during the run, not only at the end.

Output: checkpoints under `checkpoints/<run>/` in the artefact store, with a run manifest alongside.

### 4.5 Evaluate

Input: a checkpoint and a benchmark name.

The first benchmark is train passage detection on the ADIF HDAS recordings. Weak labels come from the energy-spread score with a threshold chosen once and recorded. The protocol: freeze the encoder, fit a linear probe on k labelled windows for k on a fixed ladder (for example 10, 30, 100, 300, 1000), report the metric on the held-out set, and plot it against k. The comparison is against a from-scratch model of the same size trained on the same k labels. The gap at small k is the number that matters.

The second benchmark is the same task and the same probe, but with the encoder pretrained on HDAS windows and the probe fitted and tested on Silixa iDAS windows from the same corridor. Run it for a checkpoint trained with augmentation off and one with augmentation on. The difference between those two curves is the evidence for the instrument-response augmentation, and the headline figure.

Benchmarks are defined in `fibermind/eval/benchmarks/` with their labels, splits, ladder and metric under version control. Changing any of them is a reviewed change with a new benchmark name; old results are never silently invalidated.

### 4.6 Report

Input: one or more evaluation outputs.

One command produces the benchmark figure and a markdown table from the evaluation outputs. Work-trial submissions and internal runs go through the same command, so they are compared on identical ground. Figures go to the artefact store, not to git.

## 5. Data conventions

- Arrays are `(time, channel)` everywhere, `(T, C)`. Batches are `(B, T, C)`; the model adds its own leading singleton channel dimension if a block needs it. Any transposition happens inside the block that needs it and is undone before returning.
- Time is the primary axis. Patches are taken along time, per channel. Channels are never patched.
- Sampling rate is carried in the manifest, not assumed. The current data is 100 Hz. Window length in samples and in seconds are both recorded.
- `w_norm` is unit RMS per channel over the window. `log_rms` is the natural log of the per-channel RMS of dynamic strain in the units the reader reports. Units are recorded in the manifest so that a change of instrument does not silently change the scale of `log_rms`.
- Manifest files are JSON, human-readable, small, and committed to git. They point to artefacts by store path and SHA. The artefacts themselves are never committed (`.gitignore` already enforces this for the relevant extensions).

## 6. Model family

The encoder has two stages and the stages are shared by every objective.

Stage A, temporal: each channel's `w_norm` time series is split into non-overlapping patches of `patch_t` samples and embedded by a 1D convolution; a learned positional encoding over patch index is added; a per-channel conditioning vector derived from `log_rms` is added to every patch of that channel; a transformer over the patch sequence runs independently per channel. Output: `(B, C, N, D)` with `N` the number of time patches.

Stage B, cross-channel: at each time patch, tokens are mixed across channels in two ways. Local attention with a sliding-window mask over the `k` nearest channels captures moveout and local coherence. Global attention over channels pooled down to a fixed count (currently 32) captures wavefield geometry along the whole window of fibre, and is interpolated back to full channel resolution. Stage B assumes a complete channel grid at every time patch it sees. That assumption is why masking must be tube masking: if different channels survive at different time patches, Stage B either skips silently (which it did in v0.2) or mixes tokens that are not simultaneous (which the current `das_mae.py` on `main` does). Both failure modes train without error and produce an encoder whose cross-channel blocks are untrained or scrambled. The unit test in section 9 guards this.

Heads: the MAE decoder (a small transformer over the full patch sequence with mask tokens, predicting the per-patch normalised `w_norm`) and the JEPA predictor (a small transformer predicting target-encoder embeddings at masked positions). An optional amplitude head predicting `log_rms` exists and is off by default.

Where Stage B sits in the stack (after every temporal block, every second block, or once at the end) is a config choice. The June stack applies it once after the temporal encoder; v0.3 interleaved it. Both are expressible from config and the comparison is an experiment, not a code change.

## 7. Objectives and masking

Masking strategies live in `fibermind/objectives/masking.py` and return the same structure regardless of strategy: the indices kept, the binary mask over patches, and the permutation needed to restore order. Tube masking is the only strategy allowed with Stage B present; random patch masking remains available for temporal-only ablations and is refused by config validation if a cross-channel block is configured.

Losses are functions of (prediction, target, mask) and nothing else. The MAE loss normalises each target patch by its own mean and variance with a variance floor. The JEPA loss is a smooth L1 (or MSE, by config) between predicted and target embeddings at masked positions, with the target encoder updated by EMA after each optimiser step.

## 8. Infrastructure

- Code: GitHub, `desaideep89/fibermind-das-foundation`, `main` protected; work on branches, merged by pull request with the test suite green.
- Artefacts: Cloudflare R2, bucket `fibermind-das`, prefixes `raw/`, `windows/`, `checkpoints/`, `figures/`. The bucket is the only durable home for data and checkpoints. Google Drive is for pitch material, not for pipeline artefacts; the old rclone-to-Drive checkpoint callback is retired (decision 0005).
- Compute: RunPod GPU pods, treated as disposable. A pod is bootstrapped by one script that clones the repository at a named commit, installs from the lock file, configures the R2 remote and pulls the manifests it needs. Preprocessed arrays go to `/data/arrays`, checkpoints to `/tmp/checkpoints`, Hydra output to `/tmp/hydra_output`, because the workspace mount has a hidden quota. Nothing is ever done by hand on a pod that is not also captured in `docs/RUNBOOK.md`. The rule "push before stop" is enforced by the stop script refusing to run with uncommitted or unpushed changes.
- Experiment tracking: Weights and Biases, project `fibermind-das`. A run without the git, config and manifest tags is treated as unreproducible and excluded from any report.
- Environment: `pyproject.toml` with a lock file. The same lock installs on a laptop (CPU, for tests) and on a pod (GPU, for experiments).

## 9. Engineering standards

Tests (pytest, CPU, synthetic data, under two minutes):

- Physics: a known synthetic strain-rate signal passes through the chain and produces the expected dynamic strain within tolerance; the chain is deterministic; RMS factoring reconstructs the input exactly from `w_norm` and `log_rms`.
- Selection: the same raw array, config and seed select the same windows twice.
- Loader: two DataLoader workers do not produce identical augmentation sequences; an item is identical when fetched twice with augmentation off.
- Masking: tube masking keeps every channel at every surviving time patch; the kept and mask tensors are consistent; config validation refuses random patch masking with Stage B present.
- Training: after twenty optimiser steps on synthetic windows, every Stage B parameter tensor has changed, for both MAE and JEPA. This is the test that would have caught both historical Stage B bugs.
- Shapes: every config in `configs/model/` builds, runs a forward and backward pass at a reduced size, and produces outputs of the documented shape.
- Scripts: the `prepare_data`, `pretrain` and `probe` entry points run end to end on a tiny synthetic array in a subprocess and exit zero.

Continuous integration runs the suite on every pull request. Formatting and linting (ruff) run as a pre-commit hook and in CI.

Configuration is Hydra with config groups, so an experiment is a named file in `configs/experiment/` that composes data, model, objective and training choices. A run's resolved config is saved alongside its checkpoint and hashed for the W&B tag.

Branching: short-lived branches named by purpose (`feat/`, `fix/`, `exp/`, `docs/`). Experiments that are not going to be merged still get pushed, so the code behind any W&B run is retrievable.

Decisions: anything that constrains future work (a format, a convention, a dependency, a protocol) gets a one-page entry in `docs/decisions/` with the context, the choice, the alternatives considered and the consequences. Entries are numbered and never edited after acceptance; they are superseded by a new entry.

Intellectual property: the company owns the contents of this repository. Contributors, including work-trial candidates and outside contractors, sign a contributor agreement before their first commit. The compartmentalised structure in section 3 lets outside work be scoped to one package with a stated interface, so a contractor needs the interface and a synthetic test fixture, not the rest of the system.

## 10. Lineage and consolidation

Three generations of code exist and this architecture reconciles them.

- January 2026, Colab notebook (`FiberMind_FM_CGT_v1.ipynb`): H5 ingestion, wavelet packet preprocessing, a first MAE. Superseded; kept in `docs/history/`.
- June 2026, `main` of this repository: the physics chain (strain rate, cumsum, HPF, RMS factoring), event-aware selection, the temporal MAE with `log_rms` conditioning, Stage B cross-channel block applied once after the temporal encoder, Lightning and Hydra harness, one logged W&B run (temporal-only model, 41 epochs, validation loss 0.847). Carries the Stage B masking bug described in section 6 and writes thousands of compressed npz files.
- August 2026, `fibermind-fm` v0.2 (handed over) and v0.3 (reviewed and fixed): JEPA mainline with an STFT front end, instrument-response augmentation, synthetic event generator for evaluation, `pretrain.py` and `probe.py` scripts, tube masking, worker-safe RNG, stratified sampling. Verified empirically on CPU and run on an A100 (three pretraining runs: JEPA with augmentation 0.5, JEPA with augmentation 0, MAE baseline). This code is not in this repository and exists only as a handed-over archive; it is the better stack and is to be brought in here.

Consolidation, in order:

1. Create the layout in section 3 and move the June code into it, with no behaviour change, so the first commit on the new structure is a pure refactor and the existing tests (once written) pass against it.
2. Bring in the v0.3 components, each with its unit test from section 9, and re-run the three empirical checks from the August review (Stage B parameters move; worker generators differ; JEPA loss descends on synthetic data) before merging.
3. Replace npz output with the memmap window store and add manifests at every stage boundary.
4. Add the benchmark definitions and the report command, then freeze benchmark v1.
5. Write `docs/RUNBOOK.md` and the pod bootstrap script, and run the whole pipeline once from a fresh pod on the four ADIF HDAS hour files, then register the PubDAS Silixa files for the same corridor.
6. Cut the work-trial pack from the result: clone, three commands, one figure, one written critique.

Step 1 is a prerequisite for everything else and is the only step that touches all files at once; everything after it is a normal pull request.

## 11. Not in scope for now

Fifty-million-parameter pretraining and scaling curves; multi-site and multi-dataset loaders beyond the two ADIF instruments; a platform, user interface or deployment path; integration with any operator's alarm system; a preprint. These are the founding engineer's territory once the first benchmark exists and the money has landed. Nothing in this architecture should make them harder; several things (the reader interface, the manifest chain, the config groups) are there to make them easier.

## 12. Glossary

- DAS: distributed acoustic sensing. An interrogator fires laser pulses down a fibre and turns the Rayleigh backscatter into a strain-rate time series per metre-scale section of fibre, called a channel.
- Interrogator, OEM: the instrument and its manufacturer. Different manufacturers' instruments differ in gauge length, sampling, response and gain, which is the cross-instrument problem.
- Gauge length: the length of fibre over which strain is averaged for one channel. It shapes the instrument response and is one of the quantities the augmentation varies.
- Moveout: the pattern of arrival time versus channel for a propagating disturbance; its slope is the apparent velocity. Captured by Stage B.
- `w_norm`, `log_rms`: the two factors of a preprocessed window; structure and amplitude respectively.
- MAE, JEPA: masked autoencoder (reconstruct masked input) and joint-embedding predictive architecture (predict masked embeddings).
- Tube masking: masking whole time patches across all channels, as opposed to random patches on the time-by-channel grid.
- Linear probe: a linear classifier fitted on frozen encoder features; the standard measure of representation quality.
- Manifest: a small JSON file that names an artefact, its hash and every parameter that produced it.
