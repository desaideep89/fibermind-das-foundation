"""FiberMind DAS foundation: a representation layer for distributed acoustic sensing
that transfers across interrogators, sites and conditions.

Layout and conventions are described in docs/ARCHITECTURE.md. Subpackages:

- physics:    preprocessing chain as pure functions (no I/O)
- data:       readers, window selection and storage, datasets
- augment:    instrument-response augmentation (not yet populated)
- models:     encoder blocks and the MAE model
- objectives: masking strategies and losses
- training:   Lightning module, data module, callbacks
- eval:       probes and benchmarks (not yet populated)
"""

__version__ = "0.4.0.dev0"
