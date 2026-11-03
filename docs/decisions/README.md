# Decision log

One file per decision that constrains future work: a format, a convention, a dependency, a protocol. Each entry records the context, the choice, the alternatives considered and the consequences, in a page or less.

Entries are numbered in the order they were made and are never edited after acceptance. To reverse or change a decision, write a new entry that supersedes the old one and link both ways.

Status values: proposed, accepted, superseded.

| Number | Title | Status |
| --- | --- | --- |
| 0001 | One package, one pipeline, stage boundaries as handoff boundaries | accepted |
| 0002 | Manifests in git, not a data versioning tool | accepted |
| 0003 | Memory-mapped window store instead of per-window compressed files | accepted |
| 0004 | Tube masking is the only masking strategy with a cross-channel block | accepted |
| 0005 | Cloudflare R2 is the artefact store; Google Drive is retired from the pipeline | accepted |
| 0006 | Benchmarks are frozen before the models they evaluate are trained | accepted |
| 0007 | Every run is tagged with git commit, config hash and manifest hashes | accepted |
