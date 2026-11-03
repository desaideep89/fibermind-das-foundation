"""Preprocess raw HDAS hour files into event-aware windows.

Thin wrapper over fibermind.data.windows; no logic lives here. Sources and
parameters are the June 2026 values and will move to a manifest plus Hydra
config with the artefact-store work (decisions 0002 and 0003).

Usage (from the repository root):
    python scripts/prepare_data.py
"""

from fibermind.data import SelectionConfig, process_h5_sources

SOURCES = [
    {
        "path": "/workspace/data/raw/2023_02_14_00h00m28s_HDAS_1H_Strain.h5",
        "dset": "HDAS_DATA",
        "name": "adif_00h",
    },
    {
        "path": "/workspace/data/raw/2023_02_14_06h00m28s_HDAS_1H_Strain.h5",
        "dset": "HDAS_DATA",
        "name": "adif_06h",
    },
    {
        "path": "/workspace/data/raw/2023_02_14_12h00m28s_HDAS_1H_Strain.h5",
        "dset": "HDAS_DATA",
        "name": "adif_12h",
    },
    {
        "path": "/workspace/data/raw/2023_02_14_18h00m28s_HDAS_1H_Strain.h5",
        "dset": "HDAS_DATA",
        "name": "adif_18h",
    },
]

CONFIG = SelectionConfig(win_t=512, win_c=256, n_keep=2000, candidate_mult=4, event_frac=0.7, fs=100.0)
OUT_DIR = "/data/arrays"
SEED = 42


def main():
    process_h5_sources(SOURCES, CONFIG, OUT_DIR, SEED)


if __name__ == "__main__":
    main()
