"""Centralized merge map for low-count / composite TUAR labels.
Other modules should import remap_label from here to apply consistent mapping.
"""
from typing import Dict

MERGE_LABEL_MAP: Dict[str, str] = {
    "shiv_elec": "elec",
    "eyem_shiv": "eyem",
    "chew_elec": "elec",
    "elpp": "elec",
    "chew_musc": "musc",
    "eyem_chew": "eyem",
    "eyem_elec": "eyem",
    "musc_elec": "musc",
    "eyem_musc": "musc",
    # add synonyms
    "eog": "eyem",
    "muscle": "musc",
}


def remap_label(label: str) -> str:
    if label is None:
        return label
    key = label.strip().lower()
    return MERGE_LABEL_MAP.get(key, key)
