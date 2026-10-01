"""Decoder-free 2.5D brain tumour segmentation (BraTS 2021).

Reference implementation accompanying the paper
"Decoder-Free 2.5D Brain Tumor Segmentation: Balanced Sampling and a Hybrid
Loss Eliminate Post-processing on BraTS 2021".
"""

__version__ = "2.0.0"

REGIONS = ("WT", "TC", "ET")
MODALITIES = ("t1", "t1ce", "t2", "flair")
