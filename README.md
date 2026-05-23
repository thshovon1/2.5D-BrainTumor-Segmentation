# Revisiting 2.5D Brain Tumor Segmentation

**Revisiting 2.5D Brain Tumor Segmentation: A Hybrid Loss and Balanced Sampling Strategy that Eliminates Post-processing**

> *Paper submitted for review. Code will be fully released upon acceptance.*

---

## Overview

This repository contains the official implementation of our 2.5D multi-scale feed-forward framework for brain tumor segmentation from multi-modal MRI. The key idea: instead of relying on heavy 3D architectures or post-processing tricks, we show that **principled training strategies** — balanced slice sampling + hybrid loss + decoder-free deep supervision — are enough to achieve competitive performance.

**Achieved DSC of 0.9088 on BraTS 2021 (Whole Tumor), without any post-processing.**

---

## Key Contributions

- **Decoder-free multi-scale architecture** — segmentation outputs generated directly from hierarchical feature maps via deep supervision; no transposed convolutions, no reconstruction pathway.
- **Hybrid loss function** — Dice + BCE + Focal, combining overlap accuracy, pixel-wise supervision, and hard-sample emphasis.
- **Balanced slice sampling** — tumor-present and tumor-absent slices mixed at a controlled ratio (ρ = 0.25), directly addressing class imbalance without architectural changes.
- **No post-processing** — the model produces structurally coherent predictions out-of-the-box; we additionally show that LCC filtering is *harmful* for multi-focal tumor cases.

---

## Results

| Method | Architecture Type | Post-processing | DSC (WT) |
|---|---|---|---|
| 3D U-Net | 3D CNN | Required | ~0.8950 |
| TransUNet | 2D CNN + Transformer | Required | ~0.9010 |
| Attention U-Net | 2D CNN + Attention | Required | ~0.8870 |
| UNETR | 3D ViT Transformer | Required | ~0.9120 |
| SwinUNETR | 3D Swin Transformer | Required | ~0.9150 |
| nnU-Net | 3D CNN Ensemble | Required (LCC) | ~0.9250 |
| **Proposed (Ours)** | **2.5D CNN (Multi-scale)** | **None** | **0.9088** |

Inference speed: **~268 slices/second** — full BraTS volume segmented in **under 1 second**.

---

## Repository Structure

```
├── data/
│   └── preprocess.py          # BraTS .tar → 2.5D NumPy chunks
├── models/
│   └── ascension.py           # Model architecture (ASCENSION)
├── utils/
│   └── losses.py              # Hybrid loss (Dice + BCE + Focal)
├── train.py                   # Training script
├── inference.py               # Inference script
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **BraTS 2021** dataset. You must request access separately:

- Register and download at: [Synapse BraTS 2021](https://www.synapse.org/#!Synapse:syn25829067/wiki/610863)
- After download, place the `.tar` file and update the path in `data/preprocess.py`

The dataset is **not** included in this repository.

---

## Preprocessing

The preprocessing pipeline converts raw BraTS `.nii.gz` volumes into 2.5D NumPy chunks ready for training.

**What it does:**
1. Loads all 4 MRI modalities (T1, T1ce, T2, FLAIR)
2. Applies per-subject Z-score normalization on the brain mask
3. Stacks adjacent axial slices {i-1, i, i+1} across all 4 modalities → 12-channel input tensor
4. Center-crops/pads each slice to 160×160
5. Saves inputs as `slice_XXX_x.npy` and ground-truth masks as `slice_XXX_y.npy`

```bash
# Update TAR_FILE_PATH in data/preprocess.py, then run:
python data/preprocess.py
```

---

## Training

```bash
python train.py
```

Key hyperparameters (edit in `train.py`):

| Parameter | Value |
|---|---|
| Input size | 160 × 160 |
| Channels | 12 (4 modalities × 3 slices) |
| Batch size | 32 |
| Epochs | 10 |
| Learning rate | 3×10⁻⁴ (cosine annealing) |
| Empty slice ratio (ρ) | 0.25 |
| Loss weights | Dice=0.5, BCE=0.3, Focal=0.2 |

**Hardware used:** NVIDIA RTX 4060 Ti (16GB VRAM), 32GB RAM. Training time: ~90 minutes.

---

## Inference

```bash
python inference.py --input_dir /path/to/npy_slices --threshold 0.6
```

The optimal global threshold (t = 0.6) was selected via a sweep on the validation set.

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`, `nibabel`, `tqdm`, `matplotlib`, `opencv-python`

---

## Hardware

Trained and tested on:
- GPU: NVIDIA RTX 4060 Ti (16 GB VRAM)
- CPU: AMD Ryzen 5600
- RAM: 32 GB

---

## Citation

> *Citation will be added upon paper acceptance.*

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The BraTS 2021 dataset has its own terms of use. Please comply with the dataset license when using this code.

---

## Acknowledgements

We thank the organizers of the BraTS 2021 challenge for providing the benchmark dataset. This work was conducted at **American International University – Bangladesh (AIUB)**.
