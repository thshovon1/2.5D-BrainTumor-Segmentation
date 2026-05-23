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

Inference speed: **~268 slices/second**

---

## Repository Structure

```
├── data/
│   └── preprocess.py          # BraTS .tar → 2.5D NumPy chunks
├── 2.5D Brain Tumor.ipynb     # Full training & inference notebook
├── best.pth                   # Pretrained model weights (2.62 MB)
├── requirements.txt
└── README.md
```

---

## Pretrained Weights

The best model checkpoint is included in this repository as `best.pth` (2.62 MB).

To load and run inference:

```python
import torch

# Define your model first (see notebook for full architecture)
model = ASCENSION()
model.load_state_dict(torch.load('best.pth', map_location='cpu'))
model.eval()

# Run inference on a single 2.5D slice
with torch.no_grad():
    input_tensor = torch.randn(1, 12, 160, 160)  # replace with real data
    prediction = model(input_tensor)
```

The optimal binarization threshold is **t = 0.6**, selected via sweep on the validation set.

---

## Dataset

This project uses the **BraTS 2021** dataset. You must request access separately:

- Register and download at: [Synapse BraTS 2021](https://www.synapse.org/#!Synapse:syn25829067/wiki/610863)
- After download, place the `.tar` file and update `TAR_FILE_PATH` in `data/preprocess.py`

The dataset (~240 GB) is **not** included in this repository.

---

## Preprocessing

The preprocessing pipeline converts raw BraTS `.nii.gz` volumes into 2.5D NumPy chunks ready for training.

**What it does:**
1. Loads all 4 MRI modalities (T1, T1ce, T2, FLAIR)
2. Applies per-subject Z-score normalization on the brain mask
3. Stacks adjacent axial slices {i−1, i, i+1} across all 4 modalities → 12-channel input tensor
4. Center-crops/pads each slice to 160×160
5. Saves inputs as `slice_XXX_x.npy` and ground-truth masks as `slice_XXX_y.npy`

```bash
# Update TAR_FILE_PATH in data/preprocess.py, then run:
python data/preprocess.py
```

---

## Training

The full training pipeline is available in `2.5D Brain Tumor.ipynb`. Open it in Google Colab or Jupyter.

Key hyperparameters:

| Parameter | Value |
|---|---|
| Input size | 160 × 160 |
| Channels | 12 (4 modalities × 3 slices) |
| Batch size | 32 |
| Epochs | 10 |
| Learning rate | 3×10⁻⁴ (cosine annealing) |
| Empty slice ratio (ρ) | 0.25 |
| Loss weights | Dice=0.5, BCE=0.3, Focal=0.2 |

**Hardware used:** NVIDIA RTX 4060 Ti (16 GB VRAM), 32 GB RAM. Training time: ~90 minutes.

---

## Inference

Inference is included in the notebook. The model processes full BraTS volumes slice-by-slice:

- **Speed:** ~268 slices/second
- **Full volume (155 slices):** under 1 second
- **Threshold:** t = 0.6

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
