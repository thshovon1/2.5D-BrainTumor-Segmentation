"""BraTS label handling.

BraTS 2021 annotations use 0 = background, 1 = necrotic/non-enhancing tumour
core (NCR), 2 = peritumoral edema (ED) and 4 = enhancing tumour (ET).
The three evaluation regions are nested:

    WT (whole tumour) = labels {1, 2, 4}
    TC (tumour core)  = labels {1, 4}
    ET (enhancing)    = label  {4}

Label 3 is treated as ET as well, so that data using the later BraTS
convention (ET = 3) is handled identically.
"""

import numpy as np

ET_LABELS = (3, 4)


def regions_from_label(label: np.ndarray) -> np.ndarray:
    """Convert a multi-class BraTS label array (any shape) into a stacked
    binary array of shape (3, *label.shape) ordered as (WT, TC, ET)."""
    label = np.asarray(label)
    et = np.isin(label, ET_LABELS)
    tc = (label == 1) | et
    wt = label > 0
    return np.stack([wt, tc, et]).astype(np.uint8)
