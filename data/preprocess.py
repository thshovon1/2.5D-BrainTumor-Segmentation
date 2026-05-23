import os
import tarfile
import tempfile
import numpy as np
import nibabel as nib
from pathlib import Path

# --- Configuration based on Draft 5 ---
MODALITIES = ['t1', 't1ce', 't2', 'flair']
TARGET_SHAPE = (160, 160) # As specified in the manuscript
OUTPUT_DIR = "./brats_25d_numpy"

def normalize_intensity(volume):
    """
    Applies Z-score normalization only to the non-zero brain regions.
    """
    brain_mask = volume > 0
    if brain_mask.sum() == 0:
        return volume
    
    mean = volume[brain_mask].mean()
    std = volume[brain_mask].std()
    
    normalized_volume = np.zeros_like(volume, dtype=np.float32)
    normalized_volume[brain_mask] = (volume[brain_mask] - mean) / (std + 1e-8)
    return normalized_volume

def center_crop_pad(volume, target_h, target_w):
    """
    Crops or pads the slice to the required 160x160 resolution.
    """
    h, w = volume.shape
    
    # Calculate padding or cropping bounds
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    crop_h = max(h - target_h, 0)
    crop_w = max(w - target_w, 0)
    
    # Apply padding if smaller than 160x160
    if pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, 
                        ((pad_h//2, pad_h - pad_h//2), 
                         (pad_w//2, pad_w - pad_w//2)), 
                        mode='constant', constant_values=0)
    
    # Apply cropping if larger than 160x160
    if crop_h > 0 or crop_w > 0:
        start_h = crop_h // 2
        start_w = crop_w // 2
        volume = volume[start_h:start_h+target_h, start_w:start_w+target_w]
        
    return volume

def extract_25d_chunks_from_patient(patient_dir, output_dir, patient_id):
    """
    Reads 4 modalities, normalizes, stacks into 2.5D (12 channels), and saves as .npy
    """
    volumes = []
    
    # 1. Load and Normalize all 4 modalities
    for mod in MODALITIES:
        file_path = os.path.join(patient_dir, f"{patient_id}_{mod}.nii.gz")
        if not os.path.exists(file_path):
            print(f"Warning: Missing modality {mod} for {patient_id}")
            return
            
        img = nib.load(file_path)
        vol = img.get_fdata()
        vol = normalize_intensity(vol)
        volumes.append(vol)
        
    # Stack into shape: (4_modalities, H, W, D)
    volume_stack = np.stack(volumes, axis=0)
    
    # Load Segmentation Mask (Ground Truth)
    seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
    seg_vol = nib.load(seg_path).get_fdata() if os.path.exists(seg_path) else None

    # Get depth (number of slices)
    _, h, w, depth = volume_stack.shape
    
    # Create patient output directory
    patient_out_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_out_dir, exist_ok=True)

    # 2. Extract 2.5D Chunks (Slice i-1, i, i+1)
    for i in range(depth):
        # Handle zero-padding for boundary slices
        idx_prev = max(0, i - 1)
        idx_next = min(depth - 1, i + 1)
        
        # Extract slices across all 4 modalities: Shape (4, H, W)
        slice_prev = volume_stack[:, :, :, idx_prev]
        slice_curr = volume_stack[:, :, :, i]
        slice_next = volume_stack[:, :, :, idx_next]
        
        # If at exact boundary, zero out the out-of-bounds slice
        if i == 0:
            slice_prev = np.zeros_like(slice_curr)
        if i == depth - 1:
            slice_next = np.zeros_like(slice_curr)
            
        # Concatenate to create 12-channel tensor (4 mods * 3 slices)
        # Shape becomes (12, H, W)
        tensor_25d = np.concatenate([slice_prev, slice_curr, slice_next], axis=0)
        
        # Crop/Pad to 160x160 for each channel
        tensor_160 = np.zeros((12, TARGET_SHAPE[0], TARGET_SHAPE[1]), dtype=np.float32)
        for c in range(12):
            tensor_160[c] = center_crop_pad(tensor_25d[c], TARGET_SHAPE[0], TARGET_SHAPE[1])
            
        # Save X (Input)
        chunk_filename = os.path.join(patient_out_dir, f"slice_{i:03d}_x.npy")
        np.save(chunk_filename, tensor_160)
        
        # Save Y (Ground Truth) if available
        if seg_vol is not None:
            # Binarize to Whole Tumor (WT) as per manuscript (M > 0)
            seg_slice = (seg_vol[:, :, i] > 0).astype(np.float32) 
            seg_160 = center_crop_pad(seg_slice, TARGET_SHAPE[0], TARGET_SHAPE[1])
            seg_filename = os.path.join(patient_out_dir, f"slice_{i:03d}_y.npy")
            np.save(seg_filename, seg_160)

def process_tar_archive(tar_path, output_dir):
    """
    Extracts the BraTS .tar file to a temporary directory and processes each patient.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting {tar_path}...")
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=temp_dir)
            
        print("Extraction complete. Beginning 2.5D chunk generation...")
        
        # Iterate through patient folders in the extracted directory
        # BraTS folders are usually named like 'BraTS2021_00001'
        for root, dirs, files in os.walk(temp_dir):
            for d in dirs:
                if "BraTS" in d:
                    patient_dir = os.path.join(root, d)
                    print(f"Processing patient: {d}...")
                    extract_25d_chunks_from_patient(patient_dir, output_dir, d)
                    
    print(f"✅ All NumPy chunks saved to {output_dir}")

# --- Execution ---
if __name__ == "__main__":
    TAR_FILE_PATH = "path/to/your/BraTS2021.tar" # Update this path
    
    process_tar_archive(TAR_FILE_PATH, OUTPUT_DIR)