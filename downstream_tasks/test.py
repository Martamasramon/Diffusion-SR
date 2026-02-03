from __future__ import annotations

from pathlib import Path
import torch
import nibabel as nib

from image_to_nifti import PatientToNifti, save_mask_as_nifti        
from segmentor import Segmentor          

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PatientToNifti(
        diffusion   = None,
        device      = device,
        use_T2W     = True,             
    )

    seg = Segmentor(device=device)

    n = min(10, len(ds))
    print(f"Running segmentation on first {n} patients...")

    for i in range(n):
        item = ds[i]  # triggers NIfTI caching if needed
        pid  = item["patient_id"]

    ### ERROR - IT'S USING ADC! NOT T2W!
        # Prefer T2W_HR if available, else ADC_HR
        if "T2W_HR" in item and item["T2W_HR"] is not None:
            print('t2w')
            vol_path = item["T2W_HR"]
            mod = "T2W_HR"
        else:
            print('adc')
            vol_path = item["ADC_HR"]
            mod = "ADC_HR"

        mask = seg.segment(vol_path, progress_bar=False)

        out_mask_path = Path(ds.cache_dir) / f"{pid}_{mod}_prostate158_mask.nii.gz"
        save_mask_as_nifti(mask, vol_path, out_mask_path)

        print(f"[{i+1:02d}/{n}] {pid} | segmented {mod} -> {out_mask_path}")

    print("Done.")


if __name__ == "__main__":
    main()
