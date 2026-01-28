from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import nibabel as nib

import sys
sys.path.append('../models')
from VAE import encode_latent, decode_latent
sys.path.append('../')
from transforms import get_transforms

def _natural_key(s: str):
    """Natural sort for filenames like slice_2.png < slice_10.png."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _load_grayscale(path: Path) -> np.ndarray:
    """Load a single 2D slice as uint8 array (H, W)."""
    img = Image.open(path).convert("L")
    return np.asarray(img)

def save_mask_as_nifti(mask_meta_tensor, ref_nii_path: str | Path, out_path: str | Path) -> None:
    """
    Save Segmentor output (MetaTensor or tensor-like) as NIfTI using the reference affine.
    """
    ref = nib.load(str(ref_nii_path))
    data = torch.as_tensor(mask_meta_tensor).detach().cpu().numpy().astype("uint8")
    nii = nib.Nifti1Image(data, affine=ref.affine, header=ref.header)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(out_path))
    
class PatientToNifti(Dataset):
    """
    Dataset that converts per-patient stacks of 2D images into cached NIfTI volumes.

    Returns (per item):
      {
        "patient_id": <str>,
        "T2W_path": <str>,          # path to cached .nii.gz
        "ADC_path": <str>,          # path to cached .nii.gz
      }
    root_dir = ".../{patient_id}_{slice_num}.png"
    """

    def __init__(
        self,        
        diffusion,
        device,
        *,
        csv_path:   str = 'patient_list_ALL.csv',
        use_T2W:    bool = True,
        valid_exts: Sequence[str]   = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        pixdim:     Sequence[float] = (0.5, 0.5, 0.5),   # (dx, dy, dz) in mm
        affine:     Optional[np.ndarray] = None,
        controlnet: bool = False,
        vae:        Optional[torch.nn.Module] = None,
        recompute_cache: bool = False,
        delete_cache:    bool = False,
    ):
        self.root_dir = Path('/cluster/project7/backup_masramon/IQT/PICAI/')
        self.paths_by_mod = {
            "ADC_LR":  self.root_dir / "ADC_lowfield",
            "ADC_HR":  self.root_dir / "ADC",
            "T2W_LR":  self.root_dir / "T2W_lowfield",
            "T2W_HR":  self.root_dir / "T2W",
        }
        csv_dir = '/cluster/project7/ProsRegNet_CellCount/Dataset_preparation/CSV/'
        self.patient_ids = pd.read_csv(csv_dir + csv_path)['SID'].astype(str).tolist()

        self.cache_dir = Path("/cluster/project7/backup_masramon/IQT/PICAI/temp/")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.valid_exts = valid_exts
        self.pixdim = tuple(float(x) for x in pixdim)
        if affine is None:
            # Simple diagonal affine with voxel sizes; origin at (0,0,0)
            self.affine = np.array(
                [
                    [self.pixdim[0], 0, 0, 0],
                    [0, self.pixdim[1], 0, 0],
                    [0, 0, self.pixdim[2], 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
        else:
            self.affine = np.asarray(affine, dtype=np.float32)

        self.diffusion  = diffusion
        self.device     = device
        self.vae        = vae  
        self.use_T2W    = use_T2W    
        self.controlnet = controlnet

        self.recompute_cache = recompute_cache
        self.delete_cache    = delete_cache
        self.transforms = get_transforms(2, 64, 2, type='lowfield')
                
    def __len__(self) -> int:
        return len(self.patient_ids)

    def _resolve_slice_paths(self, patient_id, mod_key='ADC_LR' ) -> List[Path]:
        """
        Find and sort all slice files for a patient from a single flat directory.
        Expected filename format:
            <patientID>_<sliceIndex>.<ext>
        """
        folder = self.paths_by_mod[mod_key]   
        paths = [
            p for p in folder.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in self.valid_exts
                and p.name.startswith(f"{patient_id}_")
            )
        ]

        if len(paths) == 0:
            raise FileNotFoundError(f"No slices found for patient {patient_id!r}.")

        def slice_index(path: Path) -> int:
            return int(path.stem.split("_")[1])

        paths = sorted(paths, key=slice_index)
        return paths

    def _build_volume(self, slice_paths: List[Path]) -> np.ndarray:
        """Stack slices into (H, W, D). Ensures consistent H/W."""
        slices = [_load_grayscale(p) for p in slice_paths]
        h0, w0 = slices[0].shape
        for k, s in enumerate(slices[1:], start=1):
            if s.shape != (h0, w0):
                raise ValueError(f"Slice size mismatch at index {k}: got {s.shape}, expected {(h0, w0)}")
        vol = np.stack(slices, axis=-1)  # (H, W, D)
        vol = vol.astype(np.uint8)
        return vol

    def _cache_path(self, patient_id, img_type='ADC_LR') -> Path:
        return self.cache_dir / f"{patient_id}_{img_type}.nii.gz"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id = self.patient_ids[idx]
        
        wanted = {
            "ADC_LR": True,
            "ADC_HR": True,
            "T2W_LR": bool(self.use_T2W),
            "T2W_HR": bool(self.use_T2W),
        }

        output = {'patient_id': patient_id}
        for key, do_it in wanted.items():
            if not do_it:
                continue

            out_nii = self._cache_path(patient_id, img_type=key)
            
            if (not self.recompute_cache) and out_nii.exists():  
                output[key] = str(out_nii)
                continue
        
            slice_paths = self._resolve_slice_paths(patient_id)
            vol         = self._build_volume(slice_paths)
            nib.save(nib.Nifti1Image(vol, affine=self.affine), out_nii)
            output[key] = str(out_nii)
            
        return output

    def set_image_format(self, img):
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
            
        img = np.transpose(img, (2, 0, 1)) # (B, H, W)    
        return torch.from_numpy(img).unsqueeze(1).to(self.device)  
     
    def _sample_slices(self, lowres_batch: torch.Tensor, t2w_batch: Optional[torch.Tensor]) -> torch.Tensor:
        """
        lowres_batch: (B, 1, H, W)
        t2w_batch:    (B, 1, H, W) or None
        returns:      (B, 1, H, W) (or whatever your model returns; we squeeze later)
        """
        kwargs = {"batch_size": lowres_batch.shape[0]}
        if self.use_T2W:
            kwargs["control" if self.controlnet else "t2w"] = t2w_batch

        with torch.no_grad():
            return self.diffusion.sample(lowres_batch, **kwargs)
               
    def generate_sr_nifti(
        self,
        idx: int,
        *,
        out_dtype: np.dtype = np.float32,
        clamp01: bool = True,
    ) -> str:
        """
        Build per-slice tensors using your original PIL+transforms pipeline,
        run diffusion SR, then save as NIfTI.

        Returns: filepath to saved SR NIfTI.
        """
        patient_id = self.patient_ids[idx]

        # --- Resolve slice lists (sorted) using the SAME naming across folders ---
        adc_lr_paths  = self._resolve_slice_paths(patient_id, "ADC_LR")
        t2w_folder    = self.paths_by_mod["T2W_LR"]

        adc_cond_slices = []
        t2w_cond_slices = [] if self.use_T2W else None

        for p_lr in adc_lr_paths:
            # ADC lowfield -> ADC_condition
            adc_lr_img = Image.open(p_lr).convert("L")
            adc_cond = self.transforms["ADC_condition"](adc_lr_img)  # (1,H,W) or (H,W) depending on your pipeline
            adc_cond_slices.append(adc_cond)

            # Optional T2W lowfield -> T2W_condition (mirrors your original snippet)
            if self.use_T2W:
                p_t2w = t2w_folder / p_lr.name  # same "<SID>_<slice>.png" name
                t2w_img = Image.open(p_t2w).convert("L")
                t2w_cond = self.transforms["T2W_condition"](t2w_img)
                t2w_cond_slices.append(t2w_cond)

        # Stack into batch tensors: (D, C, H, W)
        adc_condition = torch.stack(adc_cond_slices, dim=0).to(self.device)

        if self.use_T2W:
            t2w_condition = torch.stack(t2w_cond_slices, dim=0).to(self.device)
        else:
            t2w_condition = None

        # --- Optional VAE latent path (if your diffusion runs in latent space) ---
        if self.vae is not None:
            self.vae.eval()
            adc_condition, _ = encode_latent(adc_condition, self.vae)
            if t2w_condition is not None:
                t2w_condition, _ = encode_latent(t2w_condition, self.vae)

        # --- Diffusion sampling (chunked over slices) ---
        with torch.no_grad():
            kwargs = {"batch_size": adc_condition.shape[0]}
            if t2w_condition is not None:
                kwargs["control" if self.controlnet else "t2w"] = t2w_condition

            pred = self.diffusion.sample(adc_condition, **kwargs)  # expected (B,C,H,W)

        if self.vae is not None:
            pred = decode_latent(pred, self.vae)

        # Use channel 0 if multi-channel
        if pred.ndim == 4:
            pred = pred[:, 0, :, :]  # (D,H,W)

        pred = pred.detach().float().cpu()
        if clamp01:
            pred = pred.clamp(0.0, 1.0)

        # Convert to (H,W,D) for NIfTI
        sr_vol = pred.permute(1, 2, 0).numpy().astype(out_dtype)

        out_path = self._cache_path(patient_id, "ADC_SR")
        sr_nii = nib.Nifti1Image(sr_vol, affine=self.affine)
        nib.save(sr_nii, str(out_path))

        return str(out_path)
