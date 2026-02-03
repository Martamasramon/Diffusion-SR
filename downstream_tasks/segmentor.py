# segmentor_prostate158.py

from __future__ import annotations

import os
import json
import hashlib
import datetime
from pathlib import Path
from typing import Union, List, Optional

import torch
import tqdm
import httpx

from monai.inferers import SlidingWindowInferer
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import UNet

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Spacing,
    Orientation,
    ScaleIntensity,
    NormalizeIntensity,
    KeepLargestConnectedComponent,
)
from monai.utils.enums import GridSampleMode

from monai.data.meta_tensor import MetaTensor

from nibabel.nifti1 import Nifti1Image
from nibabel.processing import resample_from_to


# -----------------------------------------------------------------------------
# Simple local cache root (no project-level CACHE_ROOT dependency)
# -----------------------------------------------------------------------------
DEFAULT_CACHE_ROOT = Path(os.getenv("PROSTATE158_CACHE", "./cache_prostate158"))
DEFAULT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

ORIGINAL_VOLUME_SHAPE_KEY = "original_volume_shape"


# -----------------------------------------------------------------------------
# Zenodo helpers: fetch record & download weights (kept similar to your snippet)
# -----------------------------------------------------------------------------
def fetch_zenodo_record(cache_root: Path = DEFAULT_CACHE_ROOT) -> dict:
    """
    Retrieve Zenodo record metadata for 'Models for Prostate158'.
    Cached for 7 days to avoid repeated network calls.
    """
    TTL = datetime.timedelta(days=7)
    ZENODO_RECORD_ID = 7040585

    cached = cache_root / f"zenodo-record-{ZENODO_RECORD_ID}.json"
    if cached.exists():
        creation_time = datetime.datetime.fromtimestamp(
            cached.stat().st_mtime, tz=datetime.timezone.utc
        )
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        if now - creation_time < TTL:
            with open(cached, "r") as handle:
                return json.load(handle)

    # If cache is missing/stale, fetch from Zenodo API
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    r = httpx.get(url, timeout=60.0)
    r.raise_for_status()
    record = r.json()

    # Store record locally
    with open(cached, "w") as handle:
        json.dump(record, handle)

    return record


def retrieve_model_weights(cache_root: Path = DEFAULT_CACHE_ROOT, key: str = "anatomy.pt") -> Path:
    """
    Download Prostate158 anatomy weights from Zenodo (if not already cached),
    validate checksum, and return path to weights.
    """
    record = fetch_zenodo_record(cache_root=cache_root)

    # Find the file entry in the Zenodo record
    try:
        meta = next(x for x in record["files"] if x["key"] == key)
    except StopIteration:
        raise FileNotFoundError(f"Could not locate {key} in Zenodo record!")

    expected_md5: str = meta["checksum"].removeprefix("md5:")
    filepath = cache_root / f"prostate158-{key}"

    # If cached, verify checksum
    if filepath.exists():
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                md5.update(chunk)
        actual = md5.hexdigest()
        if actual != expected_md5:
            raise ValueError(f"Checksum mismatch: cached={actual} expected={expected_md5}")
        return filepath

    # Download weights and compute checksum on the fly
    url = meta["links"]["self"]
    md5 = hashlib.md5()
    with httpx.stream("GET", url, follow_redirects=False, timeout=None) as r:
        r.raise_for_status()
        total_bytes = int(r.headers.get("content-length", 0))
        chunk_size = 1 << 20

        with (
            open(filepath, "wb") as f,
            tqdm.tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading Prostate158 weights",
                disable=(total_bytes == 0),
            ) as pbar,
        ):
            for chunk in r.iter_bytes(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                md5.update(chunk)
                pbar.update(len(chunk))

    actual = md5.hexdigest()
    if actual != expected_md5:
        filepath.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch: downloaded={actual} expected={expected_md5}")

    return filepath


# -----------------------------------------------------------------------------
# Segmentor: inference-only Prostate158 anatomy segmentation
# -----------------------------------------------------------------------------
class Segmentor:
    """
    Inference-only Prostate158 U-Net segmentor for prostate anatomy.
    Produces a binary mask (1 = prostate, 0 = background) in the ORIGINAL input space.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        cache_root: Path = DEFAULT_CACHE_ROOT,
        weights_key: str = "anatomy.pt",
    ):
        self.device = device
        self.cache_root = cache_root

        # 3D UNet configuration (matches prostate158 anatomy model defaults)
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=[16, 32, 64, 128, 256, 512],
            strides=[2, 2, 2, 2, 2],
            num_res_units=4,
            act=Act.PRELU,
            norm=Norm.BATCH,
            dropout=0.15,
        ).to(device)

        # Load weights (download + cache if necessary)
        weights_path = retrieve_model_weights(cache_root=cache_root, key=weights_key)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.eval()

        # Sliding-window inferer to handle arbitrary 3D volume sizes
        self.inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            overlap=0.5,
            progress=False,
            sw_device=device,
            device=device,
        )

        # Preprocessing: load -> channel first -> resample -> orient -> intensity scaling
        self.input_transform = Compose(
            [
                LoadImage(image_only=True),  # keep metadata (affine, etc.)
                EnsureChannelFirst(channel_dim="no_channel"),
                Spacing(pixdim=(0.5, 0.5, 0.5), mode=GridSampleMode.BILINEAR),
                Orientation(axcodes="RAS"),
                ScaleIntensity(minv=0, maxv=1),
                NormalizeIntensity(),
            ]
        )

        # Postprocess: keep largest connected component of the prostate label
        self.klcc = KeepLargestConnectedComponent(applied_labels=[1])

    def _segment_one(self, filepath: Union[Path, str]) -> MetaTensor:
        """
        Segment a single NIfTI volume path, returning a binary mask MetaTensor
        in the original voxel grid & affine.
        """
        # Load + preprocess; returns a MetaTensor with .meta dict (affine etc.)
        input_volume: MetaTensor = self.input_transform(filepath)  # shape: (1, D, H, W)

        # Keep original shape & affine for resampling back after inference
        original_shape = input_volume.shape[1:]  # (D, H, W) excluding channel
        original_affine = input_volume.meta.get("original_affine", input_volume.meta.get("affine"))
        current_affine  = input_volume.meta.get("affine")

        # Add batch dim and move to device
        x = input_volume.unsqueeze(0).to(self.device)  # (B=1, C=1, D, H, W)

        # Run sliding-window inference
        with torch.no_grad():
            logits = self.inferer(inputs=x, network=self.model).squeeze(0)  # (3, D, H, W)

        # Convert logits -> hard labels in *resampled/oriented* space
        raw_mask = torch.argmax(logits, dim=0).cpu().to(torch.uint8).numpy()  # (D,H,W)

        # Resample mask back into original input space using nibabel
        resampled = resample_from_to(
            Nifti1Image(raw_mask, affine=current_affine),
            (original_shape, original_affine),
            order=0,            # nearest-neighbour
            mode="nearest",
        )

        # Wrap back into MetaTensor with original affine
        mask = MetaTensor(resampled.get_fdata(), meta=dict(input_volume.meta))
        mask.meta["affine"] = original_affine  # ensure original spatial frame

        # Binarise: background label = dominant class; prostate = everything else
        mask_t = torch.as_tensor(mask, dtype=torch.int64)
        labels, counts = torch.unique(mask_t, return_counts=True)
        background_label = labels[torch.argmax(counts)]
        mask_bin = torch.where(mask_t == background_label, 0, 1).to(torch.uint8)

        # Keep largest connected component of prostate class
        mask_bin = self.klcc(mask_bin.unsqueeze(0)).squeeze(0).to(torch.uint8)

        # Return as MetaTensor (keeps affine in meta for later saving/alignment)
        return MetaTensor(mask_bin, meta=mask.meta)

    def segment(
        self,
        *filepaths: Union[Path, str],
        progress_bar: bool = True,
    ) -> Union[MetaTensor, List[MetaTensor]]:
        """
        Segment one or more volumes. Returns:
          - MetaTensor if one path
          - list[MetaTensor] if many paths
        """
        iterator = tqdm.tqdm(filepaths, desc="Segmenting volumes") if progress_bar else filepaths
        out = [self._segment_one(p) for p in iterator]
        return out[0] if len(out) == 1 else out


def dice_score(mask_a: torch.Tensor, mask_b: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Dice for binary masks (0/1). Expects tensors broadcastable to same shape.
    """
    a = (mask_a > 0).float()
    b = (mask_b > 0).float()
    inter = (a * b).sum()
    return float((2 * inter + eps) / (a.sum() + b.sum() + eps))


def evaluate_segmentation_downstream(segmentor, batch, *, pred_path_key: str, gt_path_key: str):
    """
    Example downstream eval:
      - segment SR-predicted volume path
      - segment GT volume path (or use a provided GT mask path if you have one)
      - compute Dice between the two binary masks

    Assumes your batch contains NIfTI paths (strings/Paths) for volumes.
    """
    pred_paths = batch[pred_path_key]  # list[str] length B
    gt_paths   = batch[gt_path_key]    # list[str] length B

    pred_masks = segmentor.segment(*pred_paths, progress_bar=False)
    gt_masks   = segmentor.segment(*gt_paths, progress_bar=False)

    # If only one sample, wrap into list for uniformity
    if not isinstance(pred_masks, list):
        pred_masks = [pred_masks]
        gt_masks   = [gt_masks]

    dices = []
    for pm, gm in zip(pred_masks, gt_masks):
        dices.append(dice_score(torch.as_tensor(pm), torch.as_tensor(gm)))

    return dices

