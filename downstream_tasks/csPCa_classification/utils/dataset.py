import os
import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.misc_utils import resample_to_reference

class PicaiDataset(Dataset):

    def __init__(self, metadata_X_df, img_dir, labels=None, target_size=(224, 224)):
        self.metadata_X_df = metadata_X_df
        self.img_dir = img_dir
        self.labels = labels
        self.target_size = target_size

        self.metadata_cols = list(set(metadata_X_df.columns) - {'image_name'} - {'patient_id'})
        
    def __len__(self):
        return len(self.metadata_X_df)
    
    def __getitem__(self, idx):
        img_data = self.metadata_X_df.iloc[idx]
        image_name = img_data['image_name']
        patient_id = img_data['patient_id']

        t2w_img_path = f"{self.img_dir}/picai_public_images/{patient_id}/{image_name}_t2w.mha"
        hbv_img_path = f"{self.img_dir}/picai_public_images/{patient_id}/{image_name}_hbv.mha"
        adc_img_path = f"{self.img_dir}/picai_public_images/{patient_id}/{image_name}_adc.mha"
        lesion_mask_path = f"{self.img_dir}/lesion_masks/{image_name}.nii.gz"

        # Load images and lesion mask using SimpleITK
        t2w_img = sitk.ReadImage(t2w_img_path)
        hbv_img = sitk.ReadImage(hbv_img_path)
        adc_img = sitk.ReadImage(adc_img_path)
        lesion_mask = sitk.ReadImage(lesion_mask_path)

        # Resample T2W and lesion mask to match ADC/HBV DWI resolution and dimensions
        t2w_img = resample_to_reference(t2w_img, adc_img, is_mask=False)
        lesion_mask = resample_to_reference(lesion_mask, adc_img, is_mask=True)

        # Extract the slice with the largest lesion area
        lesion_mask_array = sitk.GetArrayFromImage(lesion_mask)
        # slice_areas = np.sum(lesion_mask_array > 0, axis=(1,2))
        slice_areas = lesion_mask_array.sum(axis=(1,2))
        largest_lesion_slice = np.argmax(slice_areas)
        if slice_areas.sum() == 0:
            largest_lesion_slice = lesion_mask_array.shape[0] // 2 # If no lesion, take the middle slice
            # print("No lesion found in mask, defaulting to middle slice for image:", image_name)
            

        # Extract the corresponding slices from T2W, HBV, ADC, and lesion mask
        t2w_slice = sitk.GetArrayFromImage(t2w_img)[largest_lesion_slice, :, :]
        hbv_slice = sitk.GetArrayFromImage(hbv_img)[largest_lesion_slice, :, :]
        adc_slice = sitk.GetArrayFromImage(adc_img)[largest_lesion_slice, :, :]
        lesion_mask_slice = lesion_mask_array[largest_lesion_slice, :, :]

        # Convert to PyTorch tensors and normalize images
        t2w_img = torch.tensor(t2w_slice, dtype=torch.float32)
        hbv_img = torch.tensor(hbv_slice, dtype=torch.float32)
        adc_img = torch.tensor(adc_slice, dtype=torch.float32)
        lesion_mask = torch.tensor(lesion_mask_slice, dtype=torch.float32)

        # Stack the three modalities to create a 3-channel image tensor -> (3, H, W)
        img = torch.stack([t2w_img, hbv_img, adc_img], dim=0)

        lesion_mask = lesion_mask.unsqueeze(0)  # Add channel dimension to lesion mask -> (1, H, W)

        # Add batch dimension for resizing -> (1, C, H, W)
        img = img.unsqueeze(0)
        lesion_mask = lesion_mask.unsqueeze(0)

        # Resize images and lesion mask to target size for network input
        img = F.interpolate(
            img, 
            size=self.target_size, 
            mode='bilinear', 
            align_corners=False
        )

        lesion_mask = F.interpolate(
            lesion_mask, 
            size=self.target_size, 
            mode='nearest'
        )

        # Remove batch dimension -> (C, H, W)
        img = img.squeeze(0)
        lesion_mask = lesion_mask.squeeze(0)

        metadata = img_data[self.metadata_cols].to_numpy().astype(np.float32)
        metadata = torch.from_numpy(metadata)

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.float32)

            return img, lesion_mask, metadata, label
        else:
            return img, lesion_mask, metadata