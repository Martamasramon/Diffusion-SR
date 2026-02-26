import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from utils.misc_utils import resample_to_reference
import pandas as pd
from PIL import Image
from torchvision import transforms as T

class PicaiDataset(Dataset):

    def __init__(self, metadata_X_df, img_dir, labels=None, target_size=(224, 224)):
        self.metadata_X_df = metadata_X_df
        self.img_dir   =  '/cluster/project7/backup_masramon/PI-CAI/'
        self.label_dir =  '/cluster/project7/backup_masramon/PI-CAI_annotations/lesion_human_original/'
        # self.img_dir = img_dir
        self.labels = labels
        self.target_size = target_size

        self.metadata_cols = list(set(metadata_X_df.columns) - {'image_name'} - {'patient_id'})
        
    def __len__(self):
        return len(self.metadata_X_df)
    
    def __getitem__(self, idx):
        img_data = self.metadata_X_df.iloc[idx]
        image_name = img_data['image_name']
        patient_id = img_data['patient_id']

        t2w_img_path = f"{self.img_dir}/{patient_id}/{image_name}_t2w.mha"
        hbv_img_path = f"{self.img_dir}/{patient_id}/{image_name}_hbv.mha"
        adc_img_path = f"{self.img_dir}/{patient_id}/{image_name}_adc.mha"
        lesion_mask_path = f"{self.label_dir}/{image_name}.nii.gz"

        # t2w_img_path = f"{self.img_dir}/picai_public_images/{patient_id}/{image_name}_t2w.mha"
        # hbv_img_path = f"{self.img_dir}/picai_public_images/{patient_id}/{image_name}_hbv.mha"
        # adc_img_path = f"{self.img_dir}/picai_public_images/{patient_id}/{image_name}_adc.mha"
        # lesion_mask_path = f"{self.img_dir}/lesion_masks/{image_name}.nii.gz"

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
        slice_areas = lesion_mask_array.sum(axis=(1,2))
        largest_lesion_slice = np.argmax(slice_areas)
        if slice_areas.sum() == 0:
            largest_lesion_slice = lesion_mask_array.shape[0] // 2 # If no lesion, take the middle slice            

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

        # Center crop to target size
        img = TF.center_crop(
            img,
            output_size=self.target_size
        )
        lesion_mask = TF.center_crop(
            lesion_mask, 
            output_size=self.target_size
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
        
        

class Transforms():
    def __init__(
        self,
        adc_size,
        downsample
    ):
        self.adc_size   = adc_size
        self.downsample = downsample
    
    def get_lowres(self):
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.Resize(self.adc_size//self.downsample, interpolation=T.InterpolationMode.NEAREST),
            T.Resize(self.adc_size,                  interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) 
        ])
    def get_highres(self):
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) 
        ]) 
    def get_t2w(self):  
        return T.Compose([
            T.CenterCrop(self.adc_size*2),
            T.Resize(self.adc_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ]) 
    def get_all_transforms(self):
        return {
            'ADC':         self.get_highres(),
            'T2W':         self.get_t2w(),
            'HBV':         self.get_highres(),
            'outputs':     self.get_highres(),
            'interpolate': self.get_lowres(),
        }
        
class PicaiDataset_png(Dataset):

    def __init__(self, metadata_X_df, labels=None, target_size=64, input_type='LF'):
        self.metadata_X_df = metadata_X_df
        self.img_dir   =  '/cluster/project7/backup_masramon/IQT/PICAI/'
        self.label_dir =  '/cluster/project7/backup_masramon/PI-CAI_annotations/lesion_human_original/'
        self.labels = labels
        self.target_size = target_size
        self.input_type  = input_type

        self.metadata_cols = list(set(metadata_X_df.columns) - {'image_name'} - {'patient_id'})
        self.slice_list    = pd.read_csv('/cluster/project7/backup_masramon/IQT/PICAI/lesion_slices.csv')
        self.transforms    = Transforms(adc_size=target_size, downsample=2).get_all_transforms()
        
    def __len__(self):
        return len(self.metadata_X_df)
    
    def __getitem__(self, idx):
        img_data = self.metadata_X_df.iloc[idx]
        patient_id = img_data['patient_id']
        slice_num  = self.slice_list[self.slice_list['name'] == patient_id]['slice'].values[0]
        
        if self.input_type in ['LF', 'interpolate']:
            folder_suf = "_lowfield"
            hbv_suf = ""
        elif self.input_type == 'outputs':
            folder_suf = "_pred"
            hbv_suf = ""
        else:
            folder_suf = ""
            hbv_suf = ""

        t2w_img_path = f"{self.img_dir}/T2W{folder_suf}/{patient_id}_{slice_num}.png"
        hbv_img_path = f"{self.img_dir}/HBV{hbv_suf}/{patient_id}_{slice_num}.png"
        adc_img_path = f"{self.img_dir}/ADC{folder_suf}/{patient_id}_{slice_num}.png"
        lesion_mask_path = f"{self.img_dir}/Lesions/{patient_id}_{slice_num}.png"
        
        # Load images and lesion mask using SimpleITK
        t2w_img     = Image.open(t2w_img_path).convert('L')
        hbv_img     = Image.open(hbv_img_path).convert('L')
        adc_img     = Image.open(adc_img_path).convert('L')
        lesion_mask = Image.open(lesion_mask_path).convert('L')

        # Convert to PyTorch tensors and normalize images
        if self.input_type == 'LF' or self.input_type == 'HF':
            t2w         = self.transforms['T2W'](t2w_img)
            hbv         = self.transforms['HBV'](hbv_img)
            adc         = self.transforms['ADC'](adc_img)
        elif self.input_type == 'interpolate':
            t2w         = self.transforms['interpolate'](t2w_img)
            hbv         = self.transforms['HBV'](hbv_img)
            adc         = self.transforms['interpolate'](adc_img)
        elif self.input_type == 'outputs':
            t2w         = self.transforms['outputs'](t2w_img)
            hbv         = self.transforms['HBV'](hbv_img)
            adc         = self.transforms['outputs'](adc_img)
            
        lesion_mask = self.transforms['ADC'](lesion_mask)    
        lesion_mask = (lesion_mask > 0.5).float()
            
        # Stack the three modalities to create a 3-channel image tensor -> (3, H, W)
        img = torch.cat([t2w, hbv, adc], dim=0)
        img = img.squeeze() 

        lesion_mask = lesion_mask.unsqueeze(0)  # Add channel dimension to lesion mask -> (1, H, W)
        
        metadata = img_data[self.metadata_cols].to_numpy().astype(np.float32)
        metadata = torch.from_numpy(metadata)

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.float32)

            return img, lesion_mask, metadata, label
        else:
            return img, lesion_mask, metadata