from torchvision            import transforms as T
from torchvision.transforms import functional as TF
import torch
import torch.nn.functional as F
import numpy as np
import random

def _ensure_tuple3(x):
    return (x, x, x) if isinstance(x, int) else tuple(x)

def to_tensor_3d_from_sitk(arr_3d):
    """
    SITK GetArrayFromImage -> numpy with shape (D,H,W). Convert to torch [1,H,W,D].
    """
    assert arr_3d.ndim == 3, f"Expected 3D array, got {arr_3d.shape}"
    D, H, W = arr_3d.shape
    x = np.nan_to_num(arr_3d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(x)                # [D,H,W]
    x = x.permute(1, 2, 0).contiguous()    # -> [H,W,D]
    x = x.unsqueeze(0)                     # -> [1,H,W,D]
    return x

def normalize_volume01(v: torch.Tensor, p_lo=1.0, p_hi=99.0):
    """
    Robust per-volume normalization to [0,1]. v: [1,H,W,D].
    """
    x = v.squeeze(0)  # [H,W,D]
    vals = x.flatten()
    lo = torch.quantile(vals, p_lo/100.0)
    hi = torch.quantile(vals, p_hi/100.0)
    if (hi - lo) > 0:
        x = (x - lo) / (hi - lo)
        x = x.clamp_(0, 1)
    else:
        # fallback: min-max
        mn = vals.min()
        mx = vals.max()
        x = (x - mn) / (mx - mn + 1e-6) if (mx > mn) else (x*0.0)
    return x.unsqueeze(0)  # [1,H,W,D]

def center_crop_3d(v: torch.Tensor, size):
    """
    Center-crop (with pad if needed). v: [1,H,W,D], size: int or (h,w,d).
    """
    h, w, d = _ensure_tuple3(size)
    _, H, W, D = v.shape

    sh = max(0, (H - h)//2); eh = sh + min(h, H)
    sw = max(0, (W - w)//2); ew = sw + min(w, W)
    sd = max(0, (D - d)//2); ed = sd + min(d, D)

    out = v[:, sh:eh, sw:ew, sd:ed]

    ph = max(0, h - out.shape[1])
    pw = max(0, w - out.shape[2])
    pd = max(0, d - out.shape[3])
    if ph or pw or pd:
        # pad: (D_right, D_left, W_right, W_left, H_right, H_left)
        pad = (0, pd, 0, pw, 0, ph)
        out = F.pad(out, pad, mode="constant", value=0.0)
    return out


class RandomJitteredCenterCrop:
    """
    Crop a fixed-size box whose center is randomly jittered from the image center.
    You can control the max jitter in pixels or as a fraction of the image size.
    """
    def __init__(self, size, max_offset_px=None, max_offset_frac=None):
        self.size = (size, size)
   
        assert (max_offset_px is not None) ^ (max_offset_frac is not None), \
            "Specify exactly one of max_offset_px or max_offset_frac"
            
        self.max_offset_px   = max_offset_px
        self.max_offset_frac = max_offset_frac

    def __call__(self, img):
        w, h   = img.size        
        ch, cw = self.size
        cx, cy = w // 2, h // 2

        # max jitter in pixels
        if self.max_offset_px is not None:
            ox = oy = int(self.max_offset_px)
        else:
            ox = int(self.max_offset_frac * w)
            oy = int(self.max_offset_frac * h)

        # sample jitter
        dx = random.randint(-ox, ox)
        dy = random.randint(-oy, oy)

        # top-left corner of crop (clamped to image bounds)
        left = max(0, min(cx - cw // 2 + dx, w - cw))
        top  = max(0, min(cy - ch // 2 + dy, h - ch))

        return TF.crop(img, top, left, ch, cw)
    
def resize_3d(v: torch.Tensor, size):
    """
    v: [1,H,W,D] -> [1,h,w,d] with trilinear.
    """
    h, w, d = _ensure_tuple3(size)
    return F.interpolate(v.unsqueeze(0), size=(h, w, d), mode="trilinear", align_corners=False).squeeze(0)

def make_lowres_from_hr(hr: torch.Tensor, downsample: int):
    """
    hr: [1,h,w,d] -> downsample then upsample back to [1,h,w,d].
    """
    if downsample <= 1:
        return hr
    h, w, d = hr.shape[1:]
    ds = (max(1, h//downsample), max(1, w//downsample), max(1, d//downsample))
    lr = F.interpolate(hr.unsqueeze(0), size=ds, mode="trilinear", align_corners=False).squeeze(0)
    lr_up = F.interpolate(lr.unsqueeze(0), size=(h, w, d), mode="trilinear", align_corners=False).squeeze(0)
    return lr_up

def _pre(arr):
    # SITK (D,H,W) -> torch [1,H,W,D] normalized
    t = to_tensor_3d_from_sitk(arr)
    t = normalize_volume01(t)
    return t


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
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ])
    def get_highres(self):
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ]) 
    def get_t2w(self):  
        return T.Compose([
            T.CenterCrop(self.adc_size*2),
            T.Resize(self.adc_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ]) 
        
    def get_all_transforms(self):
        return {
            'ADC_input'    : self.get_highres(),
            'ADC_condition': self.get_lowres(),
            'T2W':           self.get_t2w(),
            'HBV':           self.get_highres()
        }
        
class TransformsOffsetT2W(Transforms):
    def __init__(self, adc_size, downsample, max_offset_px):
        super().__init__(adc_size, downsample)
        self.max_offset_px = max_offset_px
        
    def get_t2w(self):  
        return T.Compose([
            RandomJitteredCenterCrop(self.adc_size*2, max_offset_px=self.max_offset_px),
            T.Resize(self.adc_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ]) 

class TransformsUpsample(Transforms):
    def __init__(self, *args):
        super().__init__(*args)
    
    def get_upsampled_nearest(self):
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.Resize(self.adc_size * self.downsample, interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ])
    def get_upsampled_bicubic(self):
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.Resize(self.adc_size * self.downsample, interpolation=T.InterpolationMode.BICUBIC),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ]) 
    def get_t2w(self):  
        return T.Compose([
            T.CenterCrop(self.adc_size * self.downsample),
            T.ToTensor()
        ]) 
    def get_adc(self):  
        return T.Compose([
            T.CenterCrop(self.adc_size),
            T.ToTensor()
        ]) 
    def get_all_transforms(self):
        return {
            'ADC_input'    : self.get_upsampled_bicubic(),
            'ADC_condition': self.get_upsampled_nearest(),
            'T2W':           self.get_t2w(),
            'ADC_target'   : self.get_adc(),
        }
        
class Transforms3D(Transforms):
    def __init__(self, *args):
        super().__init__(*args)

    def get_lowres(self):
        return lambda arr: center_crop_3d(_pre(arr), self.image_size)
    def get_highres(self):
        return lambda arr: make_lowres_from_hr(center_crop_3d(_pre(arr), self.image_size), self.downsample)
    def get_t2w(self):  
        return lambda arr: center_crop_3d(_pre(arr), self.image_size)
    
    
class TransformsLowField(Transforms):
    def __init__(self, *args):
        super().__init__(*args)
    
    def get_lowres(self):
        return T.Compose([
            T.CenterCrop(self.adc_size//self.downsample),
            T.Resize(self.adc_size, interpolation=T.InterpolationMode.NEAREST),
            T.Lambda(lambda img: torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255) #T.ToTensor(), 255???
        ])
        
    def get_all_transforms(self):
        return {
            'ADC_condition': self.get_lowres(),
            'ADC_input':     self.get_highres(),
            'T2W':           self.get_t2w(),
            'HBV':           self.get_lowres()
        }

def get_transforms(dims, image_size, downsample, type=None, t2w_offset=None):
    
    if type=='offset':
        transforms = TransformsOffsetT2W(image_size, downsample, t2w_offset)
    elif type=='upsample':
        transforms = TransformsUpsample(image_size, downsample) 
    elif type=='lowfield':
        transforms = TransformsLowField(image_size, downsample)
    elif dims==3:    
        transforms = Transforms3D(image_size, downsample)
    else:
        Transforms(image_size, downsample)
        
    return transforms.get_all_transforms()
    
def downsample_transform(size):  
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
    ]) 