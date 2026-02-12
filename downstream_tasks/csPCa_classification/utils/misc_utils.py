import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import SimpleITK as sitk

class PSADImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing PSAD values using PSA and Prostate Volume with the formula: PSAD = PSA / Prostate Volume
    """

    def __init__(self, psa_idx, prostate_volume_idx, psad_idx):
        self.psa_idx = psa_idx
        self.prostate_volume_idx = prostate_volume_idx
        self.psad_idx = psad_idx

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy().astype(float)
        
        psa = X[:, self.psa_idx]
        prostate_volume = X[:, self.prostate_volume_idx]
        psad = X[:, self.psad_idx]

        mask = (np.isnan(psad)) & (~np.isnan(psa)) & (~np.isnan(prostate_volume)) & (prostate_volume > 0)

        psad_value = psa[mask] / prostate_volume[mask]
        X[mask, self.psad_idx] = np.round(psad_value, 2)

        return X
    
def resample_to_reference(moving_image, reference_image, is_mask=False):
    """
    :param moving_image: sitk.Image to be resampled (T2W or Mask)
    :param reference_image: sitk.Image to which the moving_image will be resampled (ADC/HBV DWI)
    :param is_mask: True if segmenting mask, False if T2W image
    """

    resampler = sitk.ResampleImageFilter()

    # Use reference image grid (spacing, origin, direction, and size)
    resampler.SetReferenceImage(reference_image)

    # Set interpolation method
    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    # Set identity transform (no rotation or translation)
    resampler.SetTransform(sitk.Transform())

    # Set default pixel value for areas outside the original image
    if is_mask:
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(0.0)

    # Resample the moving image
    resampled_image = resampler.Execute(moving_image)
    
    return resampled_image