import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.misc_utils import PSADImputer

def create_preprocess_pipeline():

    constant_imputer = SimpleImputer(strategy='constant', fill_value='NA')
    log_transformer = FunctionTransformer(np.log1p, validate=False)
    standard_scaler = StandardScaler()

    psa_vol_imputer = ColumnTransformer(
        transformers=[
            ('psa_vol', SimpleImputer(strategy='median'), [0, 1])
        ],
        remainder='passthrough'
    )

    ratio_imputer = PSADImputer(psa_idx=0, prostate_volume_idx=1, psad_idx=2)

    num_transform = Pipeline(steps=[
        ('psa_vol_imputer', psa_vol_imputer),
        ('psad_imputer', ratio_imputer),
        ('log_transformer', log_transformer),
        ('scaler', standard_scaler)
    ])

    age_transform = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    histopath_transform = Pipeline(steps=[
        ('imputer', constant_imputer),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    cat_transform = ColumnTransformer(
        transformers=[
            ('histopath', histopath_transform, ['histopath_type']),
            ('center', OneHotEncoder(handle_unknown='ignore'), ['center'])
        ],
        remainder='passthrough'
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transform, ['psa', 'prostate_volume', 'psad']),
            ('age', age_transform, ['patient_age']),
            ('cat', cat_transform, ['histopath_type', 'center'])
        ],
        remainder='passthrough'
    )

    return preprocessor

def preprocess_metadata(metadata_csv_path, random_state=13):

    # Load metadata
    metadata = pd.read_csv(metadata_csv_path)

    # Create image_name column and drop unnecessary columns
    metadata['image_name'] = metadata.apply(lambda row: f"{row['patient_id']}_{row['study_id']}", axis=1)
    metadata = metadata.drop(columns=['study_id', 'mri_date', 'lesion_GS', 'lesion_ISUP', 'case_ISUP'])

    # Split into train, validation, and test sets with stratification and grouping by patient_id to prevent data leakage
    sgkf_1 = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=13
    )
    train_val_idx, test_idx = next(
        sgkf_1.split(X=np.zeros(len(metadata['case_csPCa'])),
                    y=metadata['case_csPCa'],
                    groups=metadata['patient_id'])
    )

    train_val_metadata = metadata.iloc[train_val_idx]
    test_metadata = metadata.iloc[test_idx]

    sgkf_2 = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=13
    )

    train_idx, val_idx = next(
        sgkf_2.split(X=np.zeros(len(train_val_metadata['case_csPCa'])), 
                    y=train_val_metadata['case_csPCa'], 
                    groups=train_val_metadata['patient_id']) ) 

    train_metadata = train_val_metadata.iloc[train_idx] 
    val_metadata = train_val_metadata.iloc[val_idx] 

    # train_val_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=random_state, stratify=metadata['case_csPCa'])
    # train_metadata, val_metadata = train_test_split(train_val_metadata, test_size=0.1, random_state=random_state, stratify=train_val_metadata['case_csPCa'])

    # Separate features and labels
    train_metadata_X = train_metadata.drop(columns=['case_csPCa']).reset_index(drop=True)
    train_metadata_y = train_metadata['case_csPCa'].reset_index(drop=True)

    val_metadata_X = val_metadata.drop(columns=['case_csPCa']).reset_index(drop=True)
    val_metadata_y = val_metadata['case_csPCa'].reset_index(drop=True)

    test_metadata_X = test_metadata.drop(columns=['case_csPCa']).reset_index(drop=True)
    test_metadata_y = test_metadata['case_csPCa'].reset_index(drop=True)

    # Create preprocessing pipeline and fit on training data, then transform validation and test data
    preprocessor = create_preprocess_pipeline()

    train_metadata_X_processed = preprocessor.fit_transform(train_metadata_X)
    train_metadata_X_df = pd.DataFrame(train_metadata_X_processed)
    train_metadata_X_df.columns = ['psa', 'prostate_volume', 'psad', 'patient_age'] + list(preprocessor.named_transformers_['cat'].get_feature_names_out()) + ['patient_id'] + ['image_name']
    train_metadata_X_df.insert(0, 'image_name', train_metadata_X_df.pop('image_name'))
    train_metadata_X_df.insert(0, 'patient_id', train_metadata_X_df.pop('patient_id'))

    val_metadata_X_processed = preprocessor.transform(val_metadata_X)
    val_metadata_X_df = pd.DataFrame(val_metadata_X_processed)
    val_metadata_X_df.columns = ['psa', 'prostate_volume', 'psad', 'patient_age'] + list(preprocessor.named_transformers_['cat'].get_feature_names_out()) + ['patient_id'] + ['image_name']
    val_metadata_X_df.insert(0, 'image_name', val_metadata_X_df.pop('image_name'))
    val_metadata_X_df.insert(0, 'patient_id', val_metadata_X_df.pop('patient_id'))
    
    test_metadata_X_processed = preprocessor.transform(test_metadata_X)
    test_metadata_X_df = pd.DataFrame(test_metadata_X_processed)
    test_metadata_X_df.columns = ['psa', 'prostate_volume', 'psad', 'patient_age'] + list(preprocessor.named_transformers_['cat'].get_feature_names_out()) + ['patient_id'] + ['image_name']
    test_metadata_X_df.insert(0, 'image_name', test_metadata_X_df.pop('image_name'))
    test_metadata_X_df.insert(0, 'patient_id', test_metadata_X_df.pop('patient_id'))

    # Map labels to binary values
    train_metadata_y = train_metadata_y.map({'NO': 0, 'YES': 1})
    val_metadata_y = val_metadata_y.map({'NO': 0, 'YES': 1})
    test_metadata_y = test_metadata_y.map({'NO': 0, 'YES': 1})

    print("Preprocessing complete. Processed metadata shapes:")
    print(f"Train metadata: {train_metadata_X_df.shape}, {train_metadata_y.shape}")
    print(f"Validation metadata: {val_metadata_X_df.shape}, {val_metadata_y.shape}")
    print(f"Test metadata: {test_metadata_X_df.shape}, {test_metadata_y.shape}")

    return train_metadata_X_df, train_metadata_y, val_metadata_X_df, val_metadata_y, test_metadata_X_df, test_metadata_y