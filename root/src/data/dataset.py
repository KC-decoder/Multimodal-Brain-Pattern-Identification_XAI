import os
import yaml
import pandas as pd
import random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from utils.data_utils import (
    baseline_correction,
    normalize_signal,
    apply_notch_filter,
    smooth_spectrogram,
    resample_spectrogram,
    handle_nan,
    pad_or_truncate,
    calculate_differential_signals,
    butter_bandpass_filter,
    denoise_filter,
    select_and_map_channels,
    mirror_eeg,
    normalize,
    labels_to_probabilities,
    load_train_eeg_frame,
    load_train_spectr_frame,
    plot_spectrograms,
    seed_everything,
    feature_to_index
)
# Load the config file using the config_loader module
from utils.cfg_utils import CFG, _Logger, _seed_everything
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch import Tensor
from tqdm import tqdm
from scipy.signal import butter, lfilter
from utils.cfg_utils import CFG




"""Data Transformer for Dilated Inception WaveNet and DiffEEG."""    
            
            
            
            
            
            
class _EEGTransformer(object):
    """Data transformer for raw EEG signals."""

    def __init__(
        self,
        n_feats: int,
        apply_chris_magic_ch8: bool = False,
        normalize: bool = True,
        apply_butter_lowpass_filter: bool = True,
        apply_mu_law_encoding: bool = False,
        downsample: Optional[int] = None,
    ) -> None:
        self.n_feats = n_feats
        self.apply_chris_magic_ch8 = apply_chris_magic_ch8
        self.normalize = normalize
        self.apply_butter_lowpass_filter = apply_butter_lowpass_filter
        self.apply_mu_law_encoding = apply_mu_law_encoding
        self.downsample = downsample
        if apply_chris_magic_ch8:
            required_channels = CFG.feats 
        else:
            required_channels = CFG.channel_feats  # 19-channel names

        self.FEAT2CODE = {f: i for i, f in enumerate(required_channels)}

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply transformation on raw EEG signals.
        
        Args:
            x: raw EEG signals, with shape (L, C)

        Return:
            x_: transformed EEG signals
        """
        x_ = x.copy()

        if self.apply_chris_magic_ch8:
            x_ = self._apply_chris_magic_ch8(x_)
        else:
            # Select 19 canonical EEG channels
            selected_indices = list(self.FEAT2CODE.values())
            x_ = x_[:, selected_indices]  # (L, 19)

        if self.normalize:
            x_ = np.clip(x_, -1024, 1024)
            x_ = np.nan_to_num(x_, nan=0) / 32.0

        if self.apply_butter_lowpass_filter:
            x_ = self._butter_lowpass_filter(x_) 

        if self.apply_mu_law_encoding:
            x_ = self._quantize_data(x_, 1)

        if self.downsample is not None:
            x_ = x_[::self.downsample, :]

        return x_

    def _apply_chris_magic_ch8(self, x: np.ndarray) -> np.ndarray:
        """Generate features based on Chris' magic formula.""" 
        x_tmp = np.zeros((CFG.EEG_PTS, self.n_feats), dtype="float32")

        # Generate features
        x_tmp[:, 0] = x[:, self.FEAT2CODE["Fp1"]] - x[:, self.FEAT2CODE["T3"]]
        x_tmp[:, 1] = x[:, self.FEAT2CODE["T3"]] - x[:, self.FEAT2CODE["O1"]]
        
        x_tmp[:, 2] = x[:, self.FEAT2CODE["Fp1"]] - x[:, self.FEAT2CODE["C3"]]
        x_tmp[:, 3] = x[:, self.FEAT2CODE["C3"]] - x[:, self.FEAT2CODE["O1"]]
        
        x_tmp[:, 4] = x[:, self.FEAT2CODE["Fp2"]] - x[:, self.FEAT2CODE["C4"]]
        x_tmp[:, 5] = x[:, self.FEAT2CODE["C4"]] - x[:, self.FEAT2CODE["O2"]]
        
        x_tmp[:, 6] = x[:, self.FEAT2CODE["Fp2"]] - x[:, self.FEAT2CODE["T4"]]
        x_tmp[:, 7] = x[:, self.FEAT2CODE["T4"]] - x[:, self.FEAT2CODE["O2"]]

        return x_tmp

    def _butter_lowpass_filter(self, data, cutoff_freq=20, sampling_rate=200, order=4):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filtered_data = lfilter(b, a, data, axis=0)

        return filtered_data
                
    def _quantize_data(self, data, classes):
        mu_x = self._mu_law_encoding(data, classes)
        
        return mu_x

    def _mu_law_encoding(self, data, mu):
        mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)

        return mu_x
    
    
    
class EEGDataset(Dataset):
    """Dataset for pure raw EEG signals.

    Args:
        data: processed data
        split: data split

    Attributes:
        _n_samples: number of samples
        _infer: if True, the dataset is constructed for inference
            *Note: Ground truth is not provided.
    """

    def __init__(
        self,
        data: Dict[str,  Any],
        split: str,
        **dataset_cfg: Any,
    ) -> None:
        self.metadata = data["meta"]
        self.all_eegs = data["eeg"]
        self.dataset_cfg = dataset_cfg

        # Raw EEG data transformer
        self.eeg_params = dataset_cfg["eeg"]
        self.eeg_trafo = _EEGTransformer(**self.eeg_params)

        self._set_n_samples()
        self._infer = True if split == "test" else False

        self._stream_X = True if self.all_eegs is None else False
        self._X, self._y = self._transform()

    def _set_n_samples(self) -> None:
        assert len(self.metadata) == self.metadata["eeg_id"].nunique()
        self._n_samples = len(self.metadata)

    def _transform(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Transform feature and target matrices."""
        if self.eeg_params["downsample"] is not None:
            eeg_len = int(CFG.EEG_PTS / self.eeg_params["downsample"])
        else:
            eeg_len = int(CFG.EEG_PTS)
        if not self._stream_X:
            X = np.zeros((self._n_samples, eeg_len, self.eeg_params["n_feats"]), dtype="float32")
        else:
            X = None
        y = np.zeros((self._n_samples, CFG.N_CLASSES), dtype="float32") if not self._infer else None

        for i, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            # Process raw EEG signals
            if not self._stream_X:
                # Retrieve raw EEG signals
                eeg = self.all_eegs[row["eeg_id"]]
                # print(f"Shape of raw eeg: {eeg.shape}")
                # Apply EEG transformer
                x = self.eeg_trafo.transform(eeg)

                X[i] = x

            if not self._infer:
                y[i] = row[CFG.TGT_VOTE_COLS] 

        return X, y

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._X is None:
            # Load data if needed...
            pass
        else:
            x = self._X[idx, ...]  # Shape: (2000, 8) (Incorrect format)
        
        # Convert to tensor and permute dimensions
        x = torch.tensor(x, dtype=torch.float32).permute(1, 0)  # Now (8, 2000)

        data_sample = {"x": x}
        
        if not self._infer:
            data_sample["y"] = torch.tensor(self._y[idx, :], dtype=torch.float32)

        return data_sample
    
class DummyEEGDataset(Dataset):
    """Dataset containing 1 EEG sample per class (6 total)."""
    def __init__(self, samples):
        self.samples = samples  # List of (x, y) tuples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return {"x": x, "y": y}    
    
    
class CombinedEEGDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return {"x": x, "y": y}
    
    


# # HMS_EEG_Dataset class
# class HMS_EEG_Dataset(Dataset):
#     def __init__(self, train_ids, cfg=cfg, training_flag=False, shuffle=False):
#         super(HMS_EEG_Dataset, self).__init__()
#         self.train_ids = train_ids
#         self.cfg = cfg
#         self.training_flag = training_flag
#         self.shuffle = shuffle

#         # Set the random seed for reproducibility
#         seed_everything()
#         # Feature to index mapping from CFG
#         self.feature_to_index = feature_to_index
#         self.differential_channels_start_index = len(feature_to_index)

#     def __getitem__(self, idx):
#         row = self.train_ids.iloc[idx]
#         # Separate the label from the data
#         label_name = row['expert_consensus']  # Assuming 'expert_consensus' column contains the label names
        
#         # Drop the label column from the row to avoid any further use
#         row_new = row.drop('expert_consensus')
        
#         label_idx = self.cfg['name2label'][label_name]
#         label = labels_to_probabilities(label_idx, self.cfg['n_classes'])      

#         # Process the data independently
#         data = self.single_map_func(row_new, self.training_flag, idx)


#         # Adding channel dimension for EEG data
#         data = data[np.newaxis, ...]  # Now shape is [1, 37, 3000] or similar

#         # Return the processed data and the separated label
#         return data.astype(np.float32), label

#     def __len__(self):
#         return len(self.train_ids)

#     def single_map_func(self, row, is_training, idx):
#         data = self.get_eeg(row, is_training)
        
#         # Check if data is a tuple (which should no longer be the case)
#         if isinstance(data, tuple):
#             print(f"Tuple detected in single_map_function. Type: {type(data)}")
#             for i, elem in enumerate(data):
#                 if isinstance(elem, np.ndarray):
#                     print(f"Element {i} shape: {elem.shape}")
#                 else:
#                     print(f"Element {i} is not a NumPy array, but a {type(elem)}")  
        
        
        
#         data = handle_nan(data)
#         if data.size == 0:
#             # Skip this sample if data is empty
#             return np.zeros((self.cfg['in_channels'] + len(self.cfg['map_features']), self.cfg['fixed_length']))
#         data = calculate_differential_signals(data)
#         data = normalize(data) 
#         data = select_and_map_channels(data, self.cfg['eeg_features'], self.differential_channels_start_index)  
#         data = pad_or_truncate(data, self.cfg['fixed_length'])

        
#         return data

#     def get_eeg(self, row, is_training, flip=False):
#         eeg_id = row['eeg_id']
#         eeg = load_train_eeg_frame(eeg_id)

#         waves = eeg.values.T

#         if self.cfg['AUGMENT']:
#             waves = mirror_eeg(waves)

#         waves = butter_bandpass_filter(waves, self.cfg)
        
#         # Check if waves is a tuple
#         if isinstance(waves, tuple):
#             print(f"Tuple detected in get_eeg for EEG ID {eeg_id}. Type: {type(waves)}")

#         return waves


# # HMS_Spectrogram_Dataset class
# class HMS_Spectrogram_Dataset(Dataset):
#     def __init__(self, train_ids, cfg, augmentations=None, plot=False):
#         super(HMS_Spectrogram_Dataset, self).__init__()
#         self.train_ids = train_ids
#         self.cfg = cfg
#         self.augmentations = augmentations
#         self.plot = plot  # Flag to control plotting

#         # Define a transform to resize the spectrogram to 224x224
#         self.transform = T.Compose([
#             T.ToTensor(),
#             T.Resize((224, 224)),  # Resize to 224x224 for ViT compatibility
#             T.Normalize(mean=[0.5], std=[0.5])  # Normalize the spectrogram
#         ])

#     def __len__(self):
#         return len(self.train_ids)

#     def __getitem__(self, idx):
#         row = self.train_ids.iloc[idx]
#         spec_id = row['spectrogram_id']
#         raw_spectrogram = load_train_spectr_frame(spec_id)
#         label_name = row['expert_consensus']
#         label_idx = self.cfg['name2label'][label_name]
#         label = labels_to_probabilities(label_idx, self.cfg['n_classes'])
#         offset = row.get("spectrogram_label_offset_seconds", None)

#         if isinstance(raw_spectrogram, pd.DataFrame):
#             raw_spectrogram = raw_spectrogram.to_numpy()

#         if offset is not None:
#             offset = offset // 2
#             basic_spectrogram = raw_spectrogram[:, offset:offset + 300]
#             pad_size = max(0, 300 - basic_spectrogram.shape[1])
#             basic_spectrogram = np.pad(basic_spectrogram, ((0, 0), (0, pad_size)), mode='constant')
#         else:
#             basic_spectrogram = raw_spectrogram

#         spectrogram = basic_spectrogram.T
#         processed_spectrogram = pad_or_truncate(spectrogram, self.cfg['image_size'])
#         processed_spectrogram = handle_nan(processed_spectrogram)
#         processed_spectrogram = baseline_correction(processed_spectrogram)
#         processed_spectrogram = apply_notch_filter(processed_spectrogram)
#         processed_spectrogram = smooth_spectrogram(processed_spectrogram)
#         processed_spectrogram = normalize_signal(processed_spectrogram)
#         processed_spectrogram = resample_spectrogram(processed_spectrogram, self.cfg['image_size'])

#         # Expand to 3 channels for compatibility with ViT
#         processed_spectrogram = np.tile(processed_spectrogram[..., None], (1, 1, 3))

#         if self.plot:
#             plot_spectrograms(basic_spectrogram, processed_spectrogram, self.train_ids.index.tolist(), num_labels=10)

#         # Apply transformations and augmentations
#         if self.augmentations:
#             processed_spectrogram = (processed_spectrogram * 255).astype(np.uint8)
#             augmented = self.augmentations(image=processed_spectrogram)
#             processed_spectrogram = augmented['image'].float() / 255.0
#         else:      
#             # Ensure the processed spectrogram is in float32
#             processed_spectrogram = self.transform(processed_spectrogram).float()

#         return processed_spectrogram, label

# # Combined dataset for Multimodal model
# class CombinedDataset(Dataset):
#     def __init__(self, metadata, cfg, training_flag=False, augmentations=None, plot=False):
#         self.metadata = metadata
#         self.cfg = cfg
#         self.training_flag = training_flag
#         self.augmentations = augmentations
#         self.plot = plot

#         # Set the random seed for reproducibility
#         self.seed_everything()

#         # Feature to index mapping from CFG
#         self.feature_to_index = self.cfg['feature_to_index']
#         self.differential_channels_start_index = len(self.feature_to_index)

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx):
#         row = self.metadata.iloc[idx]

#         # Process EEG data
#         eeg_data = self.process_eeg(row)
#         eeg_label = self.get_label(row)

#         # Process Spectrogram data
#         spectrogram_data = self.process_spectrogram(row)
#         spectrogram_label = self.get_label(row)

#         # Ensure the labels are the same
#         assert torch.equal(eeg_label, spectrogram_label), "Labels do not match!"

#         return (eeg_data, spectrogram_data), eeg_label

#     def process_eeg(self, row):
#         eeg_id = row['eeg_id']
#         eeg = load_train_eeg_frame(eeg_id)
#         waves = eeg.values.T

#         if self.cfg['AUGMENT']:
#             waves = mirror_eeg(waves, self.cfg['LL'], self.cfg['LP'], self.cfg['RL'], self.cfg['RP'], self.feature_to_index)

#         waves = butter_bandpass_filter(waves, self.cfg['bandpass_filter']['low'], self.cfg['bandpass_filter']['high'], self.cfg['sampling_rate'])
#         waves = handle_nan(waves)
#         waves = calculate_differential_signals(waves, self.cfg['map_features'], self.feature_to_index)
#         waves = denoise_filter(waves, self.cfg['bandpass_filter']['low'], self.cfg['bandpass_filter']['high'], self.cfg['sampling_rate'])
#         waves = normalize_signal(waves)
#         waves = select_and_map_channels(waves, self.cfg['eeg_features'], self.feature_to_index, self.differential_channels_start_index, self.cfg['map_features'])
#         waves = pad_or_truncate(waves, self.cfg['fixed_length'])
#         waves = waves[np.newaxis, ...]  # Add channel dimension for EEG data
#         return torch.tensor(waves, dtype=torch.float32)

#     def process_spectrogram(self, row):
#         spec_id = row['spectrogram_id']
#         raw_spectrogram = load_train_spectr_frame(spec_id)

#         if isinstance(raw_spectrogram, pd.DataFrame):
#             raw_spectrogram = raw_spectrogram.to_numpy()
        
#         offset = row.get("spectrogram_label_offset_seconds", None)
#         if offset is not None:
#             offset = offset // 2
#             basic_spectrogram = raw_spectrogram[:, offset:offset + 300]
#             pad_size = max(0, 300 - basic_spectrogram.shape[1])
#             basic_spectrogram = np.pad(basic_spectrogram, ((0, 0), (0, pad_size)), mode='constant')
#         else:
#             basic_spectrogram = raw_spectrogram

#         spectrogram = basic_spectrogram.T
#         spectrogram = pad_or_truncate(spectrogram, self.cfg['image_size'])
#         spectrogram = handle_nan(spectrogram)
#         spectrogram = baseline_correction(spectrogram)
#         spectrogram = apply_notch_filter(spectrogram)
#         spectrogram = smooth_spectrogram(spectrogram)
#         spectrogram = normalize_signal(spectrogram)
#         spectrogram = resample_spectrogram(spectrogram, self.cfg['image_size'])
#         spectrogram = np.tile(spectrogram[..., None], (1, 1, 3))

#         if self.plot:
#             plot_spectrograms(basic_spectrogram, spectrogram, self.metadata.index.tolist(), num_labels=10)

#         if self.augmentations:
#             spectrogram = (spectrogram * 255).astype(np.uint8)
#             augmented = self.augmentations(image=spectrogram)
#             spectrogram = augmented['image']
#             spectrogram = spectrogram.float() / 255.0
#         else:
#             spectrogram = spectrogram.astype(np.float32)
#             spectrogram = torch.tensor(spectrogram).permute(2, 0, 1).float()

#         return spectrogram

#     def get_label(self, row):
#         label_name = row['expert_consensus']
#         label_idx = self.cfg['name2label'][label_name]
#         label = labels_to_probabilities(label_idx, self.cfg['n_classes'])
#         return label.clone().detach().float()

#     def seed_everything(self):
#         np.random.seed(self.cfg['seed'])
#         torch.manual_seed(self.cfg['seed'])
#         random.seed(self.cfg['seed'])
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(self.cfg['seed'])
#             torch.cuda.manual_seed_all(self.cfg['seed'])
#             torch.backends.cudnn.deterministic = True
#             torch.backends.cudnn.benchmark = False
            
            
            
            
