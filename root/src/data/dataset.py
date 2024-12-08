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
from utils.config_loader import load_config

cfg = load_config()


# HMS_EEG_Dataset class
class HMS_EEG_Dataset(Dataset):
    def __init__(self, train_ids, cfg=cfg, training_flag=False, shuffle=False):
        super(HMS_EEG_Dataset, self).__init__()
        self.train_ids = train_ids
        self.cfg = cfg
        self.training_flag = training_flag
        self.shuffle = shuffle

        # Set the random seed for reproducibility
        seed_everything()
        # Feature to index mapping from CFG
        self.feature_to_index = feature_to_index
        self.differential_channels_start_index = len(feature_to_index)

    def __getitem__(self, idx):
        row = self.train_ids.iloc[idx]
        # Separate the label from the data
        label_name = row['expert_consensus']  # Assuming 'expert_consensus' column contains the label names
        
        # Drop the label column from the row to avoid any further use
        row_new = row.drop('expert_consensus')
        
        label_idx = self.cfg['name2label'][label_name]
        label = labels_to_probabilities(label_idx, self.cfg['n_classes'])      

        # Process the data independently
        data = self.single_map_func(row_new, self.training_flag, idx)


        # Adding channel dimension for EEG data
        data = data[np.newaxis, ...]  # Now shape is [1, 37, 3000] or similar

        # Return the processed data and the separated label
        return data.astype(np.float32), label

    def __len__(self):
        return len(self.train_ids)

    def single_map_func(self, row, is_training, idx):
        data = self.get_eeg(row, is_training)
        
        # Check if data is a tuple (which should no longer be the case)
        if isinstance(data, tuple):
            print(f"Tuple detected in single_map_function. Type: {type(data)}")
            for i, elem in enumerate(data):
                if isinstance(elem, np.ndarray):
                    print(f"Element {i} shape: {elem.shape}")
                else:
                    print(f"Element {i} is not a NumPy array, but a {type(elem)}")  
        
        
        
        data = handle_nan(data)
        if data.size == 0:
            # Skip this sample if data is empty
            return np.zeros((self.cfg['in_channels'] + len(self.cfg['map_features']), self.cfg['fixed_length']))
        data = calculate_differential_signals(data)
        data = normalize(data) 
        data = select_and_map_channels(data, self.cfg['eeg_features'], self.differential_channels_start_index)  
        data = pad_or_truncate(data, self.cfg['fixed_length'])

        
        return data

    def get_eeg(self, row, is_training, flip=False):
        eeg_id = row['eeg_id']
        eeg = load_train_eeg_frame(eeg_id)

        waves = eeg.values.T

        if self.cfg['AUGMENT']:
            waves = mirror_eeg(waves)

        waves = butter_bandpass_filter(waves, self.cfg)
        
        # Check if waves is a tuple
        if isinstance(waves, tuple):
            print(f"Tuple detected in get_eeg for EEG ID {eeg_id}. Type: {type(waves)}")

        return waves


# HMS_Spectrogram_Dataset class
class HMS_Spectrogram_Dataset(Dataset):
    def __init__(self, train_ids, cfg, augmentations=None, plot=False):
        super(HMS_Spectrogram_Dataset, self).__init__()
        self.train_ids = train_ids
        self.cfg = cfg
        self.augmentations = augmentations
        self.plot = plot  # Flag to control plotting

        # Define a transform to resize the spectrogram to 224x224
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),  # Resize to 224x224 for ViT compatibility
            T.Normalize(mean=[0.5], std=[0.5])  # Normalize the spectrogram
        ])

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, idx):
        row = self.train_ids.iloc[idx]
        spec_id = row['spectrogram_id']
        raw_spectrogram = load_train_spectr_frame(spec_id)
        label_name = row['expert_consensus']
        label_idx = self.cfg['name2label'][label_name]
        label = labels_to_probabilities(label_idx, self.cfg['n_classes'])
        offset = row.get("spectrogram_label_offset_seconds", None)

        if isinstance(raw_spectrogram, pd.DataFrame):
            raw_spectrogram = raw_spectrogram.to_numpy()

        if offset is not None:
            offset = offset // 2
            basic_spectrogram = raw_spectrogram[:, offset:offset + 300]
            pad_size = max(0, 300 - basic_spectrogram.shape[1])
            basic_spectrogram = np.pad(basic_spectrogram, ((0, 0), (0, pad_size)), mode='constant')
        else:
            basic_spectrogram = raw_spectrogram

        spectrogram = basic_spectrogram.T
        processed_spectrogram = pad_or_truncate(spectrogram, self.cfg['image_size'])
        processed_spectrogram = handle_nan(processed_spectrogram)
        processed_spectrogram = baseline_correction(processed_spectrogram)
        processed_spectrogram = apply_notch_filter(processed_spectrogram)
        processed_spectrogram = smooth_spectrogram(processed_spectrogram)
        processed_spectrogram = normalize_signal(processed_spectrogram)
        processed_spectrogram = resample_spectrogram(processed_spectrogram, self.cfg['image_size'])

        # Expand to 3 channels for compatibility with ViT
        processed_spectrogram = np.tile(processed_spectrogram[..., None], (1, 1, 3))

        if self.plot:
            plot_spectrograms(basic_spectrogram, processed_spectrogram, self.train_ids.index.tolist(), num_labels=10)

        # Apply transformations and augmentations
        if self.augmentations:
            processed_spectrogram = (processed_spectrogram * 255).astype(np.uint8)
            augmented = self.augmentations(image=processed_spectrogram)
            processed_spectrogram = augmented['image'].float() / 255.0
        else:
            processed_spectrogram = self.transform(processed_spectrogram)

        return processed_spectrogram, label

# Combined dataset for Multimodal model
class CombinedDataset(Dataset):
    def __init__(self, metadata, cfg, training_flag=False, augmentations=None, plot=False):
        self.metadata = metadata
        self.cfg = cfg
        self.training_flag = training_flag
        self.augmentations = augmentations
        self.plot = plot

        # Set the random seed for reproducibility
        self.seed_everything()

        # Feature to index mapping from CFG
        self.feature_to_index = self.cfg['feature_to_index']
        self.differential_channels_start_index = len(self.feature_to_index)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Process EEG data
        eeg_data = self.process_eeg(row)
        eeg_label = self.get_label(row)

        # Process Spectrogram data
        spectrogram_data = self.process_spectrogram(row)
        spectrogram_label = self.get_label(row)

        # Ensure the labels are the same
        assert torch.equal(eeg_label, spectrogram_label), "Labels do not match!"

        return (eeg_data, spectrogram_data), eeg_label

    def process_eeg(self, row):
        eeg_id = row['eeg_id']
        eeg = load_train_eeg_frame(eeg_id)
        waves = eeg.values.T

        if self.cfg['AUGMENT']:
            waves = mirror_eeg(waves, self.cfg['LL'], self.cfg['LP'], self.cfg['RL'], self.cfg['RP'], self.feature_to_index)

        waves = butter_bandpass_filter(waves, self.cfg['bandpass_filter']['low'], self.cfg['bandpass_filter']['high'], self.cfg['sampling_rate'])
        waves = handle_nan(waves)
        waves = calculate_differential_signals(waves, self.cfg['map_features'], self.feature_to_index)
        waves = denoise_filter(waves, self.cfg['bandpass_filter']['low'], self.cfg['bandpass_filter']['high'], self.cfg['sampling_rate'])
        waves = normalize_signal(waves)
        waves = select_and_map_channels(waves, self.cfg['eeg_features'], self.feature_to_index, self.differential_channels_start_index, self.cfg['map_features'])
        waves = pad_or_truncate(waves, self.cfg['fixed_length'])
        waves = waves[np.newaxis, ...]  # Add channel dimension for EEG data
        return torch.tensor(waves, dtype=torch.float32)

    def process_spectrogram(self, row):
        spec_id = row['spectrogram_id']
        raw_spectrogram = load_train_spectr_frame(spec_id)

        if isinstance(raw_spectrogram, pd.DataFrame):
            raw_spectrogram = raw_spectrogram.to_numpy()
        
        offset = row.get("spectrogram_label_offset_seconds", None)
        if offset is not None:
            offset = offset // 2
            basic_spectrogram = raw_spectrogram[:, offset:offset + 300]
            pad_size = max(0, 300 - basic_spectrogram.shape[1])
            basic_spectrogram = np.pad(basic_spectrogram, ((0, 0), (0, pad_size)), mode='constant')
        else:
            basic_spectrogram = raw_spectrogram

        spectrogram = basic_spectrogram.T
        spectrogram = pad_or_truncate(spectrogram, self.cfg['image_size'])
        spectrogram = handle_nan(spectrogram)
        spectrogram = baseline_correction(spectrogram)
        spectrogram = apply_notch_filter(spectrogram)
        spectrogram = smooth_spectrogram(spectrogram)
        spectrogram = normalize_signal(spectrogram)
        spectrogram = resample_spectrogram(spectrogram, self.cfg['image_size'])
        spectrogram = np.tile(spectrogram[..., None], (1, 1, 3))

        if self.plot:
            plot_spectrograms(basic_spectrogram, spectrogram, self.metadata.index.tolist(), num_labels=10)

        if self.augmentations:
            spectrogram = (spectrogram * 255).astype(np.uint8)
            augmented = self.augmentations(image=spectrogram)
            spectrogram = augmented['image']
            spectrogram = spectrogram.float() / 255.0
        else:
            spectrogram = spectrogram.astype(np.float32)
            spectrogram = torch.tensor(spectrogram).permute(2, 0, 1).float()

        return spectrogram

    def get_label(self, row):
        label_name = row['expert_consensus']
        label_idx = self.cfg['name2label'][label_name]
        label = labels_to_probabilities(label_idx, self.cfg['n_classes'])
        return label.clone().detach().float()

    def seed_everything(self):
        np.random.seed(self.cfg['seed'])
        torch.manual_seed(self.cfg['seed'])
        random.seed(self.cfg['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.cfg['seed'])
            torch.cuda.manual_seed_all(self.cfg['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False