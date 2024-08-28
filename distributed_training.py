import sys
import os
import gc
import copy
import yaml
import random
import shutil
import time
import typing as tp
from glob import glob
from pathlib import Path
from collections import OrderedDict, defaultdict
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from scipy.signal import butter, lfilter, freqz, filtfilt,iirnotch

import torch
from torch import nn
from torch import optim
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.cuda import amp
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, default_collate, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split

import timm
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

import albumentations as A
from albumentations.pytorch import ToTensorV2

from captum.attr import Saliency, IntegratedGradients

from lime import lime_image
from skimage.segmentation import mark_boundaries
import torchvision.transforms as transforms
from torch.nn.functional import softmax
# Generate Segments
from skimage.segmentation import slic
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler



class CFG:
    debug = False
    AUGMENT = False
    VALIDATION_FRAC = 0.4
    AUGMENTATION_FRACTION = 0.05
    EPOCHS = 50

    #checkpoint_dir = "/Users/koushani/Documents/UB-COURSEWORK/SPRING24/XAI_HMS_KAGGLE/kou/github/Brain-Pattern-Identification-using-Multimodal-Classification/checkpoints/attention_model"
    checkpoint_dir = "/eng/home/koushani/Documents/Multimodal_XAI/Brain-Pattern-Identification-using-Multimodal-Classification/checkpoint_dir/"
    last_optimizer = None
    last_regularization_lambda = None
    checkpointing_enabled = True


    SKIP_ASSERT = False


    seed = 42
    gpu_idx = 0

    device = torch.device('cuda')

    EEG_COLUMNS =  [ # to assert columns order is the same
    'Fp1','F3', 'C3', 'P3', 'F7',
    'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4',
    'F8', 'T4', 'T6', 'O2', 'EKG'
    ]

    eeg_features = [column for column in EEG_COLUMNS if column != 'EKG'] # [ 'EKG' ]

    classes = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    label2name = dict(enumerate(classes))
    name2label = {v: k for k, v in label2name.items()}
    n_classes = len(classes)

    # Spectrogram columns
    SPECTR_COLUMNS = [
    'time', 'LL_0.59', 'LL_0.78', 'LL_0.98', 'LL_1.17', 'LL_1.37',
    'LL_1.56', 'LL_1.76', 'LL_1.95', 'LL_2.15', 'LL_2.34', 'LL_2.54',
    'LL_2.73', 'LL_2.93', 'LL_3.13', 'LL_3.32', 'LL_3.52', 'LL_3.71',
    'LL_3.91', 'LL_4.1', 'LL_4.3', 'LL_4.49', 'LL_4.69', 'LL_4.88',
    'LL_5.08', 'LL_5.27', 'LL_5.47', 'LL_5.66', 'LL_5.86', 'LL_6.05',
    'LL_6.25', 'LL_6.45', 'LL_6.64', 'LL_6.84', 'LL_7.03', 'LL_7.23',
    'LL_7.42', 'LL_7.62', 'LL_7.81', 'LL_8.01', 'LL_8.2', 'LL_8.4',
    'LL_8.59', 'LL_8.79', 'LL_8.98', 'LL_9.18', 'LL_9.38', 'LL_9.57',
    'LL_9.77', 'LL_9.96', 'LL_10.16', 'LL_10.35', 'LL_10.55', 'LL_10.74',
    'LL_10.94', 'LL_11.13', 'LL_11.33', 'LL_11.52', 'LL_11.72', 'LL_11.91',
    'LL_12.11', 'LL_12.3', 'LL_12.5', 'LL_12.7', 'LL_12.89', 'LL_13.09',
    'LL_13.28', 'LL_13.48', 'LL_13.67', 'LL_13.87', 'LL_14.06', 'LL_14.26',
    'LL_14.45', 'LL_14.65', 'LL_14.84', 'LL_15.04', 'LL_15.23', 'LL_15.43',
    'LL_15.63', 'LL_15.82', 'LL_16.02', 'LL_16.21', 'LL_16.41', 'LL_16.6',
    'LL_16.8', 'LL_16.99', 'LL_17.19', 'LL_17.38', 'LL_17.58', 'LL_17.77',
    'LL_17.97', 'LL_18.16', 'LL_18.36', 'LL_18.55', 'LL_18.75', 'LL_18.95',
    'LL_19.14', 'LL_19.34', 'LL_19.53', 'LL_19.73', 'LL_19.92', 'RL_0.59',
    'RL_0.78', 'RL_0.98', 'RL_1.17', 'RL_1.37', 'RL_1.56', 'RL_1.76',
    'RL_1.95', 'RL_2.15', 'RL_2.34', 'RL_2.54', 'RL_2.73', 'RL_2.93',
    'RL_3.13', 'RL_3.32', 'RL_3.52', 'RL_3.71', 'RL_3.91', 'RL_4.1',
    'RL_4.3', 'RL_4.49', 'RL_4.69', 'RL_4.88', 'RL_5.08', 'RL_5.27',
    'RL_5.47', 'RL_5.66', 'RL_5.86', 'RL_6.05', 'RL_6.25', 'RL_6.45',
    'RL_6.64', 'RL_6.84', 'RL_7.03', 'RL_7.23', 'RL_7.42', 'RL_7.62',
    'RL_7.81', 'RL_8.01', 'RL_8.2', 'RL_8.4', 'RL_8.59', 'RL_8.79',
    'RL_8.98', 'RL_9.18', 'RL_9.38', 'RL_9.57', 'RL_9.77', 'RL_9.96',
    'RL_10.16', 'RL_10.35', 'RL_10.55', 'RL_10.74', 'RL_10.94', 'RL_11.13',
    'RL_11.33', 'RL_11.52', 'RL_11.72', 'RL_11.91', 'RL_12.11', 'RL_12.3',
    'RL_12.5', 'RL_12.7', 'RL_12.89', 'RL_13.09', 'RL_13.28', 'RL_13.48',
    'RL_13.67', 'RL_13.87', 'RL_14.06', 'RL_14.26', 'RL_14.45', 'RL_14.65',
    'RL_14.84', 'RL_15.04', 'RL_15.23', 'RL_15.43', 'RL_15.63', 'RL_15.82',
    'RL_16.02', 'RL_16.21', 'RL_16.41', 'RL_16.6', 'RL_16.8', 'RL_16.99',
    'RL_17.19', 'RL_17.38', 'RL_17.58', 'RL_17.77', 'RL_17.97', 'RL_18.16',
    'RL_18.36', 'RL_18.55', 'RL_18.75', 'RL_18.95', 'RL_19.14', 'RL_19.34',
    'RL_19.53', 'RL_19.73', 'RL_19.92', 'LP_0.59', 'LP_0.78', 'LP_0.98',
    'LP_1.17', 'LP_1.37', 'LP_1.56', 'LP_1.76', 'LP_1.95', 'LP_2.15',
    'LP_2.34', 'LP_2.54', 'LP_2.73', 'LP_2.93', 'LP_3.13', 'LP_3.32',
    'LP_3.52', 'LP_3.71', 'LP_3.91', 'LP_4.1', 'LP_4.3', 'LP_4.49',
    'LP_4.69', 'LP_4.88', 'LP_5.08', 'LP_5.27', 'LP_5.47', 'LP_5.66',
    'LP_5.86', 'LP_6.05', 'LP_6.25', 'LP_6.45', 'LP_6.64', 'LP_6.84',
    'LP_7.03', 'LP_7.23', 'LP_7.42', 'LP_7.62', 'LP_7.81', 'LP_8.01',
    'LP_8.2', 'LP_8.4', 'LP_8.59', 'LP_8.79', 'LP_8.98', 'LP_9.18',
    'LP_9.38', 'LP_9.57', 'LP_9.77', 'LP_9.96', 'LP_10.16', 'LP_10.35',
    'LP_10.55', 'LP_10.74', 'LP_10.94', 'LP_11.13', 'LP_11.33', 'LP_11.52',
    'LP_11.72', 'LP_11.91', 'LP_12.11', 'LP_12.3', 'LP_12.5', 'LP_12.7',
    'LP_12.89', 'LP_13.09', 'LP_13.28', 'LP_13.48', 'LP_13.67', 'LP_13.87',
    'LP_14.06', 'LP_14.26', 'LP_14.45', 'LP_14.65', 'LP_14.84', 'LP_15.04',
    'LP_15.23', 'LP_15.43', 'LP_15.63', 'LP_15.82', 'LP_16.02', 'LP_16.21',
    'LP_16.41', 'LP_16.6', 'LP_16.8', 'LP_16.99', 'LP_17.19', 'LP_17.38',
    'LP_17.58', 'LP_17.77', 'LP_17.97', 'LP_18.16', 'LP_18.36', 'LP_18.55',
    'LP_18.75', 'LP_18.95', 'LP_19.14', 'LP_19.34', 'LP_19.53', 'LP_19.73',
    'LP_19.92', 'RP_0.59', 'RP_0.78', 'RP_0.98', 'RP_1.17', 'RP_1.37',
    'RP_1.56', 'RP_1.76', 'RP_1.95', 'RP_2.15', 'RP_2.34', 'RP_2.54',
    'RP_2.73', 'RP_2.93', 'RP_3.13', 'RP_3.32', 'RP_3.52', 'RP_3.71',
    'RP_3.91', 'RP_4.1', 'RP_4.3', 'RP_4.49', 'RP_4.69', 'RP_4.88',
    'RP_5.08', 'RP_5.27', 'RP_5.47', 'RP_5.66', 'RP_5.86', 'RP_6.05',
    'RP_6.25', 'RP_6.45', 'RP_6.64', 'RP_6.84', 'RP_7.03', 'RP_7.23',
    'RP_7.42', 'RP_7.62', 'RP_7.81', 'RP_8.01', 'RP_8.2', 'RP_8.4',
    'RP_8.59', 'RP_8.79', 'RP_8.98', 'RP_9.18', 'RP_9.38', 'RP_9.57',
    'RP_9.77', 'RP_9.96', 'RP_10.16', 'RP_10.35', 'RP_10.55', 'RP_10.74',
    'RP_10.94', 'RP_11.13', 'RP_11.33', 'RP_11.52', 'RP_11.72', 'RP_11.91',
    'RP_12.11', 'RP_12.3', 'RP_12.5', 'RP_12.7', 'RP_12.89', 'RP_13.09',
    'RP_13.28', 'RP_13.48', 'RP_13.67', 'RP_13.87', 'RP_14.06', 'RP_14.26',
    'RP_14.45', 'RP_14.65', 'RP_14.84', 'RP_15.04', 'RP_15.23', 'RP_15.43',
    'RP_15.63', 'RP_15.82', 'RP_16.02', 'RP_16.21', 'RP_16.41', 'RP_16.6',
    'RP_16.8', 'RP_16.99', 'RP_17.19', 'RP_17.38', 'RP_17.58', 'RP_17.77',
    'RP_17.97', 'RP_18.16', 'RP_18.36', 'RP_18.55', 'RP_18.75', 'RP_18.95',
    'RP_19.14', 'RP_19.34', 'RP_19.53', 'RP_19.73', 'RP_19.92'
]


    n_folds = 5
    SPECTROGRAM_COLUMNS = [col for col in SPECTR_COLUMNS if col != 'time']

    ## Dataset Preprocessing
    bandpass_filter = {"low": 0.5, "high": 20, "order": 2}
    rand_filter = {"probab": 0.1, "low": 10, "high": 20, "band": 1.0, "order": 2}
    freq_channels = [(0.5, 4.5)]
    filter_order = 2
    random_close_zone = 0.0  # 0.2

    map_features = [
        ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
        ("Fz", "Cz"), ("Cz", "Pz")
    ]

    LL = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1']  # Example left EEG channels
    LP = ['Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']  # Example right EEG channels
    RL = LL  # Assuming RL and RP are mirrors of LL and LP respectively
    RP = LP

    feature_to_index = {x: y for x, y in zip(eeg_features, range(len(eeg_features)))}
    simple_features = ['Fz', 'Cz', 'Pz', 'EKG']  # 'Fz', 'Cz', 'Pz', 'EKG'

    n_map_features = len(map_features)
    in_channels = len(SPECTROGRAM_COLUMNS)

    seq_length = 50  # Second's
    sampling_rate = 200  # Hz
    nsamples = seq_length * sampling_rate
    out_samples = nsamples // 5

    debug_input_size = 4096
    #input_size =
    in_chans = 1

    batch_size = 2048
    num_workers = 120 
    fixed_length = 3000
    image_size = (400, 300)

    def seed_everything(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_device():
        if torch.cuda.is_available():
        # Get the current default CUDA device
            device = torch.device("cuda")
        # Get the name of the device
            device_name = torch.cuda.get_device_name(device)
            print(f"CUDA is available. Using device: {device} ({device_name})")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")
        return device


    def stop_checkpointing():
        CFG.checkpointing_enabled = False
        print("Checkpointing disabled.")

    def start_checkpointing():
        CFG.checkpointing_enabled = True
        print("Checkpointing enabled.")



def load_checkpoint(checkpoint_dir, checkpoint_filename, model, optimizer):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        train_accuracies = checkpoint['train_accuracies']
        valid_accuracies = checkpoint['valid_accuracies']
        lr_scheduler = checkpoint['lr_scheduler']
        regularization_losses = checkpoint['regularization_losses']
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        start_epoch = 0
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        lr_scheduler = []
        regularization_losses = []

    return start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, lr_scheduler, regularization_losses





def save_checkpoint(state, checkpoint_dir, checkpoint_filename):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at '{checkpoint_path}'")


def detect_and_save_checkpoint(state, checkpoint_dir, optimizer, regularization_lambda):
    # Detect changes in optimizer and regularization parameter
    optimizer_changed = CFG.last_optimizer is None or type(optimizer) != CFG.last_optimizer
    regularization_changed = CFG.last_regularization_lambda is None or regularization_lambda != CFG.last_regularization_lambda

    # Initialize the checkpoint filename
    checkpoint_filename = "checkpoint.pth.tar"

    # Modify the checkpoint filename based on the changes detected
    if optimizer_changed and regularization_changed:
        checkpoint_filename = "checkpoint_optimizer_and_regularization.pth.tar"
    elif optimizer_changed:
        checkpoint_filename = "checkpoint_optimizer.pth.tar"
    elif regularization_changed:
        checkpoint_filename = "checkpoint_regularization.pth.tar"

    if optimizer_changed or regularization_changed:
        print(f"Changes detected in {'optimizer' if optimizer_changed else ''} {'and' if optimizer_changed and regularization_changed else ''} {'regularization parameter' if regularization_changed else ''}. Creating a new checkpoint.")
        CFG.last_optimizer = type(optimizer)
        CFG.last_regularization_lambda = regularization_lambda
        save_checkpoint(state, checkpoint_dir, checkpoint_filename)




CFG.device = CFG.get_device()



#ROOT_DIR = '/Users/koushani/Documents/UB-COURSEWORK/SPRING24/XAI_HMS_KAGGLE/CODING/DATA'
ROOT_DIR = '/data2/users/koushani/HMS_data'
TRAIN_EEGS = os.path.join(ROOT_DIR , 'train_eegs')
TRAIN_SPECTR =  os.path.join(ROOT_DIR, 'train_spectrograms')
TEST_EEGS = os.path.join(ROOT_DIR, 'test_eegs')
TEST_SPECTR = os.path.join(ROOT_DIR, 'test_spectrograms')

def is_entirely_nan(eeg_id):
    eeg_data = load_train_eeg_frame(eeg_id)
    return np.isnan(eeg_data.values).all()

def load_train_eeg_frame(id):
    # Ensure the ID is an integer to avoid issues with file name construction
    id = int(id)
    # Construct the file path using the integer ID
    file_path = os.path.join(TRAIN_EEGS, f'{id}.parquet')
    # Load the EEG data from the specified Parquet file
    data = pd.read_parquet(file_path, engine='pyarrow')
    # Optional: Verify that the columns match expected EEG columns
    if not CFG.SKIP_ASSERT:
        assert list(data.columns) == CFG.EEG_COLUMNS, 'EEG columns order is not the same!'
    return data

class HMS_EEG_Dataset(Dataset):
    def __init__(self, train_ids, cfg=CFG, training_flag=False, shuffle=False):
        super(HMS_EEG_Dataset, self).__init__()
        self.train_ids = train_ids
        self.cfg = cfg
        self.training_flag = training_flag
        self.shuffle = shuffle

        # Set the random seed for reproducibility
        self.cfg.seed_everything(CFG)
        # Feature to index mapping from CFG
        self.feature_to_index = self.cfg.feature_to_index
        self.differential_channels_start_index = len(self.cfg.feature_to_index)

    def __getitem__(self, idx):
        row = self.train_ids.iloc[idx]
        data = self.single_map_func(row, self.training_flag)
        label_name = row['expert_consensus']  # Assuming 'expert_consensus' column contains the label names
        label_idx = self.cfg.name2label[label_name]
        label = self.labels_to_probabilities(label_idx, self.cfg.n_classes)
        data = data[np.newaxis, ...]  # Add channel dimension for EEG data, now shape is [1, 37, 3000]
        return data.astype(np.float32), label

    def __len__(self):
        return len(self.train_ids)

    def single_map_func(self, row, is_training):
        data = self.get_eeg(row, is_training)
        data = self.handle_nan(data)
        if data.size == 0:
            # Skip this sample if data is empty
            return np.zeros((self.cfg.in_channels + len(self.cfg.map_features), self.cfg.fixed_length)), np.zeros(self.cfg.n_classes)
        data = self.calculate_differential_signals(data)
        data = self.denoise_filter(data)
        data = self.normalize(data)
        data = self.select_and_map_channels(data, self.cfg.eeg_features, self.feature_to_index)
        data = self.pad_or_truncate(data, self.cfg.fixed_length)
        return data

    def get_eeg(self, row, is_training, flip=False):
        eeg_id = row['eeg_id']
        eeg = load_train_eeg_frame(eeg_id)

        waves = eeg.values.T

        if CFG.AUGMENT:
            waves = self.mirror_eeg(waves)

        waves = self.butter_bandpass_filter(waves, self.cfg.bandpass_filter['low'], self.cfg.bandpass_filter['high'], self.cfg.sampling_rate)

        return waves

    def handle_nan(self, data):
        """Handle NaN values by replacing them with the mean of the respective channels."""
        # Remove rows that are entirely NaN
        data = data[~np.isnan(data).all(axis=1)]

        # Check if data is empty after removing NaN rows
        if data.size == 0:
            # Handle the case where all data is NaN, e.g., by filling with zeros or skipping this sample
            data = np.zeros((self.cfg.in_channels + len(self.cfg.map_features), self.cfg.fixed_length))
        else:
            where_nan = np.isnan(data)
            mean_values = np.nanmean(data, axis=1, keepdims=True)

            # Check for channels where the mean could not be computed and replace NaN means with zeros
            mean_values[np.isnan(mean_values)] = 0
            data[where_nan] = np.take(mean_values, np.where(where_nan)[0])

        return data

    def pad_or_truncate(self, data, length):
        if data.shape[1] < length:
            # Pad with zeros if shorter
            padding = np.zeros((data.shape[0], length - data.shape[1]))
            data = np.hstack((data, padding))
        else:
            # Truncate if longer
            data = data[:, :length]
        return data

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        return butter(order, [low, high], btype='band')

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        return lfilter(b, a, data, axis=1)  # Apply along the sample axis

    def calculate_differential_signals(self, data):
        num_pairs = len(self.cfg.map_features)
        differential_data = np.zeros((num_pairs, data.shape[1]))
        for i, (feat_a, feat_b) in enumerate(self.cfg.map_features):
            if feat_a in self.feature_to_index and feat_b in self.feature_to_index:
                differential_data[i, :] = data[self.feature_to_index[feat_a], :] - data[self.feature_to_index[feat_b], :]
            else:
                print(f"Feature {feat_a} or {feat_b} not found in feature_to_index")
        return np.vstack((data, differential_data))

    def denoise_filter(self, x):
        y = self.butter_bandpass_filter(x, self.cfg.bandpass_filter['low'], self.cfg.bandpass_filter['high'], self.cfg.sampling_rate, order=6)
        y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
        y = y[:, 0:-1:4]
        return y

    def normalize(self, data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / (std + 1e-6)  # Adding epsilon to avoid division by zero

    def select_and_map_channels(self, data, channels, feature_to_index):
        selected_indices = [feature_to_index[ch] for ch in channels if ch in feature_to_index]
        differential_indices = list(range(self.differential_channels_start_index, self.differential_channels_start_index + len(self.cfg.map_features)))
        selected_data = data[selected_indices + differential_indices, :]
        return selected_data

    def mirror_eeg(self, data):
        indx1 = [self.feature_to_index[x] for x in self.cfg.LL + self.cfg.LP if x in self.feature_to_index]
        indx2 = [self.feature_to_index[x] for x in self.cfg.RL + self.cfg.RP if x in self.feature_to_index]
        data[indx1, :], data[indx2, :] = data[indx2, :], data[indx1, :]
        return data

    def labels_to_probabilities(self, labels, num_classes):
        labels = torch.eye(num_classes)[labels]
        return labels
    
    
    def get_augmentations(reference_data):
        return A.Compose([
        A.MixUp(reference_data=reference_data, read_fn=lambda x: x, p=0.5),
        A.CoarseDropout(max_holes=1, min_height=1.0, max_height=1.0,
                        min_width=0.06, max_width=0.1, p=0.5),  # freq-masking
        A.CoarseDropout(max_holes=1, min_height=0.06, max_height=0.1,
                        min_width=1.0, max_width=1.0, p=0.5),  # time-masking
        ToTensorV2()
    ])


def create_k_fold_splits(metadata, n_splits=5):
    # Drop unnecessary columns
    metadata.drop(columns=[
            'eeg_sub_id',
            'spectrogram_sub_id',
            'patient_id',
            'label_id'
        ], inplace=True)

    # Ensure correct data types
    metadata['eeg_id'] = metadata['eeg_id'].astype(int)
    metadata['spectrogram_id'] = metadata['spectrogram_id'].astype(int)
    metadata['eeg_label_offset_seconds'] = metadata['eeg_label_offset_seconds'].astype(int)
    metadata['spectrogram_label_offset_seconds'] = metadata['spectrogram_label_offset_seconds'].astype(int)

    # Debugging: Sample the data if in DEBUG mode to reduce size for faster processing
    if CFG.debug:
        metadata = metadata.sample(min(CFG.debug_input_size, len(metadata)))

    # Extract features and labels for stratification
    X = metadata['eeg_id']
    y = metadata['expert_consensus']  # Correct column name for class labels

    # Create stratified K-Folds
    skf = StratifiedKFold(n_splits=n_splits)
    fold_indices = []

    for train_index, valid_index in skf.split(X, y):
        train_ids = X.iloc[train_index].tolist()
        valid_ids = X.iloc[valid_index].tolist()
        fold_indices.append((train_ids, valid_ids))

    return fold_indices

def createTrainTestSplit(metadata, fold_indices, fold_idx):
    train_ids, valid_ids = fold_indices[fold_idx]

    train_metadata = metadata[metadata['eeg_id'].isin(train_ids)]
    valid_metadata = metadata[metadata['eeg_id'].isin(valid_ids)]

    if CFG.AUGMENT:
        # Select 10% of train_metadata for augmentation
        augmentation_size = int(len(train_metadata) * CFG.AUGMENTATION_FRACTION)
        augmentation_metadata = train_metadata.sample(augmentation_size)
        return train_metadata, valid_metadata, augmentation_metadata

    return train_metadata, valid_metadata


main_metadata = pd.read_csv('/data2/users/koushani/HMS_data/train.csv')

fold_indices = create_k_fold_splits(main_metadata, n_splits=5)
for fold_idx in range(len(fold_indices)):
    train_metadata, valid_metadata = createTrainTestSplit(main_metadata, fold_indices, fold_idx)

# Function to create DataLoader with DistributedSampler
def create_dataloader(dataset, batch_size, num_workers, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    return loader

# Main function for distributed training
def main():
    # Get rank and world size from the environment (these will be set by torch.distributed.launch)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Set the GPU for this process
    torch.cuda.set_device(rank)
    
    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Create Dataset and DataLoader
    eeg_train_dataset = HMS_EEG_Dataset(train_metadata, cfg=CFG)
    print(f"Size of the train dataset: {len(eeg_train_dataset)}")
    train_loader = create_dataloader(eeg_train_dataset, batch_size=CFG.batch_size, num_workers=120, rank=rank, world_size=world_size)
    
    # Observe the DataLoader output (for testing)
    for batch in train_loader:
        data, labels = batch
        print(f"Rank {rank} - Batch data shape: {data.shape}")
        print(f"Rank {rank} - Batch labels shape: {labels.shape}")
        break
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    main()