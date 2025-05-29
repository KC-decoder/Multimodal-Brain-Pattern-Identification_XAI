# Import necessary libraries
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import os
import yaml
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import numpy as np
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import StratifiedKFold
from skimage.transform import resize
import torch
import random



# Load the config file using the config_loader module
from utils.config_loader import load_config
import subprocess
import sys


# Function to install libraries listed in requirements.txt
def install_packages():
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All required libraries are installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install required libraries: {e}")
        sys.exit(1)


CFG = load_config()
feature_to_index = {x: y for x, y in zip(CFG['eeg_features'], range(len(CFG['eeg_features'])))}

# Access the paths directly from the configuration
TRAIN_EEGS = CFG['train_eegs']
TRAIN_SPECTR = CFG['train_spectr']
TEST_EEGS = CFG['test_eegs']
TEST_SPECTR = CFG['test_spectr']

# Function to check if the EEG data is entirely NaN
def is_entirely_nan(eeg_id):
    eeg_data = load_train_eeg_frame(eeg_id)
    return np.isnan(eeg_data.values).all()

def mirror_eeg(data):
    # Extract the relevant lists from the config
    LL = CFG['RL']  # The 'RL' key is actually referencing the 'LL' list
    LP = CFG['LP']
    RL = CFG['RP']  # The 'RP' key is referencing the 'LP' list
    RP = CFG['RL']  # The 'RL' key is also referencing the 'LL' list

    # Assuming feature_to_index is a dictionary that maps feature names to indices
    indx1 = [feature_to_index[x] for x in LL + LP if x in feature_to_index]
    indx2 = [feature_to_index[x] for x in RL + RP if x in feature_to_index]

    # Swap the data using the indices
    data[indx1, :], data[indx2, :] = data[indx2, :], data[indx1, :]
    
    return data


def load_train_eeg_frame(id):
    # Ensure the ID is an integer to avoid issues with file name construction
    id = int(id)
    # Construct the file path using the integer ID
    file_path = os.path.join(TRAIN_EEGS, f'{id}.parquet')
    # Load the EEG data from the specified Parquet file
    data = pd.read_parquet(file_path, engine='pyarrow')
    # Optional: Verify that the columns match expected EEG columns
    if not CFG['SKIP_ASSERT']:
        assert list(data.columns) == CFG['EEG_COLUMNS'], 'EEG columns order is not the same!'
    return data


def load_train_spectr_frame(id):
    # Ensure the ID is an integer to prevent file path errors
    id = int(id)
    # Construct the file path using the integer ID
    file_path = os.path.join(TRAIN_SPECTR, f'{id}.parquet')
    # Load the spectrogram data from the specified Parquet file
    data = pd.read_parquet(file_path, engine='pyarrow')
    # Optional: Verify that the columns match expected Spectrogram columns
    if not CFG['SKIP_ASSERT']:
        assert list(data.columns) == CFG['SPECTR_COLUMNS'], 'Spectrogram columns order is not the same!'
    return data


# Plot raw and processed spectrograms
def plot_spectrograms(raw, processed, labels, num_labels):
    """Plot raw and processed spectrograms."""
    x_ticks = np.linspace(0, processed.shape[1] - 1, num_labels).astype(int)
    x_labels = [labels[i] for i in x_ticks]

    plt.figure(figsize=(40, 16))

    plt.subplot(1, 2, 1)
    plt.title("Raw Signal")
    plt.imshow(raw, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=90)
    plt.gca().xaxis.set_tick_params(labelsize=10)
    plt.gcf().subplots_adjust(bottom=0.3)

    plt.subplot(1, 2, 2)
    plt.title("Processed Signal")
    if processed.ndim == 3 and processed.shape[2] > 1:
        plt.imshow(processed[:, :, 0], aspect='auto', cmap='viridis')
    else:
        plt.imshow(processed.squeeze(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=90)
    plt.gca().xaxis.set_tick_params(labelsize=10)
    plt.gcf().subplots_adjust(bottom=0.3)

    plt.tight_layout()
    plt.show()


def baseline_correction(sig):
    sig -= np.mean(sig, axis=0)
    return sig

def normalize_signal(sig):
    """Normalize the signal by scaling it to the range [0, 1], handling NaN values."""
    sig = np.nan_to_num(sig, nan=np.nanmean(sig))
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)

def apply_notch_filter(sig, freq=60, fs=200, quality=30):
    b, a = iirnotch(freq, quality, fs)
    sig = filtfilt(b, a, sig, axis=0)
    return sig

def smooth_spectrogram(sig, sigma=1.0):
    sig = gaussian_filter(sig, sigma=sigma)
    return sig

def resample_spectrogram(sig, target_shape):
    sig = resize(sig, target_shape, mode='reflect', anti_aliasing=True)
    return sig

def handle_nan(data):
    data = data[~np.isnan(data).all(axis=1)]
    if data.size == 0:
        data = np.zeros_like(data)
    else:
        where_nan = np.isnan(data)
        mean_values = np.nanmean(data, axis=1, keepdims=True)
        mean_values[np.isnan(mean_values)] = 0
        data[where_nan] = np.take(mean_values, np.where(where_nan)[0])
    return data

def pad_or_truncate(data, length):
    if isinstance(length, int):
        if data.shape[1] < length:
            padding = np.zeros((data.shape[0], length - data.shape[1]))
            data = np.hstack((data, padding))
        else:
            data = data[:, :length]
    elif isinstance(length, tuple):
        target_rows, target_cols = length
        if data.shape[0] < target_rows:
            row_padding = np.zeros((target_rows - data.shape[0], data.shape[1]))
            data = np.vstack((data, row_padding))
        else:
            data = data[:target_rows, :]
        if data.shape[1] < target_cols:
            col_padding = np.zeros((data.shape[0], target_cols - data.shape[1]))
            data = np.hstack((data, col_padding))
        else:
            data = data[:, :target_cols]
    return data

def calculate_differential_signals(data):
    """
    Calculate differential signals based on pairs of features and concatenate them to the original data.
    
    Parameters:
    - data: numpy.ndarray, shape (n_channels, n_samples)
        The EEG data to process.
    - map_features: list of tuples
        Each tuple contains a pair of feature names (feat_a, feat_b) where the differential signal is calculated as data[feat_a] - data[feat_b].
    - feature_to_index: dict
        A dictionary mapping feature names to their respective indices in `data`.
    
    Returns:
    - combined_data: numpy.ndarray
        The original data concatenated with the calculated differential signals.
    """
    num_pairs = len(CFG['map_features'])
    differential_data = np.zeros((num_pairs, data.shape[1]))
    
    for i, (feat_a, feat_b) in enumerate(CFG['map_features']):
        if feat_a in feature_to_index and feat_b in feature_to_index:
            differential_data[i, :] = data[feature_to_index[feat_a], :] - data[feature_to_index[feat_b], :]
        else:
            print(f"Skipping: Feature {feat_a} or {feat_b} not found in feature_to_index")
            differential_data[i, :] = np.zeros(data.shape[1])

    # Debugging: Print the shapes before concatenation
    #print(f"Shape of data: {data.shape}")
    #print(f"Shape of differential_data: {differential_data.shape}")

    # Ensure that the shapes match before concatenation 
    combined_data = np.vstack((data, differential_data))
    return combined_data
    

def butter_bandpass(CFG):
    nyquist = 0.5 * CFG['sampling_rate']
    low = CFG['bandpass_filter']['low'] / nyquist
    high = CFG['bandpass_filter']['high'] / nyquist
    return butter(CFG['bandpass_filter']['order'], [low, high], btype='band')

def butter_bandpass_filter(data, CFG):
    b, a = butter_bandpass(CFG)
    return lfilter(b, a, data, axis=1)



def normalize(data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / (std + 1e-6)  # Adding epsilon to avoid division by zero


def denoise_filter(x, CFG):
    y = butter_bandpass_filter(x, CFG)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[:, 0:-1:4]
    return y



# data = select_and_map_channels(data, self.cfg['eeg_features'], self.feature_to_index, self.differential_channels_start_index)
def select_and_map_channels(data, channels, differential_channels_start_index):
    selected_indices = [feature_to_index[ch] for ch in channels if ch in feature_to_index]
    differential_indices = list(range(differential_channels_start_index, differential_channels_start_index + len(CFG['map_features'])))
    selected_data = data[selected_indices + differential_indices, :]
    return selected_data

def labels_to_probabilities(labels, num_classes):
    labels = torch.eye(num_classes)[labels]
    return labels


def load_checkpoint(checkpoint_dir, checkpoint_filename, model, optimizer, new_checkpoint=False):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    if not new_checkpoint and os.path.isfile(checkpoint_path):
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
        print(f"Loaded checkpoint from epoch {start_epoch}")
    else:
        if new_checkpoint:
            print(f"Creating a new checkpoint at '{checkpoint_path}'")
        else:
            print(f"No checkpoint found at '{checkpoint_path}', starting fresh.")
        start_epoch = 0
        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        lr_scheduler = []

    return start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, lr_scheduler

def save_checkpoint(state, checkpoint_dir, checkpoint_filename):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at '{checkpoint_path}'")


def detect_and_save_checkpoint(state, checkpoint_dir, optimizer, regularization_lambda):
    # Detect changes in optimizer and regularization parameter
    optimizer_changed = CFG['last_optimizer'] is None or type(optimizer) != CFG['last_optimizer']
    regularization_changed = CFG.last_regularization_lambda is None or regularization_lambda != CFG['last_regularization_lambda']

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
        CFG['last_optimizer'] = type(optimizer)
        CFG['last_regularization_lambda'] = regularization_lambda
        save_checkpoint(state, checkpoint_dir, checkpoint_filename)
        
        
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
    if CFG['debug']:
        metadata = metadata.sample(min(CFG['debug_input_size'], len(metadata)))

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

    return train_metadata, valid_metadata


def linear_warmup_and_cosine_annealing(epoch, warmup_epochs, total_epochs, initial_lr=0.00001, target_lr=0.001):
    if epoch < warmup_epochs:
        # Linearly increase from initial_lr to target_lr during warm-up
        return initial_lr + (target_lr - initial_lr) * (epoch + 1) / warmup_epochs
    else:
        # After warm-up, apply cosine annealing
        return target_lr * 0.5 * (1 + torch.cos(torch.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))


def seed_everything():
    np.random.seed(CFG['seed'])
    torch.manual_seed(CFG['seed'])
    random.seed(CFG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CFG['seed'])
        torch.cuda.manual_seed_all(CFG['seed'])
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
    CFG['checkpointing_enabled'] = False
    print("Checkpointing disabled.")

def start_checkpointing():
    CFG['checkpointing_enabled'] = True
    print("Checkpointing enabled.")


def calculate_metrics(model, dataloader, device):
    """
    Calculate precision, recall, and F1 scores for a given model and dataloader.
    
    Args:
    - model (torch.nn.Module): Trained model.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the validation/test data.
    - device (torch.device): Device to perform calculations on (CPU/GPU).
    
    Returns:
    - precision (float): Precision score.
    - recall (float): Recall score.
    - f1 (float): F1 score.
    """
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            
            all_labels.extend(labels_max.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return precision, recall, f1

def plot_metrics(train_metrics, valid_metrics, metric_name, save_dir=CFG['save_dir']):
    """
    Plot and save the training and validation metrics over epochs.
    
    Args:
    - train_metrics (list): List of training metrics (precision, recall, or F1).
    - valid_metrics (list): List of validation metrics (precision, recall, or F1).
    - metric_name (str): Name of the metric being plotted.
    - save_dir (str): Directory to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(valid_metrics, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot with the specified metric name
    plot_path = os.path.join(save_dir, f'EEG_MODEL_{metric_name}.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"{metric_name} plot saved at {plot_path}")


def plot_learning_rate_and_regularization(lr_scheduler, regularization_losses, save_dir=CFG['save_dir']):
    """
    Plot and save the learning rate schedule and regularization loss over epochs.

    Args:
    - lr_scheduler (list): List of learning rates over epochs.
    - regularization_losses (list): List of regularization losses over epochs.
    - save_dir (str): Directory to save the generated plot.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(lr_scheduler)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.subplot(1, 2, 2)
    plt.plot(regularization_losses)
    plt.title('Regularization Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Regularization Loss')

    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot with the specified name
    plot_path = os.path.join(save_dir, 'EEG_MODEL_LearningRate_and_Regularization.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Learning Rate and Regularization Loss plot saved at {plot_path}")

def plot_accuracies(train_accuracies, valid_accuracies, save_dir=CFG['save_dir']):
    """
    Plot and save the training and validation accuracies over epochs.

    Args:
    - train_accuracies (list): List of training accuracies over epochs.
    - valid_accuracies (list): List of validation accuracies over epochs.
    - save_dir (str): Directory to save the generated plot.
    """
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(10, 5))
    
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, valid_accuracies, 'r-', label='Validation Accuracy')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot with the specified name
    plot_path = os.path.join(save_dir, 'EEG_MODEL_Accuracy.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Training and Validation Accuracy plot saved at {plot_path}")
    
    
    
def plot_confusion_matrix(y_true, y_pred, classes, save_dir=CFG['save_dir'], normalize=False):
    """
    Plot and save the confusion matrix.

    Args:
    - y_true (list): True labels.
    - y_pred (list): Predicted labels.
    - classes (list): List of class names.
    - save_dir (str): Directory to save the generated plot.
    - normalize (bool): If True, normalize the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot with the specified name
    plot_path = os.path.join(save_dir, 'Confusion_Matrix.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Confusion matrix plot saved at {plot_path}")

def create_confusion_matrix(model, dataloader, classes, device=CFG['device'], checkpoint_dir=CFG['checkpoint_dir'], checkpoint_filename='eeg_checkpoint.pth.tar', save_dir=CFG['save_dir']):
    """
    Load checkpoint, make predictions on the validation set, and plot the confusion matrix.

    Args:
    - model (torch.nn.Module): The model architecture.
    - dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
    - classes (list): List of class names.
    - device (str): Device to perform computation on ('cuda' or 'cpu').
    - checkpoint_dir (str): Directory containing the checkpoint.
    - checkpoint_filename (str): Filename of the checkpoint.
    - save_dir (str): Directory to save the generated plot.
    """
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer to load state dict
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, lr_scheduler, regularization_losses = load_checkpoint(checkpoint_dir, checkpoint_filename, model, optimizer)

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # Get the index of the max log-probability
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    plot_confusion_matrix(y_true, y_pred, classes, save_dir=save_dir)
    
def load_checkpoint_for_analysis(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        valid_accuracies = checkpoint['valid_accuracies']  # Extract validation accuracies
        return valid_accuracies
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return None

def analyze_checkpoints(checkpoint_dir):

    best_valid_acc = 0
    best_gamma = None
    best_step_size = None

    # Loop through all model files in the directory
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("model_"):
            # Extract gamma and step_size from filename (assumes filenames contain gamma and decay)
            parts = filename.split("_")
            gamma = float(parts[2])
            step_size = int(parts[4].split(".")[0])  # Extract the number before .pth
        
            # Load the checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            valid_accuracies = load_checkpoint_for_analysis(checkpoint_path)

        if valid_accuracies:
                # Get the maximum validation accuracy from this run
            max_valid_acc = max(valid_accuracies)
            print(f"Model: {filename} | Max Validation Accuracy: {max_valid_acc}")

                # Compare with the best validation accuracy
            if max_valid_acc > best_valid_acc:
                best_valid_acc = max_valid_acc
                best_gamma = gamma
                best_step_size = step_size

    print(f"Best model found with gamma={best_gamma}, step_size={best_step_size}, validation accuracy={best_valid_acc}")