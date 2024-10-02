import os
import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau , StepLR
from torch.utils.data import DataLoader
from utils.data_utils import createTrainTestSplit,create_k_fold_splits
from utils.data_utils import plot_spectrograms
from utils.training_utils import parallel_grid_search 
from data.dataset import HMS_EEG_Dataset
from data.dataset import HMS_Spectrogram_Dataset
from data.dataset import CombinedDataset
from models.models import  EEGNet, DeepConvNet
from training.training import train_and_validate_eeg_manual_lr_grid_search, train_and_validate_eeg
from utils.config_loader import load_config
from utils.logger_utils import setup_logger
from contextlib import redirect_stdout
from itertools import product
import io
import sys
from io import StringIO
from itertools import product
import numpy as np



import torch.multiprocessing as mp
import torch.distributed as dist
import torch




def main():
    cfg = load_config()
    # Setup logger
    logger = setup_logger()

    # Initialize datasets
    # Assuming 'metadata' is a DataFrame containing the necessary data
    # Replace this with your actual data loading process
    # Log the start of the process
    logger.info("Starting the training and validation process.")

    # Initialize datasets
    # Assuming 'metadata' is a DataFrame containing the necessary data
    # Replace this with your actual data loading process
    metadata = pd.read_csv('/data2/users/koushani/HMS_data/train.csv')
    logger.info(f"Loaded metadata from {cfg['root_dir']}")
    
    
    
    
        # Assuming your metadata dataframe is called 'metadata' and the label column is 'label_column'
    class_counts = metadata['expert_consensus'].value_counts()

    # Print the distribution of classes
    print("Class distribution:")
    print(class_counts)

    # # Plot the distribution
    # plt.figure(figsize=(10, 6))
    # class_counts.plot(kind='bar')
    # plt.title('Class Distribution')
    # plt.xlabel('Class')
    # plt.ylabel('Frequency')
    # plt.show()

    
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        torch.cuda.set_device(2) # Specify GPU 2 (third GPU)
        print(f"GPU 2 has been set as the device: {torch.cuda.get_device_name(cfg['device'])}")
    else:
        cfg['device'] = torch.device("cpu")
        print("GPU is not available, using CPU.")
    
    fold_indices = create_k_fold_splits(metadata, n_splits=5)
    logger.info("Created K-Fold splits.")
    

    for fold_idx in range(len(fold_indices)):
        train_metadata, valid_metadata = createTrainTestSplit(metadata, fold_indices, fold_idx)

        logger.info(f"Processing fold {fold_idx + 1}/{len(fold_indices)}")

    eeg_train_dataset = HMS_EEG_Dataset(train_metadata, cfg=cfg)
    logger.info(f"Size of the train dataset: {len(eeg_train_dataset)}")

    eeg_valid_dataset = HMS_EEG_Dataset(valid_metadata, cfg=cfg)
    logger.info(f"Size of the valid dataset: {len(eeg_valid_dataset)}")

    eeg_train_loader = DataLoader(eeg_train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
    eeg_valid_loader = DataLoader(eeg_valid_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)

    start_time = time.time()
    # Iterate over batches and log the shape of the final DataLoader object
    for batch in eeg_train_loader:
        data, labels = batch
        logger.info(f"Batch data shape: {data.shape}")  # Logs batch size and dimensions
        logger.info(f"Batch labels shape: {labels.shape}")  # Logs label dimensions
        break
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"took {duration} seconds to display dataloader info.")
    
    
    
    # Define your model
    nb_classes = cfg['n_classes']  # Number of classes in your dataset
    Chans = 37
    Samples = 3000
   # Initialize model, optimizer, and scheduler
    EEGNet_model = DeepConvNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, dropoutRate=0.5)
    EEGNet_model.to(cfg['device'])
    optimizer = torch.optim.Adam(EEGNet_model.parameters(), lr=cfg['initial_lr'])
    # Initialize KLDivLoss
    criterion = nn.KLDivLoss(reduction='batchmean')


    # Save initial model and optimizer state
    initial_model_state = EEGNet_model.state_dict()
    initial_optimizer_state = optimizer.state_dict()

    # Start the parallel grid search
    parallel_grid_search(cfg, EEGNet_model, eeg_train_loader, eeg_train_loader, initial_model_state, initial_optimizer_state, logger)



if __name__ == "__main__":
    main()