import torch
import torch.multiprocessing as mp
from itertools import product
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

def train_model_combination(model, train_loader, valid_loader, combination_idx, gamma, decay_epochs, initial_model_state, initial_optimizer_state, cfg, checkpoint_dir, device_idx, logger):
    logger.info(f"Starting training for combination {combination_idx} with gamma={gamma} and step_size={decay_epochs} on device {device_idx}")

    # Set the current device to the specific GPU
    device = torch.device(f'cuda:{device_idx}')
    torch.cuda.set_device(device)

    
    
    model.load_state_dict(initial_model_state)
    model.to(device)

    # Initialize optimizer and load initial optimizer state
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['initial_lr'])
    optimizer.load_state_dict(initial_optimizer_state)

    # Initialize KLDivLoss
    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    # Train and validate for this combination of gamma and decay_epochs
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_and_validate_eeg(
        model, train_loader, valid_loader, epochs=cfg['EPOCHS'], 
        optimizer=optimizer, criterion=criterion, device=device, checkpoint_dir=checkpoint_dir, 
        logger=logger, new_checkpoint=True, combination_idx=combination_idx, gamma=gamma, step_size=decay_epochs
    )

    # Calculate mean validation accuracy
    avg_valid_acc = np.mean(valid_accuracies)
    logger.info(f"Validation accuracy for gamma={gamma}, decay_epochs={decay_epochs}: {avg_valid_acc:.4f}")

    # Return the result of this combination
    return (combination_idx, gamma, decay_epochs, avg_valid_acc)

def parallel_grid_search(cfg, model, train_loader, valid_loader, initial_model_state, initial_optimizer_state, logger):
    # Define grid search parameters
    gamma_values = np.linspace(0.9, 0.99, num=5)
    decay_epochs_values = [2, 3, 4]

    # List all available GPUs
    available_gpus = list(range(torch.cuda.device_count()))

    # Create parameter grid
    param_grid = list(product(gamma_values, decay_epochs_values))
    num_combinations = len(param_grid)

    # Initialize the process pool
    pool = mp.Pool(processes=len(available_gpus))  # Use all available GPUs

    # Start grid search
    results = []
    for combination_idx, (gamma, decay_epochs) in enumerate(param_grid):
        device_idx = available_gpus[combination_idx % len(available_gpus)]  # Round-robin GPU allocation
        result = pool.apply_async(train_model_combination, args=(model, train_loader, valid_loader,
            combination_idx, gamma, decay_epochs, initial_model_state, initial_optimizer_state, cfg, cfg['checkpoint_dir'], device_idx, logger
        ))
        results.append(result)

    # Collect results
    best_valid_acc = 0
    best_gamma = None
    best_decay_epochs = None
    best_model_state = None

    for result in results:
        combination_idx, gamma, decay_epochs, avg_valid_acc = result.get()  # Wait for result
        logger.info(f"Result for combination {combination_idx}: gamma={gamma}, decay_epochs={decay_epochs}, validation accuracy={avg_valid_acc:.4f}")
        
        # Track the best configuration
        if avg_valid_acc > best_valid_acc:
            best_valid_acc = avg_valid_acc
            best_gamma = gamma
            best_decay_epochs = decay_epochs
            # Save the model state (you can do this if you also save model in the function)
            best_model_state = initial_model_state  # You can save model state within the train function and return here

    logger.info(f"Grid search completed. Best gamma={best_gamma}, decay_epochs={best_decay_epochs}, validation accuracy={best_valid_acc:.4f}")

    # Save the best model configuration to disk
    if best_model_state:
        best_model_checkpoint_path = os.path.join(cfg['checkpoint_dir'], 'best_model.pth')
        torch.save(best_model_state, best_model_checkpoint_path)
        logger.info(f"Best model saved to {best_model_checkpoint_path}")

    pool.close()
    pool.join()
