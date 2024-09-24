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
from training.training import train_and_validate_eeg 
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
    
   
    # Define grid search parameters
    gamma_values = np.linspace(0.9, 0.99, num=10)  # Define a range of gamma values for learning rate decay
    decay_epochs_values = [2,3,4,]  # Epoch intervals for decaying learning rate

    # Store the best configuration
    best_valid_acc = 0
    best_gamma = None
    best_decay_epochs = None
    best_model_state = None



    for gamma, decay_epochs in product(gamma_values, decay_epochs_values):
        logger.info(f"Starting grid search with gamma={gamma} and learning rate scheculer step size={decay_epochs}")
        
        
        # If an old model exists, delete it to free GPU memory
        if 'EEGNet_model' in locals():
            del EEGNet_model
            torch.cuda.empty_cache()
        
                # Define your model
        nb_classes = cfg['n_classes']  # Number of classes in your dataset
        Chans = 37
        Samples = 3000
        
        # Initialize model, optimizer, and scheduler
        EEGNet_model = DeepConvNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, dropoutRate=0.5)
        EEGNet_model.to(cfg['device'])
        optimizer = torch.optim.Adam(EEGNet_model.parameters(), lr=cfg['initial_lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = decay_epochs, gamma=gamma)
        # Initialize KLDivLoss
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        
        device = cfg['device']
        # Print model device information for debugging
        print(f"Model is on device: {next(EEGNet_model.parameters()).device}")


        # Train and validate
        train_losses, valid_losses, train_accuracies, valid_accuracies = train_and_validate_eeg(
            EEGNet_model, eeg_train_loader, eeg_valid_loader, epochs=cfg['EPOCHS'], 
            optimizer=optimizer, criterion=criterion, scheduler=scheduler, 
            device=cfg['device'], checkpoint_dir=cfg['checkpoint_dir'], logger=logger, new_checkpoint=True
        )

    # Calculate the mean validation accuracy across all folds
    avg_valid_acc = np.mean(valid_accuracies)
    logger.info(f"Average validation accuracy for gamma={gamma}, decay_epochs={decay_epochs}: {avg_valid_acc:.4f}")

    # Track the best configuration
    if avg_valid_acc > best_valid_acc:
        best_valid_acc = avg_valid_acc
        best_gamma = gamma
        best_decay_epochs = decay_epochs
        best_model_state = EEGNet_model.state_dict()

    logger.info(f"Grid search completed. Best gamma={best_gamma}, decay_epochs={best_decay_epochs}, validation accuracy={best_valid_acc:.4f}")

def unload_model(model):
    model.to('cpu')
    del model
    torch.cuda.empty_cache()
    print("Model has been unloaded from GPU memory and deleted.")


def distributed_main(rank, world_size, cfg):

    cfg = load_config()
    # Initialize the distributed environment
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Initialize the distributed environment
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # Setup logger
    logger = setup_logger()
    
    # Initialize datasets and dataloaders
    metadata = pd.read_csv('/data2/users/koushani/HMS_data/train.csv')
        
    
    fold_indices = create_k_fold_splits(metadata, n_splits=5)
    train_metadata, valid_metadata = createTrainTestSplit(metadata, fold_indices, fold_idx=0)  # Example with fold 0
    
    train_dataset = HMS_EEG_Dataset(train_metadata, cfg=cfg)
    valid_dataset = HMS_EEG_Dataset(valid_metadata, cfg=cfg)
    
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], sampler=train_sampler, num_workers=cfg['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], sampler=valid_sampler, num_workers=cfg['num_workers'])

    # Initialize the model and wrap with DDP
    #model = EnhancedEEGNetAttention(num_channels=37, num_samples=3000, num_classes=6)
    model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Define loss, optimizer, scheduler
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00400, weight_decay=model.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.995, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6)

    # Train and validate the model
    train_and_validate_eeg(model, train_loader, valid_loader, epochs=cfg['epochs'], optimizer=optimizer, 
                           criterion=criterion, scheduler=scheduler, device=rank, 
                           checkpoint_dir=cfg['checkpoint_dir'], logger=logger, rank=rank, world_size=world_size)

    # Clean up the distributed environment
    dist.destroy_process_group()

def run_distributed_training(cfg):
    world_size = torch.cuda.device_count()
    mp.spawn(distributed_main, args=(world_size, cfg), nprocs=world_size, join=True)




if __name__ == "__main__":
    main()
    unload_model()