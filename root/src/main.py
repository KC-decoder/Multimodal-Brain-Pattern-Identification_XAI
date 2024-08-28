import os
import time
import yaml
import pandas as pd
import torch
from torchsummary import summary
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.data_utils import createTrainTestSplit,create_k_fold_splits
from utils.data_utils import plot_spectrograms
from data.dataset import HMS_EEG_Dataset
from data.dataset import HMS_Spectrogram_Dataset
from data.dataset import CombinedDataset
from models.models import EnhancedEEGNetAttention
from training.training import train_and_validate_eeg 
from utils.config_loader import load_config
from utils.logger_utils import setup_logger
from contextlib import redirect_stdout
import io



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

    fold_indices = create_k_fold_splits(metadata, n_splits=5)
    logger.info("Created K-Fold splits.")

    for fold_idx in range(len(fold_indices)):
        train_metadata, valid_metadata = createTrainTestSplit(metadata, fold_indices, fold_idx)
        logger.info(f"Processing fold {fold_idx + 1}/{len(fold_indices)}")

    eeg_train_dataset = HMS_EEG_Dataset(train_metadata, cfg=cfg)
    logger.info(f"Size of the train dataset: {len(eeg_train_dataset)}")

    eeg_valid_dataset = HMS_EEG_Dataset(valid_metadata, cfg=cfg)
    logger.info(f"Size of the valid dataset: {len(eeg_valid_dataset)}")

    eeg_train_loader = DataLoader(eeg_train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    eeg_valid_loader = DataLoader(eeg_valid_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])

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
    
    
    # Initialize the model
    EEG_model = EnhancedEEGNetAttention(num_channels=37, num_samples=3000, num_classes=6)
    EEG_model.to(cfg['device'])
    
    
    criterion = nn.KLDivLoss(reduction='batchmean')  # KLDivLoss for classification
    # Define the optimizer, including the weight_decay parameter
    optimizer = torch.optim.Adam(EEG_model.parameters(), lr=0.00300, weight_decay=EEG_model.weight_decay)
    # Initialize the scheduler with desired parameters
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.995,     # Very small reduction each time
    patience=2,       # Frequent reductions (every 2 epochs of no improvement)
    threshold=0.0001, # Improvement threshold
    threshold_mode='rel',
    cooldown=0,       # No cooldown period
    min_lr=1e-6       # Ensure learning rate never reduces to zero
)
    logger.info(EEG_model)

    # Initialize a string buffer to capture the summary output
    model_summary_str = io.StringIO()

    # Capture the model summary
    with redirect_stdout(model_summary_str):
        summary(EEG_model, input_size=(1, 37, 3000))
        # Get the captured summary as a string
        model_summary_str = model_summary_str.getvalue()
    
    logger.info("Training EEG Model.")
    
    
    
    train_losses, valid_losses, train_accuracies, valid_accuracies = train_and_validate_eeg(
    EEG_model, eeg_train_loader, eeg_valid_loader, epochs=50, optimizer=optimizer, criterion=criterion, scheduler=scheduler, device=cfg['device'], checkpoint_dir=cfg['checkpoint_dir'], logger = logger) 
    
    
    logger.info("Training and validation process completed.")


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
    model = EnhancedEEGNetAttention(num_channels=37, num_samples=3000, num_classes=6)
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