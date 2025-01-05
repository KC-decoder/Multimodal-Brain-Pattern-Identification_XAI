
# import os
# import time
# import yaml
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# from torchsummary import summary
# from torch import nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau , StepLR, LambdaLR
# from torch.utils.data import DataLoader
# from utils.data_utils import createTrainTestSplit,create_k_fold_splits,analyze_checkpoints
# from utils.data_utils import plot_spectrograms
# from utils.training_utils import initialize_kaiming_weights , warmup_cosine_schedule
# from data.dataset import HMS_EEG_Dataset
# from data.dataset import HMS_Spectrogram_Dataset
# from data.dataset import CombinedDataset
# from models.models import  EEGNet, DeepConvNet, SpectrogramViT
# from training.training import train_and_validate_eeg,  train_spectrogram_model
# from utils.config_loader import load_config
# from utils.logger_utils import setup_logger
# from contextlib import redirect_stdout
# from itertools import product
# import io
# import sys
# from io import StringIO
# from itertools import product
# import numpy as np
# import torch.multiprocessing as mp
# import torch.distributed as dist
# import torch
# import argparse




# def main(gpu):
#     print(f"Running at GPU: {gpu}" )
#     cfg = load_config()
#     # Setup logger
#     logger = setup_logger()

#     # Initialize datasets
#     # Assuming 'metadata' is a DataFrame containing the necessary data
#     # Replace this with your actual data loading process
#     # Log the start of the process
#     logger.info("Observing training")

#     # Initialize datasets
#     # Assuming 'metadata' is a DataFrame containing the necessary data
#     # Replace this with your actual data loading process
#     metadata = pd.read_csv('/data2/users/koushani/HMS_data/train.csv')
#     logger.info(f"Loaded metadata from {cfg['root_dir']}")
    
    
    
    
#         # Assuming your metadata dataframe is called 'metadata' and the label column is 'label_column'
#     class_counts = metadata['expert_consensus'].value_counts()

#     # Print the distribution of classes
#     print("Class distribution:")
#     print(class_counts)

#     # # Plot the distribution
#     # plt.figure(figsize=(10, 6))
#     # class_counts.plot(kind='bar')
#     # plt.title('Class Distribution')
#     # plt.xlabel('Class')
#     # plt.ylabel('Frequency')
#     # plt.show()

    
#     # Check if CUDA (GPU) is available
#     if torch.cuda.is_available():
#         torch.cuda.set_device(gpu) # Specify GPU
#         print(f"GPU {gpu} has been set as the device: {torch.cuda.get_device_name(cfg['device'])}")
#     else:
#         cfg['device'] = torch.device("cpu")
#         print("GPU is not available, using CPU.")
    
#     fold_indices = create_k_fold_splits(metadata, n_splits=5)
#     logger.info("Created K-Fold splits.")
    

#     for fold_idx in range(len(fold_indices)):
#         train_metadata, valid_metadata = createTrainTestSplit(metadata, fold_indices, fold_idx)
        
#         logger.info(f"Processing fold {fold_idx + 1}/{len(fold_indices)}")

# #     eeg_train_dataset = HMS_EEG_Dataset(train_metadata, cfg=cfg)
# #     logger.info(f"Size of the train dataset: {len(eeg_train_dataset)}")

# #     eeg_valid_dataset = HMS_EEG_Dataset(valid_metadata, cfg=cfg)
# #     logger.info(f"Size of the valid dataset: {len(eeg_valid_dataset)}")
# #     if cfg['debug']:
# #         batch_size = cfg['debug_batch_size']
# #     else:
# #         batch_size = cfg['batch_size']

# #     eeg_train_loader = DataLoader(eeg_train_dataset, batch_size= batch_size, shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
# #     eeg_valid_loader = DataLoader(eeg_valid_dataset, batch_size= batch_size, shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)

# #     # start_time = time.time()
# #     # # Iterate over batches and log the shape of the final DataLoader object
# #     # for batch in eeg_train_loader:
# #     #     data, labels = batch
# #     #     logger.info(f"Batch data shape: {data.shape}")  # Logs batch size and dimensions
# #     #     logger.info(f"Batch labels shape: {labels.shape}")  # Logs label dimensions
# #     #     break
# #     # end_time = time.time()
# #     # duration = end_time - start_time
# #     # logger.info(f"took {duration} seconds to display dataloader info.")
    
    
    
    
    
    
    
    
    
# #     # Define your model
# #     nb_classes = cfg['n_classes']  # Number of classes in your dataset
# #     Chans = 37
# #     Samples = 3000
    
# #     warmup_epochs = 10
# #     initial_lr = 0.001
# #     target_lr = 0.004
# #     min_lr = 0.00001
    
    
    
    
    
# #     # # Path to the checkpoint
# #     # checkpoint_path = "/eng/home/koushani/Documents/Multimodal_XAI/Brain-Pattern-Identification-using-Multimodal-Classification/checkpoint_dir/Learning_rate_grid_search"
# #     # checkpoint_filename = "eeg_checkpoint_index_3.pth.tar"

    
# #     # # Load the checkpoint
# #     # checkpoint_path = os.path.join(cfg['checkpoint_dir'], checkpoint_filename)
# #     # checkpoint = torch.load(checkpoint_path)

# #     # # Extract saved parameters from the checkpoint
# #     # model_state_dict = checkpoint['state_dict']
# #     # optimizer_state_dict = checkpoint['optimizer']
# #     # train_losses = checkpoint.get('train_losses', [])
# #     # valid_losses = checkpoint.get('valid_losses', [])
# #     # train_accuracies = checkpoint.get('train_accuracies', [])
# #     # valid_accuracies = checkpoint.get('valid_accuracies', [])
# #     # start_epoch = checkpoint.get('epoch', 0)
    
    
# #     # # Print the loaded information
# #     # # logger.info(f"Checkpoint loaded from epoch {start_epoch}")
# #     # logger.info(f"Last Training Loss: {train_losses[-1] if train_losses else 'N/A'}")
# #     # logger.info(f"Last Validation Loss: {valid_losses[-1] if valid_losses else 'N/A'}")
# #     # logger.info(f"Last Training Accuracy: {train_accuracies[-1] if train_accuracies else 'N/A'}")
# #     # logger.info(f"Last Validation Accuracy: {valid_accuracies[-1] if valid_accuracies else 'N/A'}")
    
    
# #    # Initialize model, optimizer, and scheduler
# #     EEGNet_model_3 = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, dropoutRate=0.5)
# #     # Print weights before initialization
# #     #print("Weights before initialization:")
# #     #print(EEGNet_model_3.conv1.weight)

# #     # Apply the weight initialization
# #     EEGNet_model_3.apply(initialize_kaiming_weights)

# #     # Print weights after initialization
# #     #print("Weights after initialization:")
# #     #print(EEGNet_model_3.conv1.weight)
# #     EEGNet_model_3.to(cfg['device'])
# #     optimizer = torch.optim.Adam(EEGNet_model_3.parameters(), lr=initial_lr)
# #     # Define scheduler
# #     # Define the LambdaLR scheduler
# #     #lambda_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_cosine_schedule(epoch, warmup_epochs, total_epochs, initial_lr, target_lr, min_lr))

# #     # Initialize KLDivLoss
# #     criterion = nn.KLDivLoss(reduction='batchmean')


# #     # # Load model and optimizer state from checkpoint
# #     # EEGNet_model.load_state_dict(model_state_dict)
# #     # optimizer.load_state_dict(optimizer_state_dict)



# #     logger.info(f"Starting training with Learning Rate: {initial_lr}, warming up to {target_lr} within {warmup_epochs} epochs, followed by learning rate scheduling using cosine annealing to {min_lr} ")
        
# #         # # If an old model exists, delete it to free GPU memory
# #         # if 'EEGNet_model' in locals():
# #         #     del EEGNet_model
# #         #     torch.cuda.empty_cache()

        

# #         # Train and validate
# #     train_and_validate_eeg(EEGNet_model_3, eeg_train_loader, eeg_valid_loader, epochs = cfg['EPOCHS'], optimizer = optimizer, criterion = criterion, device = cfg['device'], checkpoint_dir = cfg['checkpoint_dir'], logger = logger, new_checkpoint = True, initial_lr = initial_lr, peak_lr = target_lr, warmup_epochs = warmup_epochs, min_lr = min_lr)
        

# #     # # Calculate the mean validation accuracy
# #     # avg_valid_acc = np.mean(valid_accuracies)
# #     # logger.info(f"Validation accuracy for gamma={gamma}, decay_epochs={decay_epochs}: {avg_valid_acc:.4f}")

# #     # Save the model state
# #     # model_checkpoint_path = os.path.join(cfg['checkpoint_dir'], f'model_reduce_on_plateau.pth')
# #     # torch.save(EEGNet_model_2.state_dict(), model_checkpoint_path)
# #     # logger.info(f"Model saved to {model_checkpoint_path}")





#     spec_train_dataset = HMS_Spectrogram_Dataset(train_metadata, cfg=cfg)
#     print(f"Size of the train dataset: {len(spec_train_dataset)}")

#     spec_valid_dataset = HMS_Spectrogram_Dataset(valid_metadata, cfg=cfg)
#     print(f"Size of the valid dataset: {len(spec_valid_dataset)}")


#     spec_train_loader = DataLoader(spec_train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
#     spec_valid_loader = DataLoader(spec_valid_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
    
#     start_time = time.time()
#     # Iterate over batches and log the shape of the final DataLoader object
#     for batch in spec_train_loader:
#         data, labels = batch
#         print(f"Batch data shape: {data.shape}")  # Logs batch size and dimensions
#         print(f"Batch labels shape: {labels.shape}")  # Logs label dimensions
#         break
#     end_time = time.time()
#     duration = end_time - start_time
#     print(f"took {duration} seconds to display dataloader info.")
    
#     # Create the ViT model
#     SPEC_model = SpectrogramViT(image_size=(400, 300), num_classes=6)

#     # Device setup
#     SPEC_model.to(cfg['device'])
#     logger.info(SPEC_model)
    
    
#     # Loss function adjusted for KL divergence
#     criterion = nn.KLDivLoss(reduction='batchmean')
#     optimizer = torch.optim.Adam(SPEC_model.parameters(), lr=0.001)
    
#     logger.info(f"Starting training")
    
#     train_spectrogram_model(SPEC_model, spec_train_loader, spec_valid_loader,epochs = cfg['EPOCHS'], optimizer = optimizer, criterion = criterion, device = cfg['device'], checkpoint_dir = cfg['checkpoint_dir'], logger = logger, new_checkpoint = True)


# if __name__ == "__main__":
#     gpu = 1
#     main(gpu)
    