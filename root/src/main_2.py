
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
if CFG.train_models:
    oof = np.zeros((len(train), N_CLASSES))
    prfs = []

    cv = GroupKFold(n_splits=5)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(train, train[TGT_COL], train["patient_id"])):
        logger.info(f"== Train and Eval Process - Fold{fold} ==")

        # Build dataloaders
        data_tr, data_val = train.iloc[tr_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)
        train_loader = DataLoader(
            EEGDataset({"meta": data_tr, "eeg": all_eegs}, "train", **CFG.dataset),
            shuffle=CFG.trainer["dataloader"]["shuffle"],
            batch_size=CFG.trainer["dataloader"]["batch_size"],
            num_workers=CFG.trainer["dataloader"]["num_workers"]
        )
        val_loader = DataLoader(
            EEGDataset({"meta": data_val, "eeg": all_eegs}, "valid", **CFG.dataset),
            shuffle=False,
            batch_size=CFG.trainer["dataloader"]["batch_size"],
            num_workers=CFG.trainer["dataloader"]["num_workers"]
        )



        logger.info(f"Number of samples in train_loader before augmentation: {len(train_loader.dataset)}")

        # print(f"Using device: {CFG["gpu"]})
        # print(f"train_steps:{train_steps}")
        # Get one batch
        batch = next(iter(train_loader))

        # Inspect shapes
        logger.info("x shape:", batch["x"].shape)  # Should be [B, C=8, T=2000]
        logger.info("y shape:", batch["y"].shape)  # Should be [B, N_CLASSES]
            
        real_data = []

        for batch in train_loader:
            xs, ys = batch["x"], batch["y"]  # xs: [B, 8, 2000], ys: [B, N_CLASSES]
            for x, y in zip(xs, ys):
                real_data.append((x, y))
        # print(f"Created real data dictionary of shape: {real_data.shape}")
                
                
        generated_data = []
        gen_data_dir = Path("/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/generated_data")  # Where to save .npy files   
        for class_idx in range(CFG.N_CLASSES):
            path = gen_data_dir / f"generated_class_{class_idx}.npy"
            if not path.exists():
                print(f"Warning: {path} missing")
                continue

            arr = np.load(path)  # Shape: (N, 8, 2000)
            
            # Create one-hot vector for this class
            target = torch.zeros(CFG.N_CLASSES, dtype=torch.float32)
            target[class_idx] = 1.0

            for i in range(arr.shape[0]):
                x = torch.tensor(arr[i], dtype=torch.float32)  # shape: [8, 2000]
                generated_data.append((x, target.clone()))
            # print(f"Created generated data with shape: {generated_data.shape}")
            
        combined_data = real_data + generated_data
        combined_data = shuffle(combined_data, random_state=CFG.seed)
        
        
    
            
        
        augmented_train_dataset = CombinedEEGDataset(combined_data)

        augmented_train_loader = DataLoader(
            augmented_train_dataset,
            shuffle=True,
            batch_size=CFG.diffEEG_trainer["batch_size"],
            num_workers=CFG.diffEEG_trainer["num_workers"]
        )
        
        logger.info(f"Number of samples in augmented_train_loader: {len(augmented_train_loader.dataset)}")
        
        batch_aug = next(iter(augmented_train_loader))
        logger.info("Augmented x shape:", batch_aug["x"].shape)
        logger.info("Augmented y shape:", batch_aug["y"].shape)



        # Build model
        logger.info(f"Build model...")
        model = DilatedInceptionWaveNet()
        model.to(CFG.device)

        # Build criterion
        loss_fn = KLDivWithLogitsLoss()

        # Build solvers
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.trainer["lr"])
        num_training_steps = (
            math.ceil(
                len(train_loader.dataset)
                / (CFG.trainer["dataloader"]["batch_size"] * CFG.trainer["grad_accum_steps"])
            )
            * CFG.trainer["epochs"]
        )
        lr_skd = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        # Build evaluator
        evaluator = Evaluator(metric_names=["kldiv"])

        # Build trainer
        trainer: _BaseTrainer = None
        trainer = MainTrainer(
            logger=logger,
            trainer_cfg=CFG.trainer,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_skd=lr_skd,
            ckpt_path=CFG.exp_dump_path,
            evaluator=evaluator,
            scaler=None,
            train_loader=train_loader,
            eval_loader=val_loader,
            use_wandb=False
        )

        # Run main training and evaluation for one fold
        y_preds = trainer.train_eval(fold)
        oof[val_idx, :] = y_preds["val"]

        # Dump output objects
        for model_path in CFG.exp_dump_path.glob("*.pth"):
            if "seed" in str(model_path) or "fold" in str(model_path):
                continue

            # Rename model file
            model_file_name_dst = f"{model_path.stem}_fold{fold}.pth"
            model_path_dst = exp.ckpt_path / model_file_name_dst
            model_path.rename(model_path_dst)

        # Free mem.
        del (data_tr, data_val, train_loader, val_loader, model, optimizer, lr_skd, evaluator, trainer)
        _ = gc.collect()

        if CFG.one_fold_only:
            logger.info("Cross-validatoin stops at first fold!!!")
            break

    np.save(CFG.exp_dump_pat / "oof.npy", oof)
else:
    file_path = DATA_PATH / "kaggle/input/hms-oof-demo/oof_seed0.npy"
    print(f"Checking file path: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    oof = np.load(file_path)