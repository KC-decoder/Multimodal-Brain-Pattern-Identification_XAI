import gc
import logging
import math
import json
import os
import random
import sys
import pickle
import warnings
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.signal import butter, lfilter
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
import ipywidgets as widgets

from utils.training_utils import KLDivWithLogitsLoss, Evaluator, _ModelCheckpoint, _BaseTrainer, MainTrainer
from models.models import DilatedInceptionWaveNet
from data.dataset import EEGDataset , CombinedEEGDataset, DummyEEGDataset
from utils.cfg_utils import CFG, _Logger, _seed_everything, _get_eeg_window
from models.diffEEG import DiffEEG_SanityCheck
from utils.DiffEEG_utils import * 
from training.DiffEEG_train import DiffEEGTrainer

import matplotlib.pyplot as plt
from scipy.signal import stft

    
    
    
TGT_VOTE_COLS = CFG.TGT_VOTE_COLS
EEG_WLEN = CFG.EEG_WLEN
EEG_FREQ = CFG.EEG_FREQ 
TGT_COL = CFG.TGT_COL   
N_CLASSES = CFG.N_CLASSES
EEG_PTS = CFG.EEG_PTS


if not CFG.exp_dump_path.exists():
    os.mkdir(CFG.exp_dump_path)
    
logger = _Logger(logging_file=CFG.exp_dump_path / "train_eval.log").get_logger()
_seed_everything(CFG.seed)


DATA_PATH = CFG.DATA_PATH
train = pd.read_csv(DATA_PATH / "train.csv")
print(f"Train data shape | {train.shape}")


# Define paths
eeg_file_path = DATA_PATH/"kaggle" /"input"/ "brain-eegs" / "eegs_all_channles.npy"


# Ensure directories exist
eeg_file_path.parent.mkdir(parents=True, exist_ok=True)


# Unique EEG IDs
uniq_eeg_ids = train["eeg_id"].unique()
n_uniq_eeg_ids = len(uniq_eeg_ids)

# Check if file exists
if not eeg_file_path.exists():
    logger.info("Generate cropped EEGs...")
    all_eegs = {}
    
    for i, eeg_id in tqdm(enumerate(uniq_eeg_ids), total=n_uniq_eeg_ids):
        eeg_win = _get_eeg_window(DATA_PATH / "train_eegs" / f"{eeg_id}.parquet")
        all_eegs[eeg_id] = eeg_win
    
    # Save the new file
    np.save(eeg_file_path, all_eegs)
    logger.info(f"Saved EEGs to {eeg_file_path}")
else:
    # logger.info("Load cropped EEGs...")
    all_eegs = np.load(eeg_file_path, allow_pickle=True).item()
    assert len(all_eegs) == n_uniq_eeg_ids






# Debug: Print a sample EEG shape
logger.info(f"Demo EEG shape | {list(all_eegs.values())[0].shape}")

logger.info(f"Process labels...")
df_tmp = train.groupby("eeg_id")[["patient_id"]].agg("first")
labels_tmp = train.groupby("eeg_id")[TGT_VOTE_COLS].agg("sum")
for col in TGT_VOTE_COLS:
    df_tmp[col] = labels_tmp[col].values

# Normalize target columns
y_data = df_tmp[TGT_VOTE_COLS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
df_tmp[TGT_VOTE_COLS] = y_data

tgt = train.groupby("eeg_id")[["expert_consensus"]].agg("first")
df_tmp[TGT_COL] = tgt 

train = df_tmp.reset_index()
logger.info(f"Training DataFrame shape | {train.shape}")

# if CFG.add_augmentation:
#     # ---- Settings ----
#     checkpoint_path = Path("/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/models/DiffEEG_model_1000.pt" )  # ← Update path here
#     output_dir = Path("/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/generated_data_50")  # Where to save .npy files
#     output_dir.mkdir(parents=True, exist_ok=True)
#     n_samples_per_class = 50

#     # ---- Logger ----
#     logger.info("Starting Augmented EEG generation for each class...")

#     # ---- Load Model ----
#     logger.info("Loading Diffusion model...")
#     model = DiffEEG_Updated(config=CFG).to(CFG.device)
#     checkpoint = torch.load(checkpoint_path, map_location=CFG.device)
#     model.load_state_dict(checkpoint['model'])
#     model.eval()

#     # ---- Diffusion Wrapper ----
#     diffusion_module = DiffEEGDiffusion(model=model, config=CFG, device=CFG.device)



#     # ---- Generate for All Classes ----
#     for cls_id in range(CFG.N_CLASSES):
#         logger.info(f"Generating {n_samples_per_class} EEGs for class {cls_id}")
#         eeg_data = generate_for_class(cls_id, n_samples_per_class, diffusion_module)

#         # Save
#         save_path = output_dir / f"generated_class_{cls_id}.npy"
#         np.save(save_path, eeg_data)
#         logger.info(f"Saved generated EEGs for class {cls_id} to {save_path}")

#     logger.info("Done generating EEGs for all classes.")
        
        
# # **Start Training**
if CFG.train_models:
    oof = np.zeros((len(train), N_CLASSES))
    prfs = []

    cv = GroupKFold(n_splits=2)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(train, train[TGT_COL], train["patient_id"])):
        logger.info(f"== Train and Eval Process - Fold{fold} ==")

        # Build dataloaders
        # data_tr, data_val = train.iloc[tr_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)
        # train_loader = DataLoader(
        #     EEGDataset({"meta": data_tr, "eeg": all_eegs}, "train", **CFG.dataset),
        #     shuffle=CFG.trainer["dataloader"]["shuffle"],
        #     batch_size=CFG.trainer["dataloader"]["batch_size"],
        #     num_workers=CFG.trainer["dataloader"]["num_workers"]
        # )
        # val_loader = DataLoader(
        #     EEGDataset({"meta": data_val, "eeg": all_eegs}, "valid", **CFG.dataset),
        #     shuffle=False,
        #     batch_size=CFG.trainer["dataloader"]["batch_size"],
        #     num_workers=CFG.trainer["dataloader"]["num_workers"]
        # )



        # logger.info(f"Number of samples in train_loader before augmentation: {len(train_loader.dataset)}")

        # print(f"Using device: {CFG["gpu"]})
        # print(f"train_steps:{train_steps}")
        # Get one batch
        # batch = next(iter(train_loader))

        # # Inspect shapes
        # logger.info(f"x shape: {batch['x'].shape}")
        # logger.info(f"y shape: {batch['y'].shape}")
        
        
        # # Step 1–3: Collect one sample per class
        # seen_classes = set()
        # dummy_samples = []

        # for batch in train_loader:
        #     xs, ys = batch["x"], batch["y"]  # xs: [B, C, T], ys: [B, N_CLASSES]
        #     for x, y in zip(xs, ys):
        #         label = y.argmax().item()
        #         if label not in seen_classes:
        #             dummy_samples.append((x, y))
        #             seen_classes.add(label)
        #         if len(seen_classes) == CFG.N_CLASSES:
        #             break
        #     if len(seen_classes) == CFG.N_CLASSES:
        #         break

        # # Step 4: Create the dummy dataset
        # dummy_dataset = DummyEEGDataset(dummy_samples)
        # dummy_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)

        # print(f"Dummy dataset ready with {len(dummy_dataset)} samples (1 per class)")
        
        # batch = next(iter(dummy_loader))

        # # Inspect shapes
        # logger.info(f"x shape: {batch['x'].shape}")
        # logger.info(f"y shape: {batch['y'].shape}")
        
        save_dir = CFG.exp_dump_path / f"mnist_validation_samples"
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(root="./", train=True, download=True, transform=transform)
        loader = DataLoader(train_data, batch_size=32, shuffle=True)
        x_batch = next(iter(loader))
        x_batch = x_batch.to(CFG.device)

        


        model = DiffEEG_SanityCheck().to(CFG.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(50):
            model.train()
            opt.zero_grad()
            out = model(x_batch)
            loss = criterion(out, x_batch)
            loss.backward()
            opt.step()

        
            if epoch % 10 == 0:
                os.makedirs(save_dir, exist_ok=True)    
                model.eval()
                x, _ = next(iter(x_batch))
                x = x.to(CFG.device)
                with torch.no_grad():
                    y = model(x, class_label=None)

                # Plot first 5 samples
                for i in range(5):
                    plt.subplot(2, 5, i+1)
                    plt.imshow(x[i].squeeze().cpu().numpy(), cmap='gray')
                    plt.title("Input")
                    plt.axis("off")

                    plt.subplot(2, 5, i+6)
                    plt.imshow(y[i].squeeze().cpu().numpy(), cmap='gray')
                    plt.title("Output")
                    plt.axis("off")

                plt.tight_layout()
                plt.savefig(f"{save_dir}/epoch_{epoch}.png")
                plt.show()

        
    #     logger.info("Initializing DiffEEG Model for EEG Augmentation...")
    #     train_steps = CFG.diffEEG_trainer["epochs"] * len(dummy_loader) # can be customized to a fixed number, however, it should reflect the dataset size.
    #     train_steps = max(train_steps, 1000)
    #     print('train_steps:',train_steps)
    #     # **Initialize DiffEEG Model**
    #     model = DiffEEG_Updated(config=CFG).to(CFG.device)

    #     # **Define Diffusion Module**
    #     diffusion_module = DiffEEGDiffusion(model, CFG, device=CFG.device)

    #     # **Define Trainer**
    #     trainer = DiffEEGTrainer(
    #         logger = logger,
    #         # wandb_logger = wandb_logger,
    #         diffusion_module=diffusion_module,
    #         dataloader_train=dummy_loader,  # Pass DataLoader
    #         dataloader_val=dummy_loader,      
    #         config=CFG,
    #         device=CFG.device
    #     )

    #     # **Train the DiffEEG Model**
    #     logger.info(f"Results will be saved in: {CFG.diffEEG_trainer['results_folder']}")
    #     logger.info("Starting DiffEEG Training...")
    #     trainer.train()

    #     # **Save Final Model**
    #     model_save_path = CFG.diffEEG_trainer["results_folder"] / f"diffEEG_final_fold{fold}.pth"
    #     torch.save(model.state_dict(), model_save_path)
    #     logger.info(f"Final model saved to {model_save_path}")

    #     # **Run Final Evaluation**
    #     logger.info("Running Final Validation...")
    #     real_eeg, generated_eeg = trainer.generate_augmented_samples(n_samples = 100, class_label = 2)

    #     # **Save Generated EEG Samples**
    #     generated_save_path = CFG.exp_dump_path / f"generated_eegs_fold{fold}.npy"
    #     np.save(generated_save_path, generated_eeg)
    #     logger.info(f"Saved generated EEG samples to {generated_save_path}")

    # logger.info("Training Completed Successfully!")

        
        
            
        # real_data = []

        # for batch in train_loader:
        #     xs, ys = batch["x"], batch["y"]  # xs: [B, 8, 2000], ys: [B, N_CLASSES]
        #     for x, y in zip(xs, ys):
        #         real_data.append((x, y))
                
                
                
        # class_label = 2      
        # generated_data_dir = Path("/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/generated_data_50")
        # plot_save_path = Path(f"/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/plots/eeg_class_{class_label}.png")

        # plot_eeg_comparison(real_data, generated_data_dir, class_label=class_label, save_path=plot_save_path)
        # print(f"Created real data dictionary of shape: {real_data.shape}")
                
                
    #     generated_data = []
    #     gen_data_dir = Path("/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/generated_data_50")  # Where to save .npy files   
    #     for class_idx in range(CFG.N_CLASSES):
    #         path = gen_data_dir / f"generated_class_{class_idx}.npy"
    #         if not path.exists():
    #             print(f"Warning: {path} missing")
    #             continue

    #         arr = np.load(path)  # Shape: (N, 8, 2000)
            
    #         # Create one-hot vector for this class
    #         target = torch.zeros(CFG.N_CLASSES, dtype=torch.float32)
    #         target[class_idx] = 1.0

    #         for i in range(arr.shape[0]):
    #             x = torch.tensor(arr[i], dtype=torch.float32)  # shape: [8, 2000]
    #             generated_data.append((x, target.clone()))
    #         # print(f"Created generated data with shape: {generated_data.shape}")
            
    #     combined_data = real_data + generated_data
    #     combined_data = shuffle(combined_data, random_state=CFG.seed)
        
        
    
            
        
    #     augmented_train_dataset = CombinedEEGDataset(combined_data)

    #     augmented_train_loader = DataLoader(
    #         augmented_train_dataset,
    #         shuffle=True,
    #         batch_size=CFG.trainer["dataloader"]["batch_size"],
    #         num_workers=CFG.trainer["dataloader"]["num_workers"]
    #     )
        
    #     logger.info(f"Number of samples in augmented_train_loader: {len(augmented_train_loader.dataset)}")
        
    #     batch_aug = next(iter(augmented_train_loader))
    #     logger.info(f"Augmented x shape: {batch_aug['x'].shape}")
    #     logger.info(f"Augmented y shape: {batch_aug['y'].shape}")

    #     plot_class_distribution_comparison(
    #     real_data=real_data,
    #     generated_data=generated_data,
    #     n_classes=CFG.N_CLASSES,
    #     output_dir= CFG.exp_dump_path / "Plots"
    #     )

    #     # Build model
    #     logger.info(f"Build model...")
    #     model = DilatedInceptionWaveNet()
    #     model.to(CFG.device)

    #     # Build criterion
    #     loss_fn = KLDivWithLogitsLoss()

    #     # Build solvers
    #     optimizer = torch.optim.Adam(model.parameters(), lr=CFG.trainer["lr"])
    #     num_training_steps = (
    #         math.ceil(
    #             len(train_loader.dataset)
    #             / (CFG.trainer["dataloader"]["batch_size"] * CFG.trainer["grad_accum_steps"])
    #         )
    #         * CFG.trainer["epochs"]
    #     )
    #     lr_skd = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    #     # Build evaluator
    #     evaluator = Evaluator(metric_names=["kldiv"])

    #     # Build trainer
    #     trainer: _BaseTrainer = None
    #     trainer = MainTrainer(
    #         logger=logger,
    #         trainer_cfg=CFG.trainer,
    #         model=model,
    #         loss_fn=loss_fn,
    #         optimizer=optimizer,
    #         lr_skd=lr_skd,
    #         ckpt_path=CFG.exp_dump_path,
    #         evaluator=evaluator,
    #         scaler=None,
    #         train_loader=train_loader,
    #         eval_loader=val_loader,
    #         use_wandb=False
    #     )

    #     # Run main training and evaluation for one fold
    #     y_preds = trainer.train_eval(fold)
    #     oof[val_idx, :] = y_preds["val"]

    #     # Dump output objects
    #     for model_path in CFG.exp_dump_path.glob("*.pth"):
    #         if "seed" in str(model_path) or "fold" in str(model_path):
    #             continue

    #         # Rename model file
    #         model_file_name_dst = f"{model_path.stem}_fold{fold}.pth"
    #         model_path_dst = exp.ckpt_path / model_file_name_dst
    #         model_path.rename(model_path_dst)

    #     # Free mem.
    #     del (data_tr, data_val, train_loader, val_loader, model, optimizer, lr_skd, evaluator, trainer)
    #     _ = gc.collect()

    #     if CFG.one_fold_only:
    #         logger.info("Cross-validatoin stops at first fold!!!")
    #         break

    # np.save(CFG.exp_dump_pat / "oof.npy", oof)
else:
    file_path = DATA_PATH / "kaggle/input/hms-oof-demo/oof_seed0.npy"
    print(f"Checking file path: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    oof = np.load(file_path)
    
    
    
    

#         logger.info("Initializing DiffEEG Model for EEG Augmentation...")
#         train_steps = CFG.diffEEG_trainer["epochs"] * len(train_loader) # can be customized to a fixed number, however, it should reflect the dataset size.
#         train_steps = max(train_steps, 10000)
#         print('train_steps:',train_steps)
#         # **Initialize DiffEEG Model**
#         model = DiffEEG_Updated(config=CFG).to(CFG.device)

#         # **Define Diffusion Module**
#         diffusion_module = DiffEEGDiffusion(model, CFG, device=CFG.device)

#         # **Define Trainer**
#         trainer = DiffEEGTrainer(
#             logger = logger,
#             # wandb_logger = wandb_logger,
#             diffusion_module=diffusion_module,
#             dataloader_train=train_loader,  # Pass DataLoader
#             dataloader_val=val_loader,      
#             config=CFG,
#             device=CFG.device
#         )

#         # **Train the DiffEEG Model**
#         logger.info(f"Results will be saved in: {CFG.diffEEG_trainer['results_folder']}")
#         logger.info("Starting DiffEEG Training...")
#         trainer.train()

#         # **Save Final Model**
#         model_save_path = CFG.diffEEG_trainer["results_folder"] / f"diffEEG_final_fold{fold}.pth"
#         torch.save(model.state_dict(), model_save_path)
#         logger.info(f"Final model saved to {model_save_path}")

#         # **Run Final Evaluation**
#         logger.info("Running Final Validation...")
#         real_eeg, generated_eeg = trainer.generate_augmented_samples(n_samples = 100, class_label = 2)

#         # **Save Generated EEG Samples**
#         generated_save_path = CFG.exp_dump_path / f"generated_eegs_fold{fold}.npy"
#         np.save(generated_save_path, generated_eeg)
#         logger.info(f"Saved generated EEG samples to {generated_save_path}")

#     logger.info("Training Completed Successfully!")





# # ---- Settings ----
# checkpoint_path = Path("/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/models/DiffEEG_model_1000.pt" )  # ← Update path here
# output_dir = Path("/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/generated_data")  # Where to save .npy files
# output_dir.mkdir(parents=True, exist_ok=True)
# n_samples_per_class = 5

# # ---- Logger ----
# logger = _Logger(logging_file="/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/logs/generation.log").get_logger()
# logger.info("Starting EEG generation for each class...")

# # ---- Load Model ----
# logger.info("Loading model...")
# model = DiffEEG_Updated(config=CFG).to(CFG.device)
# checkpoint = torch.load(checkpoint_path, map_location=CFG.device)
# model.load_state_dict(checkpoint['model'])
# model.eval()

# # ---- Diffusion Wrapper ----
# diffusion_module = DiffEEGDiffusion(model=model, config=CFG, device=CFG.device)



# # ---- Generate for All Classes ----
# for cls_id in range(CFG.N_CLASSES):
#     logger.info(f"Generating {n_samples_per_class} EEGs for class {cls_id}")
#     eeg_data = generate_for_class(cls_id, n_samples_per_class, diffusion_module)

#     # Save
#     save_path = output_dir / f"generated_class_{cls_id}.npy"
#     np.save(save_path, eeg_data)
#     logger.info(f"Saved generated EEGs for class {cls_id} to {save_path}")

# logger.info("Done generating EEGs for all classes.")
 
 
 
 
# DATA_PATH = CFG.DATA_PATH  
# # Load constants
# TGT_VOTE_COLS = CFG.TGT_VOTE_COLS
# EEG_WLEN = CFG.EEG_WLEN
# EEG_FREQ = CFG.EEG_FREQ 
# TGT_COL = CFG.TGT_COL   
# N_CLASSES = CFG.N_CLASSES
# EEG_PTS = CFG.EEG_PTS

# Load CSV
# train_csv_path = DATA_PATH / "train.csv"
# train = pd.read_csv(train_csv_path)

# # Path to saved cropped EEG signals
# eeg_file_path = DATA_PATH / "kaggle" / "input" / "brain-eegs" / "eegs.npy"
# assert eeg_file_path.exists(), f"Missing EEG file: {eeg_file_path}"
# all_eegs = np.load(eeg_file_path, allow_pickle=True).item()

# # Ensure integrity
# uniq_eeg_ids = train["eeg_id"].unique()
# assert len(all_eegs) == len(uniq_eeg_ids)

# # Reconstruct metadata (normalized label dataframe)
# df_tmp = train.groupby("eeg_id")[["patient_id"]].agg("first")
# labels_tmp = train.groupby("eeg_id")[TGT_VOTE_COLS].agg("sum")
# for col in TGT_VOTE_COLS:
#     df_tmp[col] = labels_tmp[col].values

# # Normalize the vote columns to probabilities
# y_data = df_tmp[TGT_VOTE_COLS].values
# y_data = y_data / y_data.sum(axis=1, keepdims=True)
# df_tmp[TGT_VOTE_COLS] = y_data

# # Add hard target (expert_consensus)
# df_tmp[TGT_COL] = train.groupby("eeg_id")[["expert_consensus"]].agg("first")

# # Final metadata DataFrame
# train_df = df_tmp.reset_index()  
    
# gen_data_dir = Path("/data2/users/koushani/HMS_data/kaggle/working/0401-18-02-56/generated_data")  # Where to save .npy files   

# real_dataset = EEGDataset({"meta": data_tr, "eeg": all_eegs}, "train", **CFG.dataset)
        




# # Reload dataloader
# augmented_train_loader = DataLoader(
#     EEGDataset({"meta": augmented_df, "eeg": augmented_eegs}, "train", **CFG.dataset),
#     shuffle=True,
#     batch_size=CFG.diffEEG_trainer["batch_size"],
#     num_workers=CFG.diffEEG_trainer["num_workers"]
# )


    #     # Build model
    #     logger.info(f"Build model...")
    #     model = DilatedInceptionWaveNet()
    #     model.to(CFG.device)

    #     # Build criterion
    #     loss_fn = KLDivWithLogitsLoss()

    #     # Build solvers
    #     optimizer = torch.optim.Adam(model.parameters(), lr=CFG.trainer["lr"])
    #     num_training_steps = (
    #         math.ceil(
    #             len(train_loader.dataset)
    #             / (CFG.trainer["dataloader"]["batch_size"] * CFG.trainer["grad_accum_steps"])
    #         )
    #         * CFG.trainer["epochs"]
    #     )
    #     lr_skd = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    #     # Build evaluator
    #     evaluator = Evaluator(metric_names=["kldiv"])

    #     # Build trainer
    #     trainer: _BaseTrainer = None
    #     trainer = MainTrainer(
    #         logger=logger,
    #         trainer_cfg=CFG.trainer,
    #         model=model,
    #         loss_fn=loss_fn,
    #         optimizer=optimizer,
    #         lr_skd=lr_skd,
    #         ckpt_path=CFG.exp_dump_path,
    #         evaluator=evaluator,
    #         scaler=None,
    #         train_loader=train_loader,
    #         eval_loader=val_loader,
    #         use_wandb=False
    #     )

    #     # Run main training and evaluation for one fold
    #     y_preds = trainer.train_eval(fold)
    #     oof[val_idx, :] = y_preds["val"]

    #     # Dump output objects
    #     for model_path in CFG.exp_dump_path.glob("*.pth"):
    #         if "seed" in str(model_path) or "fold" in str(model_path):
    #             continue

    #         # Rename model file
    #         model_file_name_dst = f"{model_path.stem}_fold{fold}.pth"
    #         model_path_dst = CFG.exp_dump_path / model_file_name_dst
    #         model_path.rename(model_path_dst)

    #     # Free mem.
    #     del (data_tr, data_val, train_loader, val_loader, model, optimizer, lr_skd, evaluator, trainer)
    #     _ = gc.collect()

    #     if CFG.one_fold_only:
    #         logger.info("Cross-validation stops at first fold!!!")
    #         break

    # np.save(CFG.exp_dump_path / "oof.npy", oof)
# else:
#     file_path = DATA_PATH / "kaggle/input/hms-oof-demo/oof_seed0.npy"
#     print(f"Checking file path: {file_path}")
#     print(f"File exists: {os.path.exists(file_path)}")
#     oof = np.load(file_path)