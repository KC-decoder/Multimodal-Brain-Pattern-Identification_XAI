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
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
import ipywidgets as widgets

from utils.training_utils import KLDivWithLogitsLoss, Evaluator, _ModelCheckpoint, _BaseTrainer, MainTrainer
from models.models import DilatedInceptionWaveNet
from data.dataset import EEGDataset
from utils.cfg_utils import CFG, _Logger, _seed_everything
from models.diffusion_model import DiffEEG

    
    
    
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



def _get_eeg_window(file: Path) -> np.ndarray:
    """Return cropped EEG window.

    Default setting is to return the middle 50-sec window.

    Args:
        file: EEG file path
        test: if True, there's no need to truncate EEGs

    Returns:
        eeg_win: cropped EEG window 
    """
    eeg = pd.read_parquet(file, columns=CFG.feats)
    n_pts = len(eeg)
    offset = (n_pts - EEG_PTS) // 2
    eeg = eeg.iloc[offset:offset + EEG_PTS]
    
    eeg_win = np.zeros((EEG_PTS, len(CFG.feats)))
    for j, col in enumerate(CFG.feats):
        if CFG.cast_eegs:
            eeg_raw = eeg[col].values.astype("float32")
        else:
            eeg_raw = eeg[col].values 

        # Fill missing values
        mean = np.nanmean(eeg_raw)
        if np.isnan(eeg_raw).mean() < 1:
            eeg_raw = np.nan_to_num(eeg_raw, nan=mean)
        else: 
            # All missing
            eeg_raw[:] = 0
        eeg_win[:, j] = eeg_raw 
        
    return eeg_win 

DATA_PATH = CFG.DATA_PATH
train = pd.read_csv(DATA_PATH / "train.csv")
logger.info(f"Train data shape | {train.shape}")


# Define paths
eeg_file_path = DATA_PATH/"kaggle" /"input"/ "brain-eegs" / "eegs.npy"

# Ensure directories exist
eeg_file_path.parent.mkdir(parents=True, exist_ok=True)

# Unique EEG IDs
uniq_eeg_ids = train["eeg_id"].unique()
n_uniq_eeg_ids = len(uniq_eeg_ids)

# # Check if file exists
# if not eeg_file_path.exists() or CFG.gen_eegs:
#     logger.info("Generate cropped EEGs...")
#     all_eegs = {}
    
#     for i, eeg_id in tqdm(enumerate(uniq_eeg_ids), total=n_uniq_eeg_ids):
#         eeg_win = _get_eeg_window(DATA_PATH / "train_eegs" / f"{eeg_id}.parquet")
#         all_eegs[eeg_id] = eeg_win
    
#     # Save the new file
#     np.save(eeg_file_path, all_eegs)
#     logger.info(f"Saved EEGs to {eeg_file_path}")
# else:
logger.info("Load cropped EEGs...")
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

    
if CFG.add_augmentation:
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
    
    diffusion_trainer = 
        
        
        
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
            model_path_dst = CFG.exp_dump_path / model_file_name_dst
            model_path.rename(model_path_dst)

        # Free mem.
        del (data_tr, data_val, train_loader, val_loader, model, optimizer, lr_skd, evaluator, trainer)
        _ = gc.collect()

        if CFG.one_fold_only:
            logger.info("Cross-validation stops at first fold!!!")
            break

    np.save(CFG.exp_dump_path / "oof.npy", oof)
else:
    file_path = DATA_PATH / "kaggle/input/hms-oof-demo/oof_seed0.npy"
    print(f"Checking file path: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    oof = np.load(file_path)