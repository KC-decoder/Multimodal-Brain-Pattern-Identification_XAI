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


import wandb
from collections import deque , defaultdict





class CFG:
    gpu = 2
    train_models = True
    add_augmentation = True
    seed = 42
    DATA_PATH = Path("/data2/users/koushani/HMS_data")
    
    exp_id = datetime.now().strftime("%m%d-%H-%M-%S")
    # Define experiment ID and path
    exp_dump_path = Path(DATA_PATH/"kaggle"/"working"/exp_id)
    # print(exp_dump_path) 
    # Create the directory
    exp_dump_path.mkdir(parents=True, exist_ok=True)
    
    
    # Check if specified GPU is available, else default to CPU
    if torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{gpu}")
            # Test if the specified GPU index is valid
            _ = torch.cuda.get_device_name(device)
        except AssertionError:
            print(f"GPU {gpu} is not available. Falling back to GPU 0.")
            device = torch.device("cuda:0")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    
    EEG_FREQ = 200  # Hz
    EEG_WLEN = 50  # sec
    EEG_PTS = int(EEG_FREQ * EEG_WLEN)
    N_CLASSES = 6
    TGT_VOTE_COLS = [
        "seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote",
        "grda_vote", "other_vote"
    ]
    TGT_COL = "target"
    
        
    # == Data ==
    gen_eegs = False
    # Chris' 8 channels
    feats = [
        "Fp1", "T3", "C3", "O1",
        "Fp2", "C4", "T4", "O2"
    ]
    # Original EEG channels (Except EKG)
    channel_feats =  [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz", 
    "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"
    ]
    cast_eegs = True
    dataset = {
        "eeg": {
            "n_feats": 19,
            "apply_chris_magic_ch8": False,
            "normalize": True,
            "apply_butter_lowpass_filter": True,
            "apply_mu_law_encoding": False,
            "downsample": 5
        }
    }

    # == Trainer ==
    trainer = {
        "epochs": 50,
        "lr": 1e-3,
        "dataloader": {
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2
        },
        "use_amp": True,
        "grad_accum_steps": 1,
        "model_ckpt": {
            "ckpt_metric": "kldiv",
            "ckpt_mode": "min",
            "best_ckpt_mid": "last"
        },
        "es": {"patience": 0},
        "step_per_batch": True,
        "one_batch_only": False
    }
    
    # == Debug ==
    one_fold_only = F
    
    
    # == WandB Logging ==
    
    use_wandb = True
    
    # == DiffEEG Trainer ==
    
    diffEEG_trainer = {
         
        "epochs" : 10 ,
        "n_channels" : 19 ,
        "input_length" : 2000 ,
        "n_classes" : 6 ,
        "hidden_channels" : 32 ,
        "n_residual_layers" : 16 ,
        "n_heads" : 12  ,
        "dropout" : 0.1 ,
        "n_diffusion_steps" : 1000 ,
        
        
        "ema_decay" : 0.995 ,
        "step_start_ema" : 20,
        "update_ema_every": 10, # 10
        "save_and_sample_every" : 200 , # 200
        "gradient_accumulate_every" : 50,
        "evaluate_every": 50,
            
            
            
        "lr" : 1e-5,
        "batch_size": 64 ,
        "shuffle": True,
        "num_workers": 50,
        "results_folder" : exp_dump_path,
        
        
        # STFT parameters
        "stft_n_fft" : 64  , # 2 seconds window
        "stft_hop_length" : 32 ,
        "stft_window" : 'hann'
        
        
        
  }
    





class WandbLogger:
    """
    Custom logger for tracking experiments using Weights & Biases (wandb).

    Features:
    - Logs training loss per step
    - Logs evaluation metrics per evaluation step
    - Logs custom plots (loss vs step, evaluation metrics)
    - Saves model checkpoints to wandb
    """
    def __init__(self, project_name: str, config: dict, exp_path: Path):
        self.exp_id = datetime.now().strftime("%m%d-%H-%M-%S")
        self.exp_path = exp_path  # Experiment directory where logs & models are stored

        # Initialize wandb
        wandb.init(
            project=project_name,
            name=f"run_{self.exp_id}",
            dir=str(exp_path),
            config=config
        )

        self.step = 0  # Training step counter
        self.eval_step = 0  # Evaluation step counter
        self.loss_history = []  # Track loss per step
        self.metric_history = defaultdict(list)  # Store evaluation metric history

    def log_loss(self, loss: float):
        """Log training loss per step."""
        self.step += 1
        self.loss_history.append(loss)  # Store loss for plotting
        wandb.log({"train_loss": loss, "step": self.step})

    def log_evaluation(self, metrics: dict):
        """Log evaluation metrics (e.g., MMD, Fréchet distance, Pearson correlation)."""
        self.eval_step += 1

        # Store metrics in history for future plotting
        for metric_name, value in metrics.items():
            self.metric_history[metric_name].append(value)

        wandb.log(metrics | {"eval_step": self.eval_step})  # Merge dictionary with eval_step

    def plot_loss(self):
        """Explicitly log loss vs step plot to WandB."""
        if len(self.loss_history) > 1:
            wandb.log({"Loss vs Step": wandb.plot.line_series(
                xs=list(range(1, len(self.loss_history) + 1)),
                ys=[self.loss_history],  # Ensure it's a list of lists
                keys=["Training Loss"],
                title="Loss vs Step",
                xname="Step"
            )})

    def plot_metrics(self):
        """Log all stored evaluation metrics as line plots."""
        if self.metric_history:
            wandb.log({
                f"{metric_name} vs Evaluation Step": wandb.plot.line_series(
                    xs=list(range(len(values))),
                    ys=[values],  # Convert into list of lists for multiple metrics
                    keys=[metric_name],
                    title=f"{metric_name} vs Evaluation Step",
                    xname="Evaluation Step"
                )
                for metric_name, values in self.metric_history.items() if len(values) > 1
            })

    def save_model(self, model_path: Path):
        """Save model checkpoint to wandb."""
        wandb.save(str(model_path))

    def finish(self):
        """Finish wandb session."""
        wandb.finish()





    
    
class _Logger:
    """Customized logger.

    Args:
        logging_level: lowest-severity log message the logger handles
        logging_file: file stream for logging
            *Note: If `logging_file` isn't specified, message is only
                logged to system standard output.
    """

    _logger: logging.Logger = None

    def __init__(
        self,
        logging_level: str = "INFO",
        logging_file: Optional[Path] = None,
        wandb_logger: Optional[WandbLogger] = None,
    ):
        self.logging_level = logging_level
        self.logging_file = logging_file
        self.wandb_logger = wandb_logger  # Link WandbLogger instance

        self._build_logger()

    def get_logger(self) -> logging.Logger:
        """Return customized logger."""
        return self._logger

    def _build_logger(self) -> None:
        """Build logger."""
        self._logger = logging.getLogger()
        self._logger.setLevel(self._get_level())
        self._add_handler()

    def _get_level(self) -> int:
        """Return lowest severity of the events the logger handles.

        Returns:
            level: severity of the events
        """
        level = 0

        if self.logging_level == "DEBUG":
            level = logging.DEBUG
        elif self.logging_level == "INFO":
            level = logging.INFO
        elif self.logging_level == "WARNING":
            level = logging.WARNING
        elif self.logging_level == "ERROR":
            level = logging.ERROR
        elif self.logging_level == "CRITICAL":
            level = logging.CRITICAL

        return level

    def _add_handler(self) -> None:
        """Add stream and file (optional) handlers to logger."""
        s_handler = logging.StreamHandler(sys.stdout)
        self._logger.addHandler(s_handler)

        if self.logging_file is not None:
            f_handler = logging.FileHandler(self.logging_file, mode="a")
            self._logger.addHandler(f_handler)
            
    def log_wandb(self, msg: str):
        """Logs a message to both console and wandb."""
        if self.wandb_logger:
            self.wandb_logger.log_loss(msg)

            
def _seed_everything(seed: int) -> None:
    """Seed current experiment to guarantee reproducibility.

    Args:
        seed: manually specified seed number
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running with cudnn backend
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True 
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    
EEG_PTS = CFG.EEG_PTS  
    
def _get_eeg_window(file: Path, use_magic_8: bool = True) -> np.ndarray:
    """
    Return cropped EEG window with optional channel set (Chris' magic 8 or full 19).

    Args:
        file: Path to the EEG `.parquet` file
        use_magic_8: If True, use 8 channels from CFG.feats, else use 19 from CFG.channel_feats

    Returns:
        eeg_win: Cropped EEG window of shape (EEG_PTS, num_channels)
    """
    use_magic_8 = CFG.dataset["eeg"]["apply_chris_magic_ch8"]
    channel_list = CFG.feats if use_magic_8 else CFG.channel_feats

    eeg = pd.read_parquet(file, columns=channel_list)
    n_pts = len(eeg)
    offset = (n_pts - EEG_PTS) // 2
    eeg = eeg.iloc[offset:offset + EEG_PTS]

    eeg_win = np.zeros((EEG_PTS, len(channel_list)), dtype="float32")
    for j, col in enumerate(channel_list):
        if CFG.cast_eegs:
            eeg_raw = eeg[col].values.astype("float32")
        else:
            eeg_raw = eeg[col].values

        # Fill missing values
        mean = np.nanmean(eeg_raw)
        if np.isnan(eeg_raw).mean() < 1:
            eeg_raw = np.nan_to_num(eeg_raw, nan=mean)
        else:
            eeg_raw[:] = 0

        eeg_win[:, j] = eeg_raw

    return eeg_win
    
   