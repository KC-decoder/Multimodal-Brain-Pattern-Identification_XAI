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




class CFG:
    gpu = 1
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
    cast_eegs = True
    dataset = {
        "eeg": {
            "n_feats": 8,
            "apply_chris_magic_ch8": True,
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
            "batch_size": 32,
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
    one_fold_only = False
    
    
    
    
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
    ):
        self.logging_level = logging_level
        self.logging_file = logging_file

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
    
   