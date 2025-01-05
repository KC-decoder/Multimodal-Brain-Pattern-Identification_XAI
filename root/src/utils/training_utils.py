import torch
import math
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
from models.models import  EEGNet, DeepConvNet
from training.training import train_and_validate_eeg
from utils.config_loader import load_config
from utils.logger_utils import setup_logger
from contextlib import redirect_stdout
from itertools import product
import torch.nn.init as init


from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path
from utils.cfg_utils import _Logger 
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from abc import ABC, abstractmethod
import io
from utils.cfg_utils import CFG
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch.nn.functional as F
import gc


import sys
from io import StringIO
from itertools import product
import numpy as np
import logging

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
    
    
def initialize_kaiming_weights(model):
    """Applies custom weight initialization to the given model."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            init.constant_(m.bias, 0)
            
            
# Define the warm-up and cosine annealing schedule
def warmup_cosine_schedule(epoch, warmup_epochs, total_epochs, initial_lr, peak_lr, min_lr):
    if epoch < warmup_epochs:
        # Linear warm-up phase from initial_lr to peak_lr
        return initial_lr + (peak_lr - initial_lr) * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing phase from peak_lr to min_lr
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_scale = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cosine_scale
    
    
    
    
    
class KLDivWithLogitsLoss(nn.KLDivLoss):
    """Kullback-Leibler divergence loss with logits as input."""

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = F.log_softmax(y_pred,  dim=1)
        kldiv_loss = super().forward(y_pred, y_true)

        return kldiv_loss
    
    
class Evaluator(object):
    """Custom evaluator.

    Args:
        metric_names: evaluation metrics
    """

    eval_metrics: Dict[str, Callable[..., float]] = {}
    EPS: float = 1e-6

    def __init__(self, metric_names: List[str]) -> None:
        self.metric_names = metric_names

        self._build()

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        scaler: Optional[object] = None,
    ) -> Dict[str, float]:
        """Run evaluation using pre-specified metrics.

        Args:
            y_true: ground truth
            y_pred: prediction
            scaler: scaling object

        Returns:
            eval_result: evaluation performance report
        """
        if scaler is not None:
            # Do inverse transformation to rescale y values
            y_pred, y_true = self._rescale_y(y_pred, y_true, scaler)

        eval_result = {}
        for metric_name, metric in self.eval_metrics.items():
            eval_result[metric_name] = metric(y_pred, y_true).item()

        return eval_result

    def _build(self) -> None:
        """Build evaluation metric instances."""
        for metric_name in self.metric_names:
            if metric_name == "kldiv":
                self.eval_metrics[metric_name] = KLDivWithLogitsLoss() 
            elif metric_name == "ce":
                self.eval_metrics[metric_name] = nn.CrossEntropyLoss()

    def _rescale_y(self, y_pred: Tensor, y_true: Tensor, scaler: Any) -> Tuple[Tensor, Tensor]:
        """Rescale y to the original scale.

        Args:
            y_pred: prediction
            y_true: ground truth
            scaler: scaling object

        Returns:
            y_pred: rescaled prediction
            y_true: rescaled ground truth
        """
        # Do inverse transform...

        return y_pred, y_true
    
    
    
    
class _ModelCheckpoint(object):
    """Model checkpooint.

    Args:
        ckpt_path: path to save model checkpoint
        ckpt_metric: quantity to monitor during training process
        ckpt_mode: determine the direction of metric improvement
        best_ckpt_mid: model identifier of the probably best checkpoint
            used to do the final evaluation
    """

    def __init__(self, ckpt_path: Path, ckpt_metric: str, ckpt_mode: str, best_ckpt_mid: str) -> None:
        self.ckpt_path = ckpt_path
        self.ckpt_metric = ckpt_metric
        self.ckpt_mode = ckpt_mode
        self.best_ckpt_mid = best_ckpt_mid

        # Specify checkpoint direction
        self.ckpt_dir = -1 if ckpt_mode == "max" else 1

        # Initialize checkpoint status
        self.best_val_score = 1e18
        self.best_epoch = 0

    def step(
        self, epoch: int, model: nn.Module, val_loss: float, val_result: Dict[str, float], last_epoch: bool = False
    ) -> None:
        """Update checkpoint status for the current epoch.

        Args:
            epoch: current epoch
            model: current model instance
            val_loss: validation loss
            val_result: evaluation result on validation set
            last_epoch: if True, current epoch is the last one
        """
        val_score = val_loss if self.ckpt_metric is None else val_result[self.ckpt_metric]
        val_score = val_score * self.ckpt_dir
        if val_score < self.best_val_score:  # type: ignore
            logging.info(f"Validation performance improves at epoch {epoch}!!")
            self.best_val_score = val_score
            self.best_epoch = epoch

            # Save model checkpoint
            mid = "loss" if self.ckpt_metric is None else self.ckpt_metric
            self._save_ckpt(model, mid)

        if last_epoch:
            self._save_ckpt(model, "last")

    def save_ckpt(self, model: nn.Module, mid: Optional[str] = None) -> None:
        """Save the checkpoint.

        Args:
            model: current model instance
            mid: model identifer
        """
        self._save_ckpt(model, mid)

    def load_best_ckpt(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Load and return the best model checkpoint for final evaluation.

        Args:
            model: current model instance
                *Note: Model weights are overrided by the best checkpoint.
            device: device of the model instance

        Returns:
            best_model: best model checkpoint
        """
        model = self._load_ckpt(model, device, self.best_ckpt_mid)

        return model

    def _save_ckpt(self, model: nn.Module, mid: Optional[str] = None) -> None:
        """Save the model checkpoint.

        Args:
            model: current model instance
            mid: model identifer
        """
        model_file = "model.pth" if mid is None else f"model-{mid}.pth"
        torch.save(model.state_dict(), os.path.join(self.ckpt_path, model_file))

    def _load_ckpt(self, model: nn.Module, device: torch.device, mid: str = "last") -> nn.Module:
        """Load the model checkpoint.

        Args:
            model: current model instance
                *Note: Model weights are overrided by the best checkpoint.
            device: device of the model instance
            mid: model identifier

        Returns:
            model: model instance with the loaded weights
        """
        model_file = f"model-{mid}.pth"
        model.load_state_dict(torch.load(os.path.join(self.ckpt_path, model_file), map_location=device))

        return model
    
    
    
    
class _BaseTrainer:
    """Base class for all customized trainers.

    Args:
        logger: message logger
        trainer_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_skd: learning rate scheduler
        ckpt_path: path to save model checkpoints
        es: early stopping tracker
        evaluator: task-specific evaluator
        use_wandb: if True, training and evaluation processes are
            tracked with wandb
    """

    train_loader: DataLoader  # Tmp. workaround
    eval_loader: DataLoader  # Tmp. workaround

    def __init__(
        self,
        logger: _Logger,
        trainer_cfg: Dict[str, Any],
        model: nn.Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
        ckpt_path: Path,
        evaluator: Evaluator,
        use_wandb: bool = False,
    ):
        self.logger = logger
        self.trainer_cfg = trainer_cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_skd = lr_skd
        self.ckpt_path = ckpt_path
        self.evaluator = evaluator
        self.use_wandb = use_wandb

        self.device = CFG.device
        self.epochs = trainer_cfg["epochs"]
        self.use_amp = trainer_cfg["use_amp"]
        self.grad_accum_steps = trainer_cfg["grad_accum_steps"]
        self.step_per_batch = trainer_cfg["step_per_batch"]

        # Debug options
        self.one_batch_only = trainer_cfg["one_batch_only"]

        # Model checkpoint
        self.model_ckpt = _ModelCheckpoint(ckpt_path, **trainer_cfg["model_ckpt"])

        # Early stopping
        if trainer_cfg["es"]["patience"] != 0:
            self.logger.info("Please disable early stop!")
#             self.es = EarlyStopping(**trainer_cfg["es"])
        else:
            self.es = None

        self._iter = 0
        self._track_best_model = True  # (Deprecated)

    def train_eval(self, proc_id: int) -> Dict[str, np.ndarray]:
        """Run training and evaluation processes.

        Args:
            proc_id: identifier of the current process
        """
        self.logger.info("Start training and evaluation processes...")
        for epoch in range(self.epochs):
            self.epoch = epoch  # For interior use
            train_loss = self._train_epoch()
            val_loss, val_result, _ = self._eval_epoch()

            # Adjust learning rate
            if self.lr_skd is not None and not self.step_per_batch:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(val_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result (by epoch)
            self._log_proc(epoch, train_loss, val_loss, val_result)

            # Record the best checkpoint
            self.model_ckpt.step(
                epoch, self.model, val_loss, val_result, last_epoch=False if epoch != self.epochs - 1 else True
            )

            # Check early stopping is triggered or not
            if self.es is not None:
                self.es.step(val_loss)
                if self.es.stop:
                    self.logger.info(f"Early stopping is triggered at epoch {epoch}, training process is halted.")
                    break
        if self.use_wandb:
            wandb.log({"best_epoch": self.model_ckpt.best_epoch + 1})  # `epoch` starts from 0

        # Run final evaluation
        final_prf_report, y_preds = self._run_final_eval()
        self._log_best_prf(final_prf_report)

        return y_preds

    @abstractmethod
    def _train_epoch(self) -> Union[float, Dict[str, float]]:
        """Run training process for one epoch.

        Returns:
            train_loss_avg: average training loss over batches
                *Note: If MTL is used, returned object will be dict
                    containing losses of sub-tasks and the total loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(self, return_output: bool = False) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Args:
            return_output: whether to return prediction

        Returns:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: prediction
        """
        raise NotImplementedError

    def _log_proc(
        self,
        epoch: int,
        train_loss: Union[float, Dict[str, float]],
        val_loss: Optional[float] = None,
        val_result: Optional[Dict[str, float]] = None,
        proc_id: Optional[str] = None,
    ) -> None:
        """Log message of training process.

        Args:
            epoch: current epoch number
            train_loss: training loss
            val_loss: validation loss
            val_result: evaluation performance report
            proc_id: identifier of the current process
        """
        proc_msg = [f"Epoch{epoch} [{epoch+1}/{self.epochs}]"]

        # Construct training loss message
        if isinstance(train_loss, float):
            proc_msg.append(f"Training loss {train_loss:.4f}")
        else:
            for loss_k, loss_v in train_loss.items():
                loss_name = loss_k.split("_")[0].capitalize()
                proc_msg.append(f"{loss_name} loss {round(loss_v, 4)}")

        # Construct eval prf message
        if val_loss is not None:
            proc_msg.append(f"Validation loss {val_loss:.4f}")
        if val_result is not None:
            for metric, score in val_result.items():
                proc_msg.append(f"{metric.upper()} {round(score, 4)}")

        proc_msg = " | ".join(proc_msg)
        self.logger.info(proc_msg)

        if self.use_wandb:
            # Process loss dict and log
            log_dict = train_loss if isinstance(train_loss, dict) else {"train_loss": train_loss}
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            if val_result is not None:
                for metric, score in val_result.items():
                    log_dict[metric] = score

            if proc_id is not None:
                log_dict = {f"{k}_{proc_id}": v for k, v in log_dict.items()}

            wandb.log(log_dict)

    def _run_final_eval(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
        """Run final evaluation process with designated model checkpoint.

        Returns:
            final_prf_report: performance report of final evaluation
            y_preds: prediction on different datasets
        """
        # Load the best model checkpoint
        self.model = self.model_ckpt.load_best_ckpt(self.model, self.device)

        # Reconstruct dataloaders
        self._disable_shuffle()
        val_loader = self.eval_loader

        final_prf_report, y_preds = {}, {}
        for data_split, dataloader in {
            # "train": self.train_loader,
            "val": val_loader,
        }.items():
            self.eval_loader = dataloader
            _, eval_result, y_pred = self._eval_epoch(return_output=True)
            final_prf_report[data_split] = eval_result
            y_preds[data_split] = y_pred.numpy()

        return final_prf_report, y_preds

    def _disable_shuffle(self) -> None:
        """Disable shuffle in train dataloader for final evaluation."""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,  # Reset shuffle to False
            num_workers=self.train_loader.num_workers,
            collate_fn=self.train_loader.collate_fn,
        )

    def _log_best_prf(self, prf_report: Dict[str, Any]) -> None:
        """Log performance evaluated with the best model checkpoint.

        Args:
            prf_report: performance report
        """
        self.logger.info(">>>>> Performance Report - Best Ckpt <<<<<")
        self.logger.info(json.dumps(prf_report, indent=4))

        if self.use_wandb:
            wandb.log(prf_report)
            
            
            
            
class MainTrainer(_BaseTrainer):
    """Main trainer.

    Args:
        logger: message logger
        trainer_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        scaler: scaling object
        train_loader: training data loader
        eval_loader: validation data loader
        use_wandb: if True, training and evaluation processes are
            tracked with wandb
    """

    def __init__(
        self,
        logger: _Logger,
        trainer_cfg: Dict[str, Any],
        model: nn.Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
        ckpt_path: Path,
        evaluator: Evaluator,
        scaler: Any,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        use_wandb: bool = False,
    ):
        super(MainTrainer, self).__init__(
            logger,
            trainer_cfg,
            model,
            loss_fn,
            optimizer,
            lr_skd,
            ckpt_path,
            evaluator,
            use_wandb,
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader
        self.scaler = scaler

        self.loss_name = self.loss_fn.__class__.__name__

        # Mixed precision training
        self.grad_scaler = GradScaler(enabled=self.use_amp)

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Returns:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            if i % self.grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            # Retrieve batched raw data
            inputs = {}
            for k, v in batch_data.items():
                if k != "y":
                    inputs[k] = v.to(self.device)
                else:
                    y = v.to(self.device)

            with autocast(enabled=self.use_amp):
                # Forward pass and derive loss
                output = self.model(inputs)
                loss = self.loss_fn(output, y)
            train_loss_total += loss.item()
            loss = loss / self.grad_accum_steps

            # Backpropagation
            self.grad_scaler.scale(loss).backward()
            if (i + 1) % self.grad_accum_steps == 0:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                if self.step_per_batch:
                    self.lr_skd.step()

            self._iter += 1

            # Free mem.
            del inputs, y, output
            _ = gc.collect()

            if self.one_batch_only:
                break

        train_loss_avg = train_loss_total / len(self.train_loader)

        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(
        self,
        return_output: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Args:
            return_output: whether to return prediction

        Returns:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: prediction
        """
        eval_loss_total = 0
        y_true, y_pred = [], []

        self.model.eval()
        for i, batch_data in enumerate(self.eval_loader):
            # Retrieve batched raw data
            inputs = {}
            for k, v in batch_data.items():
                if k != "y":
                    inputs[k] = v.to(self.device)
                else:
                    y = v.to(self.device)

            # Forward pass
            output = self.model(inputs)

            # Derive loss
            loss = self.loss_fn(output, y)
            eval_loss_total += loss.item()

            # Record batched output
            y_true.append(y.detach().cpu())
            y_pred.append(output.detach().cpu())

            del inputs, y, output
            _ = gc.collect()

        eval_loss_avg = eval_loss_total / len(self.eval_loader)

        # Run evaluation with the specified evaluation metrics
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        eval_result = self.evaluator.evaluate(y_true, y_pred, self.scaler)

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None





class DiffusionAugmentationTrainer(_BaseTrainer):
    """Diffusion Model trainer.

    Args:
        logger: message logger
        trainer_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        scaler: scaling object
        train_loader: training data loader
        eval_loader: validation data loader
        use_wandb: if True, training and evaluation processes are
            tracked with wandb
    """

    def __init__(
        self,
        logger: _Logger,
        trainer_cfg: Dict[str, Any],
        model: nn.Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
        ckpt_path: Path,
        evaluator: Evaluator,
        scaler: Any,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        use_wandb: bool = False,
    ):
        super(DiffusionAugmentationTrainer, self).__init__(
            logger,
            trainer_cfg,
            model,
            loss_fn,
            optimizer,
            lr_skd,
            ckpt_path,
            evaluator,
            use_wandb,
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader
        self.scaler = scaler

        self.loss_name = self.loss_fn.__class__.__name__

        # Mixed precision training
        self.grad_scaler = GradScaler(enabled=self.use_amp)

    def _train_epoch(self) -> float:
        """Run training process for one epoch.
        Returns:
            train_loss_avg: average training loss over batches
        """
        
        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(
        self,
        return_output: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Args:
            return_output: Similarity between Generated 

        Returns:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: prediction
        """

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None