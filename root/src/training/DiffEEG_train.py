import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
from utils.cfg_utils import CFG, _Logger, WandbLogger
from functools import partial
import matplotlib.pyplot as plt
import random



import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import copy
from utils.DiffEEG_utils import compute_mmd, compute_frechet_distance, pearson_correlation, compute_stft, EMA, loss_backwards
from pathlib import Path
import copy

class DiffEEGTrainer:
    """
    Trainer for DiffEEG model using diffusion-based augmentation.

    Args:
        logger: message logger
        diffusion_module (DiffEEGDiffusion): Handles noise and denoising.
        dataloader_train (DataLoader): Training dataloader.
        dataloader_val (DataLoader): Validation dataloader.
        config (DiffEEGConfig): Configuration parameters.
        device (str): 'cuda' or 'cpu'.
    """
    def __init__(self, logger: _Logger, diffusion_module, dataloader_train, dataloader_val, config, fp16=False, device="cuda"):
        
        self.logger = logger
        self.diffusion = diffusion_module
        self.device = device
        self.config = config

        # **Enable Mixed Precision Optimization**
        self.scaler = torch.amp.GradScaler(enabled=fp16)
        self.fp16 = fp16  # Track FP16 usage

        # **Use AdamW instead of Adam (Lower Memory Use)**
        self.optimizer = optim.AdamW(self.diffusion.model.parameters(), lr=config.diffEEG_trainer["lr"])

        self.loss_fn = nn.MSELoss()
        self.epochs = self.config.diffEEG_trainer["epochs"]

        # **Use DataLoader with Fewer Workers**
        self.dataloader = dataloader_train  
        self.dataloader_val = dataloader_val  
        self.data_iter = iter(self.dataloader)  

        # **Enable EMA (Exponential Moving Average)**
        self.ema = EMA(beta=config.diffEEG_trainer["ema_decay"])
        self.ema_model = copy.deepcopy(self.diffusion.model)

        self.gradient_accumulate_every = self.config.diffEEG_trainer["gradient_accumulate_every"]
        self.save_and_sample_every = self.config.diffEEG_trainer["save_and_sample_every"]
        self.update_ema_every = self.config.diffEEG_trainer["update_ema_every"]
        self.n_diffusion_steps = self.config.diffEEG_trainer["n_diffusion_steps"]
        self.step_start_ema = self.config.diffEEG_trainer["step_start_ema"]
        self.evaluate_every = self.config.diffEEG_trainer["evaluate_every"]

        # Model Checkpointing
        self.results_folder = Path(config.diffEEG_trainer["results_folder"])
        self.results_folder.mkdir(exist_ok=True)
        self.step = 0

    def reset_parameters(self):
        """Resets EMA model parameters to match the model state."""
        self.ema_model.load_state_dict(self.diffusion.model.state_dict())

    def step_ema(self):
        """Updates EMA parameters if step count is sufficient."""
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return 
        self.ema.update_model_average(self.ema_model, self.diffusion.model)

    def save(self, itrs=None):
        """Saves the model and EMA model states."""
        data = {
            'step': self.step,
            'model': self.diffusion.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        filename = self.results_folder / f'DiffEEG_model_{itrs}.pt' if itrs else self.results_folder / 'DiffEEG_model.pt'
        torch.save(data, str(filename))

    def load(self, load_path):
        """Loads the model and EMA model from a checkpoint."""
        print(f"Loading model from: {load_path}")
        data = torch.load(load_path)
        self.step = data['step']
        self.diffusion.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        """Train DiffEEG model via noise prediction loss with gradient accumulation and EMA updates."""
        self.diffusion.model.train()
        backwards = partial(loss_backwards, self.fp16)

        self.train_num_steps = max(self.epochs * len(self.dataloader), 10000)
        acc_loss = 0
        pbar = tqdm(range(self.train_num_steps), desc="LOSS")

        for step in pbar:
            self.step = step
            u_loss = 0  

            # **Enable Gradient Accumulation**
            self.optimizer.zero_grad()
            
            for _ in range(self.gradient_accumulate_every):
                try:
                    batch = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.dataloader)
                    batch = next(self.data_iter)

                x0, class_labels = batch["x"].to(self.device, non_blocking=True), batch["y"].to(self.device, non_blocking=True)

                # Compute STFT spectrogram
                spectrogram = compute_stft(x0, self.config).to(self.device, non_blocking=True)
                # print(f"Shape of Spectrogram during training after stft: {spectrogram.shape}")
                

                # Select random diffusion step t
                t = torch.randint(0, self.n_diffusion_steps, (x0.shape[0],), device=self.device)

                # **Enable Mixed Precision (Autocast)**
                # Ensure inputs to loss function are in FP16 during autocast
                with torch.amp.autocast(device_type=str(self.device), enabled=self.fp16):
                    x_t, true_noise = self.diffusion.forward_diffusion(x0, t)
                    predicted_noise = self.diffusion.model(x_t, class_labels, t.unsqueeze(-1).float(), spectrogram)

                    # 🔹 Convert `true_noise` to the same dtype as `predicted_noise`
                    true_noise = true_noise.to(dtype=predicted_noise.dtype)  #  Ensure both have the same dtype

                    loss = self.loss_fn(predicted_noise, true_noise) / self.gradient_accumulate_every

                    # print(f"predicted_noise dtype: {predicted_noise.dtype}")
                    # print(f"true_noise dtype: {true_noise.dtype}")  # Should now be the same as predicted_noise
                    # print(f"loss dtype: {loss.dtype}")

                # **Use GradScaler for FP16 Stability**
                self.scaler.scale(loss.half()).backward()
                u_loss += loss.item()

            pbar.set_description(f"Loss = {u_loss / self.gradient_accumulate_every:.6f}")
            self.logger.info(f"Step: {self.step}, Loss: {u_loss / self.gradient_accumulate_every:.6f}")
            
            acc_loss += u_loss / self.gradient_accumulate_every

            # **Step Optimizer with Mixed Precision**
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)  # **Zero Grad Efficiently**

            # EMA Update
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Save Model
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                acc_loss /= self.save_and_sample_every
                self.logger.info(f"Mean LOSS during last sampling at {self.step}: {acc_loss}")
                acc_loss = 0
                self.save(self.step)
            # **Run Evaluation at Defined Intervals**
            
            if self.step % self.evaluate_every == 0 and self.step > 0:
                self.logger.info(f"\nRunning evaluation at step {self.step}...")
                self.evaluate()

        self.save(self.step + 1)
        self.logger.info("\nFinal evaluation after training...")
        self.evaluate()
        self.logger.info("Training completed")

    @torch.no_grad()
    

    def evaluate(self, sampling_fraction = 0.2):
        """Evaluate EEG quality using statistical similarity metrics and visualize samples."""
        self.logger.info("Evaluating model on validation dataset...")
        self.ema_model.eval()
        # But also:
        self.diffusion.model.eval()

        real_eeg_list, generated_eeg_list = [], []
        
        val_samples = list(self.dataloader_val)
        sample_size = max(1, int(len(val_samples) * sampling_fraction))
        sampled_batches = random.sample(val_samples, sample_size)

        with torch.no_grad():
            for batch in tqdm(sampled_batches, desc=f"Evaluating {sample_size} batches"):
                real_eeg, class_labels = batch["x"].to(self.device), batch["y"].to(self.device)

                # Compute STFT spectrogram for conditioning
                spectrogram = compute_stft(real_eeg, self.config).to(self.device)
                # print(f"Shape of Spectrogram during evaluation after stft: {spectrogram.shape}")
                # Generate EEG samples
                generated_eeg = self.diffusion.reverse_diffusion(real_eeg.shape[0], class_labels, spectrogram)
                # print("generated_eeg.shape:", generated_eeg.shape)
                if torch.isnan(generated_eeg).any() or torch.isinf(generated_eeg).any():
                    print("NaNs or Infs detected in generated EEG")
                    print("Spectrogram stats:", spectrogram.mean().item(), spectrogram.std().item())
                    print("Class labels:", class_labels)
                    print("Generated EEG stats:", generated_eeg.min().item(), generated_eeg.max().item())

                real_eeg_list.append(real_eeg)
                generated_eeg_list.append(generated_eeg)

        # Concatenate all validation samples
        real_eeg = torch.cat(real_eeg_list, dim=0)
        generated_eeg = torch.cat(generated_eeg_list, dim=0)

        # Compute Metrics
        # pearson_corr = pearson_correlation(real_eeg, generated_eeg)
        mmd_score = compute_mmd(real_eeg, generated_eeg)
        # frechet_dist = compute_frechet_distance(real_eeg, generated_eeg)
        

        # Print Results
        self.logger.info(f"\nEvaluation Metrics After Epoch {self.step}:")
        self.logger.info(f"MMD Score: {mmd_score:.5f} (Lower is better)")
        # self.logger.info(f"Fréchet Distance: {frechet_dist:.5f} (Lower is better)")
        # self.logger.info(f"Pearson Correlation: {pearson_corr:.5f} (Closer to 1 is better)")
        self.logger.info("----------------------------------------------------\n")

        # # Define Save Directory
        # save_dir = os.path.join(self.results_folder, "eval_plots")
        # os.makedirs(save_dir, exist_ok=True)

        # # Visualizing Real vs. Generated EEG Signals
        # fig, axs = plt.subplots(3, 2, figsize=(12, 8))

        # for i in range(3):  # Plot 3 sample EEG signals
        #     axs[i, 0].plot(real_eeg[i][0], label="Real EEG", color="blue")
        #     axs[i, 0].set_title(f"Real EEG - Sample {i+1}")
        #     axs[i, 0].legend()

        #     axs[i, 1].plot(generated_eeg[i][0], label="Generated EEG", color="red")
        #     axs[i, 1].set_title(f"Generated EEG - Sample {i+1}")
        #     axs[i, 1].legend()

        # plt.tight_layout()

        # #Save Image with Step Number in Filename**
        # save_path = os.path.join(save_dir, f"EEG_Eval_Step_{self.step}.png")
        # plt.savefig(save_path)
        # plt.close()

        # self.logger.info(f"Saved evaluation plot at: {save_path}")
        
        # # Log metrics to WandB
        # self.wandb_logger.log_evaluation({
        #     "MMD Score": mmd_score,
        #     "Fréchet Distance": frechet_dist,
        #     "Pearson Correlation": pearson_corr,
        #     "Step": self.step  # Include step info for tracking
        # })

        # # Explicitly call plot function to ensure visualization is logged
        # self.wandb_logger.plot_metrics()

    @torch.no_grad()
    def generate_augmented_samples(self, n_samples, class_label):
        """Generates augmentable EEG signals and evaluates quality."""
        
        generated_eeg = self.diffusion.reverse_diffusion(n_samples, class_label, spectrogram_prior)
        return generated_eeg
    
    
    
    
    
