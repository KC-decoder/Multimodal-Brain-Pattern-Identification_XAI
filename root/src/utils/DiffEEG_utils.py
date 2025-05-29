import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import stft
from utils.cfg_utils import CFG, _Logger, _seed_everything
from dataclasses import dataclass
import pandas as pd
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import List, Tuple
from tqdm import tqdm
from data.dataset import EEGDataset 
try:
    from apex import amp
 
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
    



# ------------------ COMPUTES SHORT TIME FOURIER TRANSFORM OF EEG SIGNALS ------------------ 
def compute_stft(signal, config):
    """Compute Short-Time Fourier Transform (STFT) for EEG signals.
    
    Args:
        signal (torch.Tensor): EEG signal of shape (batch_size, n_channels, length)
        config (DiffEEGConfig): Model configuration
        
    Returns:
        torch.Tensor: STFT spectrograms of shape (batch_size, n_channels, freq_bins, time_bins)
    """
    batch_size, n_channels, signal_length = signal.shape
    specs = []
    
    # Convert to NumPy only when necessary
    signal_np = signal.cpu().numpy()

    for batch_idx in range(batch_size):
        batch_specs = []
        for i in range(n_channels):  # Process each channel separately
            f, t, Zxx = stft(
                signal_np[batch_idx, i], 
                fs=200,  # Sampling rate
                nperseg=config.diffEEG_trainer["stft_n_fft"],
                noverlap=config.diffEEG_trainer["stft_hop_length"],
                window=config.diffEEG_trainer["stft_window"])

            # Convert magnitude to log scale
            S = np.log1p(np.abs(Zxx))  # Shape: (freq_bins, time_bins)

            # Interpolate along the time axis to match 2000 time steps
            time_interp = np.linspace(0, t[-1], 2000)  # New uniform time grid
            S_interp = np.zeros((S.shape[0], 2000))  # (freq_bins, 2000)

            for freq_idx in range(S.shape[0]):  # Interpolate for each frequency bin
                S_interp[freq_idx, :] = np.interp(time_interp, t, S[freq_idx, :])

            batch_specs.append(S_interp)  # Shape: (freq_bins, 2000)

        # Stack along channel dimension: (n_channels, freq_bins, 2000)
        specs.append(np.stack(batch_specs, axis=0))

    # Convert list to tensor
    specs_tensor = torch.from_numpy(np.array(specs)).float().to(signal.device)  # Shape: (batch_size, n_channels, freq_bins, 2000)

    # Normalize per batch (avoiding issues with min-max normalization)
    min_vals = specs_tensor.min(dim=-1, keepdim=True)[0]
    max_vals = specs_tensor.max(dim=-1, keepdim=True)[0]
    specs_tensor = (specs_tensor - min_vals) / (max_vals - min_vals + 1e-8)

    return specs_tensor  # Final shape: (batch_size, n_channels, freq_bins, time_bins=2000)




# -------------- EXPONENTIAL MOVING AVERAGE MODEL FOR STABILIZATION OF DIFFUSION TRAINING ------------------ 
class EMA:
    """Exponential Moving Average (EMA) for model weights."""
    def __init__(self, beta=0.995):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        """Updates EMA model parameters using exponential decay."""
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """Computes the exponential moving average of model parameters."""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    



# ------------------ DIFFUSION MODULE FOR DIFFEEG MODEL ------------------ 
class DiffEEGDiffusion(nn.Module):
    """
    Diffusion module for EEG signals using Gaussian noising and denoising.
    
    Args:
        model (nn.Module): The DiffEEG model (denoiser).
        config (DiffEEGConfig): Configuration containing diffusion parameters.
    """
    def __init__(self, model, config: CFG, device="cuda"):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.num_channels = config.diffEEG_trainer["n_channels"]
        self.num_timesteps = config.diffEEG_trainer["input_length"]

        betas = np.linspace(1e-4, 0.02, self.num_timesteps, dtype=np.float32)
        self.beta_schedule = torch.tensor(betas, dtype=torch.float32, device=self.device)
        self.noise_scale = torch.sqrt(self.beta_schedule)
        

        # Create a noise schedule (α_t, β_t) for forward diffusion
        self.alpha_t, self.beta_t = self._get_noise_schedule(config.diffEEG_trainer["n_diffusion_steps"])

    def _get_noise_schedule(self, timesteps, schedule="cosine"):
        """Generates noise schedule (α_t, β_t) for diffusion."""
        if schedule == "cosine":
            s = 0.008
            f_t = np.cos((np.linspace(0, 1, timesteps) + s) / (1 + s) * np.pi / 2) ** 2
            alpha_t = f_t / f_t[0]
            beta_t = 1 - (alpha_t / alpha_t[0])
        else:
            beta_t = np.linspace(0.0001, 0.02, timesteps)
            alpha_t = np.cumprod(1 - beta_t)
        return torch.tensor(alpha_t, dtype=torch.float32, device=self.device), torch.tensor(beta_t, dtype=torch.float32, device=self.device)

    def forward_diffusion(self, x0, t):
        """Applies Gaussian noise to EEG signals at time step t."""
        noise = torch.randn_like(x0, device=self.device)
        alpha_t = self.alpha_t[t].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return x_t, noise

    @torch.no_grad()
    def reverse_diffusion(self, batch_size, class_labels, spectrogram):
        x = torch.randn((batch_size, self.num_channels, self.num_timesteps), device=self.device)
        # print("x.shape:", x.shape)
        for t in reversed(range(self.num_timesteps)):
            x.requires_grad = False
            if torch.isnan(x).any():
                print(f"❗ NaN detected at timestep {t} in x (before model). Breaking.")
                break
            # print("spectrogram.shape:", spectrogram.shape)
            # Predict noise
            t_tensor = torch.full((batch_size, 1), t, dtype=torch.float32, device=self.device)
            noise_pred = self.model(x, class_labels, t_tensor, spectrogram)
            # print("noise_pred.shape:", noise_pred.shape)

            if torch.isnan(noise_pred).any():
                print(f"❗ NaN detected at timestep {t} in predicted noise. Breaking.")
                break

            # Standard DDPM step (adjust if your implementation differs)
            x = x - self.beta_schedule[t] * noise_pred

            # Optional noise re-addition
            if t > 0:
                x += torch.randn_like(x) * self.noise_scale[t]

            # Check again
            if torch.isnan(x).any():
                print(f" NaN after noise addition at timestep {t}. Breaking.")
                break

        return x
    
    
    
    

# -------------- TRAINING METRICS : MAXIMUM MEAN DISCREPENCY ------------------ 
def compute_mmd(real, generated, kernel_bandwidth=1.0):
    def gaussian_kernel(x, y, bandwidth):
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)
        y_norm = (y ** 2).sum(dim=-1, keepdim=True)
        dist = x_norm + y_norm.T - 2 * (x @ y.T)
        k = torch.exp(-dist / (2 * bandwidth ** 2))

        if torch.isnan(k).any():
            print("NaN detected in kernel matrix")
        if torch.isinf(k).any():
            print("Inf detected in kernel matrix")
        return k

    real = real.view(real.shape[0], -1)
    generated = generated.view(generated.shape[0], -1)
    
    print("real max:", real.max().item(), "min:", real.min().item())
    print("generated max:", generated.max().item(), "min:", generated.min().item())

    if torch.isnan(real).any() or torch.isinf(real).any():
        print("NaN or Inf in real EEG")
    if torch.isnan(generated).any() or torch.isinf(generated).any():
        print("NaN or Inf in generated EEG")

    k_real_real = gaussian_kernel(real, real, kernel_bandwidth)
    k_generated_generated = gaussian_kernel(generated, generated, kernel_bandwidth)
    k_real_generated = gaussian_kernel(real, generated, kernel_bandwidth)

    mmd = k_real_real.mean() + k_generated_generated.mean() - 2 * k_real_generated.mean()

    if torch.isnan(mmd):
        print("⚠️ Warning: MMD is NaN. real.mean:", real.mean().item(), "generated.mean:", generated.mean().item())

    return mmd.item() if not torch.isnan(mmd) else 0.0




# -------------- TRAINING METRICS: FRECHET DISTANCE ------------------ 

def compute_frechet_distance(real, generated, eps=1e-6):
    real = real.view(real.shape[0], -1).cpu().numpy()
    generated = generated.view(generated.shape[0], -1).cpu().numpy()

    mu_real, cov_real = real.mean(axis=0), np.cov(real, rowvar=False)
    mu_generated, cov_generated = generated.mean(axis=0), np.cov(generated, rowvar=False)

    # Add small epsilon for numerical stability
    cov_real += np.eye(cov_real.shape[0]) * eps
    cov_generated += np.eye(cov_generated.shape[0]) * eps

    mean_diff = np.sum((mu_real - mu_generated) ** 2)

    # sqrtm might still return complex results due to numerical error
    cov_sqrt, _ = sqrtm(cov_real @ cov_generated, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    frechet_dist = mean_diff + np.trace(cov_real + cov_generated - 2 * cov_sqrt)
    return float(frechet_dist)




# -------------- TRAINING METRICS PEARSON CORRELATION ------------------ 

def pearson_correlation(real, generated):
    """
    Compute Pearson Correlation between real and generated EEG signals.

    Args:
        real (torch.Tensor): Real EEG signals of shape (batch_size, channels, time_dim).
        generated (torch.Tensor): Generated EEG signals of shape (batch_size, channels, time_dim).

    Returns:
        float: Pearson correlation coefficient (closer to 1 is better).
    """
    # Flatten EEG signals to (batch_size, feature_dim)
    real = real.view(real.shape[0], -1)
    generated = generated.view(generated.shape[0], -1)

    # Compute means
    mean_real = real.mean(dim=1, keepdim=True)
    mean_generated = generated.mean(dim=1, keepdim=True)

    # Compute Pearson Correlation
    num = ((real - mean_real) * (generated - mean_generated)).sum(dim=1)
    den = torch.sqrt(((real - mean_real) ** 2).sum(dim=1) * ((generated - mean_generated) ** 2).sum(dim=1))
    correlation = num / (den + 1e-8)  # Add epsilon to avoid division by zero

    return correlation.mean().item()




# -------------- BACKWARDS LOSS FOR MIXED PRECISION TRAINING ------------------
def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)
        
        
        
        
# ---- Generation Function ----
@torch.no_grad()
def generate_for_class(class_id, n_samples_per_class,diffusion_module ):
    # One-hot encoded class tensor
    class_tensor = F.one_hot(
        torch.tensor([class_id] * n_samples_per_class),
        num_classes=CFG.N_CLASSES
    ).float().to(CFG.device)

    # Spectrogram prior (zeros)
    spec_shape = (
        n_samples_per_class,
        diffusion_module.model.n_channels,
        50,
        50
    )
    spectrogram_prior = torch.zeros(spec_shape, device=CFG.device)
    # Generate EEGs
    generated_eeg = diffusion_module.reverse_diffusion(
        batch_size=n_samples_per_class,
        class_labels=class_tensor,
        spectrogram=spectrogram_prior
    )
    return generated_eeg.cpu().numpy()




def augment_dataset_balanced(real_df, all_real_eegs, gen_data_dir, samples_per_class=5, start_idx=100000):
    """
    Augments EEG data in a class-balanced way using synthetic .npy files.

    Args:
        real_df: Original metadata DataFrame
        all_real_eegs: Dict of original EEG signals {eeg_id: np.array}
        gen_data_dir: Path to directory containing generated_class_{i}.npy
        samples_per_class: Number of synthetic samples to add per class
        start_idx: Unique ID offset for synthetic eeg_ids

    Returns:
        aug_df: Augmented metadata DataFrame
        aug_eeg_dict: Updated EEG dict with synthetic entries
    """
    aug_rows = []
    aug_eeg_dict = all_real_eegs.copy()

    for class_id in range(CFG.N_CLASSES):
        file_path = gen_data_dir / f"generated_class_{class_id}.npy"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        generated = np.load(file_path)  # shape: (N, 8, 2000)
        assert isinstance(generated, np.ndarray)
        assert not isinstance(generated.shape[1], tuple)  # e.g., no ragged arrays 
        # print(f"generated shape: {generated.shape[1:]}")
        
        # if generated.ndim != 3 or generated.shape[1:] != (8, CFG.EEG_PTS):
        #     raise ValueError(f"Invalid shape in {file_path}: {generated.shape}")

        # Limit to desired number of samples
        selected = generated[:samples_per_class]

        for i, eeg in enumerate(selected):
            new_id = f"synthetic_{start_idx + class_id * samples_per_class + i}"
            aug_eeg_dict[new_id] = eeg.T  # Shape (2000, 8) for downstream compatibility

            row = {
                "eeg_id": new_id,
                "patient_id": f"synthetic_patient_{class_id}_{i}",
                CFG.TGT_COL: class_id,
            }

            for j, col in enumerate(CFG.TGT_VOTE_COLS):
                row[col] = 1.0 if j == class_id else 0.0  # one-hot

            aug_rows.append(row)

    # Merge with original
    aug_df = pd.concat([real_df, pd.DataFrame(aug_rows)], ignore_index=True)
    return aug_df, aug_eeg_dict



def plot_class_distribution_comparison(real_data, generated_data, n_classes, output_dir):
    """
    Plots and saves class distribution before and after augmentation.
    
    Args:
        real_data (List[Tuple[x, y]]): Real EEG dataset (x, y) where y is a torch tensor of shape [N_CLASSES].
        generated_data (List[Tuple[x, y]]): Generated EEG dataset (x, y) where y is a torch tensor of shape [N_CLASSES].
        n_classes (int): Total number of classes.
        output_dir (Path): Path to save the plot under working/plots/
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count real samples per class
    real_counts = np.zeros(n_classes)
    for _, y in real_data:
        cls_idx = int(torch.argmax(y).item())
        real_counts[cls_idx] += 1

    # Count generated samples per class
    gen_counts = np.zeros(n_classes)
    for _, y in generated_data:
        cls_idx = int(torch.argmax(y).item())
        gen_counts[cls_idx] += 1

    # Combined
    total_counts = real_counts + gen_counts

    # Plot
    labels = [f'Class {i}' for i in range(n_classes)]
    x = np.arange(n_classes)
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, real_counts, width, label='Before Augmentation', color='skyblue')
    plt.bar(x + width/2, total_counts, width, label='After Augmentation', color='salmon')

    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Before and After Augmentation')
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()

    # Save
    save_path = output_dir / "class_distribution_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Class distribution plot saved to {save_path}")


def plot_eeg_comparison(real_data, generated_data_dir, class_label: int, save_path: str):
    """
    Plot raw EEG waveforms for all channels in real and generated EEGs for a given class.

    Args:
        real_data (List[Tuple[x, y]]): List of (EEG, label) pairs
        generated_data_dir (Path): Directory containing generated_class_{label}.npy files
        class_label (int): Class index to visualize
        save_path (str): Path to save output image
    """
    
    # Assume: TGT_VOTE_COLS = CFG.TGT_VOTE_COLS
    class_labels = {i: name.replace("_vote", "").upper() for i, name in enumerate(CFG.TGT_VOTE_COLS)}
    # Find one real EEG sample for the class
    real_eeg = None
    for x, y in real_data:
        if y.argmax().item() == class_label:
            real_eeg = x  # shape: (8, 2000)
            break
    assert real_eeg is not None, f"No real EEG sample found for class {class_label}"

    # Load generated EEG
    gen_path = generated_data_dir / f"generated_class_{class_label}.npy"
    assert gen_path.exists(), f"Missing generated file: {gen_path}"
    gen_eeg = torch.tensor(np.load(gen_path)[0], dtype=torch.float32)

    assert real_eeg.shape == gen_eeg.shape, f"Mismatch: real {real_eeg.shape} vs generated {gen_eeg.shape}"

    # Plot 8 channels in 4x2 layout
    fig, axs = plt.subplots(4, 2, figsize=(14, 10))
    for ch in range(8):
        ax = axs[ch // 2, ch % 2]
        ax.plot(real_eeg[ch].cpu().numpy(), label="Real", alpha=0.8)
        ax.plot(gen_eeg[ch].cpu().numpy(), label="Generated", alpha=0.7, linestyle='--')
        ax.set_title(f"Channel {ch}")
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")

    # Get class name from dictionary
    class_name = class_labels.get(class_label, f"Class {class_label}")

    fig.suptitle(f"EEG Comparison for {class_name} (Class {class_label})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"Saved multi-channel EEG comparison plot to {save_path}")
        
        
        


def plot_spectrogram_comparison(real_data, generated_data_dir, class_label: int, config, save_path: str):
    """
    Plot STFT spectrograms for real and generated EEGs of a given class.
    
    Args:
        real_data: List of (eeg, label) pairs
        generated_data_dir: Path to .npy directory
        class_label: Class ID to visualize
        config: CFG object (should contain STFT params)
        save_path: Path to save plot
    """
    # Select real EEG
    real_eeg = None
    for x, y in real_data:
        if y.argmax().item() == class_label:
            real_eeg = x
            break
    assert real_eeg is not None, f"No real EEG sample found for class {class_label}"

    # Load generated EEG
    gen_path = generated_data_dir / f"generated_class_{class_label}.npy"
    assert gen_path.exists(), f"Missing generated EEG file: {gen_path}"
    gen_eeg = torch.tensor(np.load(gen_path)[0], dtype=torch.float32)

    # Compute STFT for both
    def compute_stft_single(signal, config):
        _, _, Z = stft(signal.cpu().numpy(), fs=200, 
                       nperseg=config.diffEEG_trainer["stft_n_fft"],
                       noverlap=config.diffEEG_trainer["stft_hop_length"],
                       window=config.diffEEG_trainer["stft_window"])
        return np.log1p(np.abs(Z))  # log scaled spectrogram

    real_spec = compute_stft_single(real_eeg[0], config)
    gen_spec = compute_stft_single(gen_eeg[0], config)

    # Plot spectrograms
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(real_spec, aspect="auto", origin="lower", cmap="viridis")
    plt.title("Real Spectrogram")

    plt.subplot(1, 2, 2)
    plt.imshow(gen_spec, aspect="auto", origin="lower", cmap="viridis")
    plt.title("Generated Spectrogram")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved spectrogram comparison plot at {save_path}")
    
    
    
def visualize_samples(epoch, inputs, outputs, save_dir="mnist_samples"):
    os.makedirs(save_dir, exist_ok=True)
    inputs = inputs.view(-1, 28, 28).cpu().detach().numpy()
    outputs = outputs.view(-1, 28, 28).cpu().detach().numpy()

    fig, axs = plt.subplots(2, 6, figsize=(12, 4))
    for i in range(6):
        axs[0, i].imshow(inputs[i], cmap='gray')
        axs[0, i].set_title("Input")
        axs[0, i].axis('off')

        axs[1, i].imshow(outputs[i], cmap='gray')
        axs[1, i].set_title("Output")
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.close()

def train_loop(model, train_loader, val_loader, optimizer, loss_fn,logger, num_epochs=50, save_dir = "mnist_validation_samples"):
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            x = x.to(CFG.device)               # Shape: [B, 784]
            y_onehot = F.one_hot(y, num_classes=10).float().to(CFG.device)

            # Forward diffusion step (here: add noise)
            noise = torch.randn_like(x)
            noisy_x = x + noise

            # Predict the noise from noisy_x
            predicted_noise = model(noisy_x, y_onehot)

            loss = loss_fn(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f">>> Train Loss: {avg_loss:.4f}")

        # ---- Validation every 10 epochs ----
        if epoch % 10 == 0:
            model.eval()
            val_loss = 0
            all_inputs, all_outputs = [], []

            with torch.no_grad():
                for x, y in train_loader:
                    x = x.to(CFG.device)
                    y_onehot = F.one_hot(y, num_classes=10).float().to(CFG.device)

                    noise = torch.randn_like(x)
                    noisy_x = x + noise
                    predicted_noise = model(noisy_x, y_onehot)
                    loss = loss_fn(predicted_noise, noise)
                    val_loss += loss.item()

                    reconstructed = noisy_x - predicted_noise
                    all_inputs.append(x[:6])
                    all_outputs.append(reconstructed[:6])

            logger.info(f">>> Val Loss: {val_loss / len(train_loader):.4f}")
            visualize_samples(epoch, torch.cat(all_inputs)[:6], torch.cat(all_outputs)[:6],save_dir)
            
            
            
def freeze_except(model, names_to_train):
    for name, param in model.named_parameters():
        param.requires_grad = any(n in name for n in names_to_train)


