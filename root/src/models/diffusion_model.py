import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from utils.cfg_utils import CFG

class DiffEEG(nn.Module):
    """
    Diffusion-based EEG augmentation model following the architecture from the paper.
    """
    def __init__(self, config = CFG):
        super().__init__()
        self.n_classes = config.diffEEG_trainer["n_classes"]
        self.n_channels = config.diffEEG_trainer["n_channels"]
        self.hidden_dim = config.diffEEG_trainer["hidden_channels"]
        self.dropout = config.diffEEG_trainer["dropout"]
        
        # Step embedding (Sin-Cos Encoding)
        self.step_embedding_dim = self.hidden_dim
        
        # Class conditioning embedding
        self.class_embedding = nn.Embedding(self.n_classes, self.hidden_dim)
        
        # Spectrogram condition embedding
        self.spectrogram_embed = nn.Conv1d(self.n_channels, self.hidden_dim, kernel_size=1)
        
        
        # Step embedding (Sin-Cos Encoding + MLP projection)
        self.step_embedding_dim = self.hidden_dim
        self.step_embedding_mlp = nn.Sequential(
            nn.Linear(self.step_embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        
        # Spectrogram conditioning: Use 2D Transposed Convolutions for upsampling
        self.spectrogram_upconv1 = nn.ConvTranspose2d(
            in_channels=self.n_channels,  # Input channels from spectrogram
            out_channels=self.hidden_dim // 2,  # Expand feature space
            kernel_size=(3, 3),  # Small upsampling kernel
            stride=(2, 2),  # Upsample by 2x
            padding=1
        )
        
        self.spectrogram_upconv2 = nn.ConvTranspose2d(
            in_channels=self.hidden_dim // 2,
            out_channels=self.hidden_dim,  # Match EEG feature space
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1
        )

        # Final 1×1 Convolution to match EEG feature space
        self.spectrogram_embed = nn.Conv2d(
            in_channels=self.hidden_dim, 
            out_channels=self.hidden_dim, 
            kernel_size=1
        )
        
        # Input Block: 1x1 Conv to map EEG into residual space
        self.input_conv = nn.Conv1d(self.n_channels, self.hidden_dim, kernel_size=1)
        
        # Residual Blocks with Bi-DilConv
        self.res_block1 = self._residual_block(self.hidden_dim, dilation=1, dropout= self.dropout)
        self.res_block2 = self._residual_block(self.hidden_dim, dilation=2, dropout=self.dropout)
        self.res_block3 = self._residual_block(self.hidden_dim, dilation=4, dropout=self.dropout)
        self.res_block4 = self._residual_block(self.hidden_dim, dilation=8, dropout=self.dropout)
        
        # Skip Connection Summation
        self.skip_sum = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        
        # Output Block: 1x1 Conv to transform back to EEG format
        self.output_conv = nn.Conv1d(self.hidden_dim, self.n_channels, kernel_size=1)
    
    def _residual_block(self, channels, dilation, dropout=0.1):
        """Bi-Dilated Convolution with Gated Tanh Unit (GTU) and Dropout."""
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.Sigmoid(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Dropout(dropout)  # Add dropout after convolution layers
        )
    
    def sinusoidal_embedding(self, diffusion_step, dim):
        """Sinusoidal positional encoding for diffusion step + MLP transformation."""
        half_dim = dim // 2
        emb = torch.exp(torch.arange(half_dim, device=diffusion_step.device) * -np.log(10000) / (half_dim - 1))
        emb = diffusion_step * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        # Pass through MLP transformation
        emb = self.step_embedding_mlp(emb)

        return emb
    
    
    def forward(self, x, class_label, diffusion_step, spectrogram):
        """Forward pass through DiffEEG model.

        Args:
            x (Tensor): EEG signal (batch_size, n_channels, time_dim)
            class_label (Tensor): Class label tensor (batch_size,)
            diffusion_step (Tensor): Diffusion step tensor (batch_size, 1)
            spectrogram (Tensor): STFT spectrograms (batch_size, n_channels, time_dim)

        Returns:
            Tensor: Predicted noise (batch_size, n_channels, time_dim)
        """
        # Get batch size, channel count, and time dimension
        batch_size, n_channels, freq_dim, time_dim = spectrogram.shape
        print(f"Shape of spectrogram before conditional embedding: {spectrogram.shape}")
        
        
        class_label = class_label.long()

        # Compute step embedding using sinusoidal encoding
        step_emb = self.sinusoidal_embedding(diffusion_step, self.step_embedding_dim)
        step_emb = step_emb.unsqueeze(-1).expand(-1, -1, time_dim)
        # print(f"Shape of step_emb after sinusoidal embedding: {step_emb.shape}")

        # Embed class label
        class_label = class_label.argmax(dim=1).long()  # Convert one-hot to indices
        class_emb = self.class_embedding(class_label) 
        #print(f"Shape after class_embedding: {class_emb.shape}")
        class_emb = class_emb.unsqueeze(-1)  # [batch_size, hidden_dim, 1]
        #print(f"Shape after unsqueeze: {class_emb.shape}")
        class_emb = class_emb.expand(-1, -1, x.shape[-1])  # [batch_size, hidden_dim, 2000] 
        #print(f"Shape after expansion: {class_emb.shape}")


        # # Ensure the correct shape: [batch_size, channels, freq_dim, time_dim]
        # spectrogram = spectrogram.permute(0, 3, 1, 2)  # Swap last axis to second axis
        
        # print(f"Shape of spectrogram after dimension permutation: {spectrogram.shape}")
        
        # Apply transposed convolutions to spectrogram
        spectrogram = self.spectrogram_upconv1(spectrogram)  # Upsample
        spectrogram = F.relu(spectrogram)  # Non-linearity
        spectrogram = self.spectrogram_upconv2(spectrogram)  # Further upsample
        spectrogram = F.relu(spectrogram)  
        spectrogram = self.spectrogram_embed(spectrogram)  # Final dimension matching
        
        # Flatten spectrogram to match EEG shape
        spectrogram = spectrogram.view(batch_size, self.hidden_dim, -1)  # Convert to (batch, hidden_dim, time)

        # Initial EEG feature transformation
        x = self.input_conv(x) + step_emb + class_emb + spectrogram

        # Residual Block Processing
        x1 = self.res_block1(x)
        x2 = self.res_block2(x1)
        x3 = self.res_block3(x2)
        x4 = self.res_block4(x3)

        # Skip Connection Summation
        x = self.skip_sum(x1 + x2 + x3 + x4)

        # Output transformation
        x = self.output_conv(x)

        return x