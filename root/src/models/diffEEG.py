import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.utils.checkpoint as checkpoint


class GTU(nn.Module):
    """Gated Tanh Unit (GTU) for spectrogram fusion."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        return torch.tanh(self.conv1(x)) * torch.sigmoid(self.conv2(x))


class DiffEEG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_classes = config.diffEEG_trainer["n_classes"]
        self.n_channels = config.diffEEG_trainer["n_channels"]
        self.hidden_dim = config.diffEEG_trainer["hidden_channels"]
        self.dropout = config.diffEEG_trainer["dropout"]

        # Handle device
        self.device_type = (
            config.device.type if isinstance(config.device, torch.device) else str(config.device)
        )

        # Step embedding MLP (sinusoidal + MLP)
        self.step_embedding_dim = self.hidden_dim
        self.step_embedding_mlp = nn.Sequential(
            nn.Linear(self.step_embedding_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Class embedding
        self.class_embedding = nn.Embedding(self.n_classes, self.hidden_dim)

        # Spectrogram upsampling (conv + interpolation)
        self.spectrogram_upsample1 = nn.ConvTranspose2d(
            in_channels=self.n_channels,
            out_channels=self.hidden_dim // 2,
            kernel_size=(3, 3),
            stride=(1, 8),
            padding=(1, 2)
        )

        self.channel_expand = nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=1)
        self.spectrogram_project = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.gtu = GTU(self.hidden_dim)

        # EEG projection
        self.input_conv = nn.Conv1d(self.n_channels, self.hidden_dim, kernel_size=1)

        # Residual blocks
        self.res_block1 = self._residual_block(self.hidden_dim, dilation=1)
        self.res_block2 = self._residual_block(self.hidden_dim, dilation=2)
        self.res_block3 = self._residual_block(self.hidden_dim, dilation=4)
        self.res_block4 = self._residual_block(self.hidden_dim, dilation=8)

        # Skip connection projection + norm
        self.skip_sum = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.layer_norm = nn.GroupNorm(1, self.hidden_dim)

        # Final EEG reconstruction
        self.final_projection = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.GroupNorm(1, self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.n_channels, kernel_size=1)
        )

    def _residual_block(self, channels, dilation):
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
            nn.Dropout(self.dropout)
        )

    def sinusoidal_embedding(self, diffusion_step, dim):
        half_dim = dim // 2
        emb = torch.exp(torch.arange(half_dim, device=diffusion_step.device) * -np.log(10000) / (half_dim - 1))
        emb = diffusion_step * emb
        return torch.cat((emb.sin(), emb.cos()), dim=-1).view(-1, dim)

    def forward(self, x, class_label, diffusion_step, spectrogram):
        with torch.amp.autocast(device_type=self.device_type):
            B, _, T = x.shape

            # === Step Embedding ===
            step_emb = self.sinusoidal_embedding(diffusion_step.view(-1, 1), self.step_embedding_dim)
            step_emb = self.step_embedding_mlp(step_emb).unsqueeze(-1).expand(-1, -1, T)

            # === Class Embedding ===
            class_label = class_label.argmax(dim=1).long()
            class_emb = self.class_embedding(class_label).unsqueeze(-1).expand(-1, -1, T)

            # === Spectrogram Processing ===
            if self.training:
                spectrogram = self.recombine_spectrograms(spectrogram, class_label)

            spec = self.spectrogram_upsample1(spectrogram)
            spec = F.relu(spec).flatten(start_dim=2)
            spec = F.interpolate(spec, size=T, mode="linear", align_corners=False)
            spec = self.channel_expand(spec)
            spec = self.spectrogram_project(spec)
            spec = self.gtu(spec)

            # === Input Projection ===
            eeg_proj = self.input_conv(x)

            # === Combine Features ===
            x = eeg_proj + step_emb + class_emb + spec

            # === Residual Processing ===
            x1 = checkpoint.checkpoint(self.res_block1, x)
            x2 = checkpoint.checkpoint(self.res_block2, x1)
            x3 = checkpoint.checkpoint(self.res_block3, x2)
            x4 = checkpoint.checkpoint(self.res_block4, x3)

            x = self.layer_norm(self.skip_sum(x1 + x2 + x3 + x4))

            # === Final EEG Reconstruction ===
            return self.final_projection(x)

    def recombine_spectrograms(self, spectrograms, class_labels):
        """Mix spectrograms only within same class."""
        new_specs = spectrograms.clone()
        for c in range(self.n_classes):
            idx = (class_labels == c).nonzero(as_tuple=True)[0]
            if len(idx) > 1:
                perm = idx[torch.randperm(len(idx), device=spectrograms.device)]
                alpha = 0.5
                new_specs[idx] = alpha * spectrograms[idx] + (1 - alpha) * spectrograms[perm]
        return new_specs
    
    
    
    
    
class DiffEEG_SanityCheck(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection from MNIST (28x28) to hidden_dim
        self.input_proj = nn.Linear(28 * 28, hidden_dim)

        # Dummy Step Embedding (constant for testing)
        self.step_embed = nn.Parameter(torch.randn(1, hidden_dim))

        # Dummy Class Embedding (optional)
        self.class_embed = nn.Parameter(torch.randn(1, hidden_dim))

        # Residual Blocks
        self.res1 = self._res_block(hidden_dim)
        self.res2 = self._res_block(hidden_dim)
        self.res3 = self._res_block(hidden_dim)
        self.res4 = self._res_block(hidden_dim)

        # Skip connection + normalization
        self.skip_sum = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection to image size
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 28 * 28),
            nn.Sigmoid()
        )

    def _res_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)  # flatten MNIST
        x = self.input_proj(x)

        step = self.step_embed.expand(B, -1)
        cls = self.class_embed.expand(B, -1)

        x = x + step + cls

        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)

        x = self.skip_sum(x1 + x2 + x3 + x4)
        x = self.norm(x)
        x = self.output_proj(x)

        return x.view(B, 1, 28, 28)
    
    
    
    
#     import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import random

# class DiffEEG_Updated(nn.Module):
#     """
#     Diffusion-based EEG augmentation model with class-specific spectrogram recombination, 
#     GTU, ConvTranspose2d-based spectrogram upsampling, and LayerNorm-enhanced skip connection fusion.
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.n_classes = config.diffEEG_trainer["n_classes"]
#         self.n_channels = config.diffEEG_trainer["n_channels"]
#         self.hidden_dim = config.diffEEG_trainer["hidden_channels"]
#         self.dropout = config.diffEEG_trainer["dropout"]
        
#         # Step embedding (Sin-Cos Encoding + MLP projection)
#         self.step_embedding_dim = self.hidden_dim
#         self.step_embedding_mlp = nn.Sequential(
#             nn.Linear(self.step_embedding_dim, self.hidden_dim),
#             nn.Sigmoid(),  # ✅ Updated: Added Sigmoid activation
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.hidden_dim)
#         )

#         # Class conditioning embedding
#         self.class_embedding = nn.Embedding(self.n_classes, self.hidden_dim)
        
#         # Spectrogram Processing - **Now Uses ConvTranspose2d**
#         self.spectrogram_upsample1 = nn.ConvTranspose2d(
#             in_channels=self.n_channels,  
#             out_channels=self.hidden_dim // 2,  
#             kernel_size=(3, 3),  
#             stride=(1, 16),  # ✅ Matches paper: stride=1x16
#             padding=(1, 2)  # ✅ Matches paper: padding=1x2
#         )
        
#         self.spectrogram_upsample2 = nn.ConvTranspose2d(
#             in_channels=self.hidden_dim // 2,
#             out_channels=self.hidden_dim,  
#             kernel_size=(3, 3),
#             stride=(1, 16),  # ✅ Matches paper: second upsampling layer
#             padding=(1, 2)
#         )

#         # Final 1x1 Convolution to align with EEG hidden space
#         self.spectrogram_project = nn.Conv1d(
#             in_channels=self.hidden_dim,
#             out_channels=self.hidden_dim,
#             kernel_size=1
#         )
        
#         # Input Block: 1x1 Conv to map EEG into residual space
#         self.input_conv = nn.Conv1d(self.n_channels, self.hidden_dim, kernel_size=1)
        
#         # Residual Blocks with Bi-DilConv and GTU
#         self.res_block1 = self._residual_block(self.hidden_dim, dilation=1, dropout=self.dropout)
#         self.res_block2 = self._residual_block(self.hidden_dim, dilation=2, dropout=self.dropout)
#         self.res_block3 = self._residual_block(self.hidden_dim, dilation=4, dropout=self.dropout)
#         self.res_block4 = self._residual_block(self.hidden_dim, dilation=8, dropout=self.dropout)

#         # Skip Connection Summation with LayerNorm before summation
#         self.layer_norm = nn.LayerNorm(self.hidden_dim)
#         self.skip_sum = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1)

#         # Final Projection Layer (Refinement before output)
#         self.final_projection = nn.Sequential(
#             nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
#             nn.ReLU(),
#             nn.LayerNorm(self.hidden_dim),  # ✅ Ensures stable feature mapping
#             nn.Conv1d(self.hidden_dim, self.n_channels, kernel_size=1)  # Maps back to EEG space
#         )
    
#     def _residual_block(self, channels, dilation, dropout=0.1):
#         """GTU-based Residual Block with LayerNorm applied before summation."""
#         return nn.Sequential(
#             nn.Conv1d(channels, channels, kernel_size=1),
#             GTU(channels),  # GTU replaces individual activations
#             nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
#             nn.Conv1d(channels, channels, kernel_size=1),
#             nn.LayerNorm(channels),  # Normalize feature maps before returning
#             nn.Dropout(dropout)  
#         )
    
#     def sinusoidal_embedding(self, diffusion_step, dim):
#         """Sinusoidal positional encoding for diffusion step + MLP transformation."""
#         half_dim = dim // 2
#         emb = torch.exp(torch.arange(half_dim, device=diffusion_step.device) * -np.log(10000) / (half_dim - 1))
#         emb = diffusion_step * emb
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

#         # Pass through MLP transformation
#         emb = self.step_embedding_mlp(emb)

#         return emb

#     def forward(self, x, class_label, diffusion_step, spectrogram):
#         """Forward pass through DiffEEG model."""
#         batch_size, n_channels, freq_dim, time_dim = spectrogram.shape
        
#         # Apply Class-Specific Spectrogram Recombination
#         spectrogram = self.recombine_spectrograms(spectrogram, class_label)

#         class_label = class_label.long()

#         # Compute step embedding using sinusoidal encoding
#         step_emb = self.sinusoidal_embedding(diffusion_step, self.step_embedding_dim)
#         step_emb = step_emb.unsqueeze(-1).expand(-1, -1, time_dim)

#         # Embed class label
#         class_label = class_label.argmax(dim=1).long()  
#         class_emb = self.class_embedding(class_label)  
#         class_emb = class_emb.unsqueeze(-1)  
#         class_emb = class_emb.expand(-1, -1, x.shape[-1])  

#         # **Spectrogram Upsampling with ConvTranspose2d**
#         spectrogram = self.spectrogram_upsample1(spectrogram)  
#         spectrogram = F.relu(spectrogram)  
#         spectrogram = self.spectrogram_upsample2(spectrogram)  
#         spectrogram = F.relu(spectrogram)  

#         # **Final Projection to Match EEG shape**
#         spectrogram = self.spectrogram_project(spectrogram)

#         # Initial EEG feature transformation
#         x = self.input_conv(x) + step_emb + class_emb + spectrogram

#         # Residual Block Processing
#         x1 = self.res_block1(x)
#         x2 = self.res_block2(x1)
#         x3 = self.res_block3(x2)
#         x4 = self.res_block4(x3)

#         # Skip Connection Summation + LayerNorm
#         x = self.layer_norm(self.skip_sum(x1 + x2 + x3 + x4))

#         # Apply Final Projection before output
#         x = self.final_projection(x.permute(0, 2, 1))  # (B, Hidden, T) → (B, T, Channels)
#         x = x.permute(0, 2, 1)  # Back to (B, Channels, T)

#         return x