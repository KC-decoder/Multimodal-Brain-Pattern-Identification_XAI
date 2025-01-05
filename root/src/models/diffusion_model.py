import torch
import torch.nn as nn
import torch.nn.functional as F


class _InputBlock(nn.Module):
    """Input block for DiffEEG."""
    def __init__(self, n_channels):
        super(_InputBlock, self).__init__()
        self.input_conv = nn.Conv2d(n_channels, 128, kernel_size=(1, 1))
        self.relu = nn.ReLU()

    def forward(self, eeg_input, noise):
        """
        Args:
            eeg_input: Raw EEG input (B, C, H, W)
            noise: Noise added to the EEG (B, C, H, W)
        Returns:
            Processed EEG (B, 128, H, W)
        """
        x = self.input_conv(eeg_input + noise)
        x = self.relu(x)
        return x


class _StepEmbedding(nn.Module):
    """Step embedding block for DiffEEG."""
    def __init__(self, n_timesteps):
        super(_StepEmbedding, self).__init__()
        self.fc1 = nn.Linear(n_timesteps, 512)
        self.fc2 = nn.Linear(512, 512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, timestep):
        """
        Args:
            timestep: Timestep tensor (B,)
        Returns:
            Embedded timesteps (B, 512, 1, 1)
        """
        t_emb = self.fc1(timestep)
        t_emb = self.sigmoid(self.fc2(t_emb))
        return t_emb.view(t_emb.size(0), -1, 1, 1)


class _ConditionBlock(nn.Module):
    """Condition block for DiffEEG."""
    def __init__(self, condition_dim):
        super(_ConditionBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(256, condition_dim, kernel_size=(1, 1))

    def forward(self, condition):
        """
        Args:
            condition: Conditioning input (e.g., STFT features) (B, 1, H, W)
        Returns:
            Processed condition (B, 256, H, W)
        """
        c = self.conv1(condition)
        c = self.conv2(c)
        return c


class _ResidualBlock(nn.Module):
    """Residual block for DiffEEG."""
    def __init__(self, n_channels, dilation, condition_dim):
        super(_ResidualBlock, self).__init__()
        self.dilated_conv = nn.Conv2d(n_channels, 256, kernel_size=(3, 3), padding=(dilation, dilation), dilation=dilation)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.output_conv = nn.Conv2d(256, n_channels, kernel_size=(1, 1))
        self.condition_conv = nn.Conv2d(condition_dim, 256, kernel_size=(1, 1))

    def forward(self, x, timestep_embedding, condition):
        """
        Args:
            x: Input tensor (B, C, H, W)
            timestep_embedding: Embedded timesteps (B, C, 1, 1)
            condition: Conditioning input (B, C, H, W)

        Returns:
            Residual and skip connections
        """
        h = self.dilated_conv(x)

        # Apply timestep and condition embeddings
        h += timestep_embedding
        if condition is not None:
            h += self.condition_conv(condition)

        # Gated activation
        h = self.tanh(h) * self.sigmoid(h)

        # Residual and skip connections
        residual = self.output_conv(h)
        return x + residual, residual


class _OutputBlock(nn.Module):
    """Output block for DiffEEG."""
    def __init__(self, n_channels):
        super(_OutputBlock, self).__init__()
        self.skip_conv1 = nn.Conv2d(128, 128, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.skip_conv2 = nn.Conv2d(128, n_channels, kernel_size=(1, 1))

    def forward(self, skip_connections):
        """
        Args:
            skip_connections: Summed skip connections (B, 128, H, W)
        Returns:
            Final output (B, n_channels, H, W)
        """
        x = self.skip_conv1(skip_connections)
        x = self.relu(x)
        x = self.skip_conv2(x)
        return x


class DiffEEG(nn.Module):
    """Main DiffEEG model."""
    def __init__(self, n_channels=8, n_residual_layers=128, n_timesteps=1000):
        super(DiffEEG, self).__init__()
        self.n_channels = n_channels
        self.n_residual_layers = n_residual_layers

        # Blocks
        self.input_block = _InputBlock(n_channels)
        self.step_embedding = _StepEmbedding(n_timesteps)
        self.condition_block = _ConditionBlock(256)
        self.residual_blocks = nn.ModuleList([
            _ResidualBlock(128, dilation=2**i, condition_dim=256) for i in range(n_residual_layers)
        ])
        self.output_block = _OutputBlock(n_channels)

    def forward(self, eeg_input, noise, timestep, condition=None):
        """
        Args:
            eeg_input: EEG input (B, C, H, W)
            noise: Added noise (B, C, H, W)
            timestep: Timestep tensor (B,)
            condition: Conditioning input (e.g., STFT features) (B, 1, H, W)

        Returns:
            Reconstructed EEG signals
        """
        # Input block
        x = self.input_block(eeg_input, noise)

        # Step embedding
        t_emb = self.step_embedding(timestep)

        # Condition block
        if condition is not None:
            c = self.condition_block(condition)
        else:
            c = None

        # Residual blocks
        skip_connections = []
        for layer in self.residual_blocks:
            x, skip = layer(x, t_emb, c)
            skip_connections.append(skip)

        # Sum skip connections
        skip_sum = sum(skip_connections)

        # Output block
        output = self.output_block(skip_sum)
        return output