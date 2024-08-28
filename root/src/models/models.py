import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """
    A basic building block for the Spectrogram Model, consisting of three convolutional layers,
    pooling, batch normalization, and dropout, with a skip connection.
    """
    def __init__(self, in_channels, out_channels, pool_type='max', pool_size=(2, 2), dropout_p=0.5):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=pool_size)

        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout_p)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 conv to match channels for skip connection

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.bn(x)
        x = self.dropout(x)

        # Match the dimensions of identity to x
        if identity.shape != x.shape:
            identity = F.interpolate(identity, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            identity = self.conv1x1(identity)

        x += identity  # Add skip connection
        return x

class Spectrogram_Model(nn.Module):
    """
    The Spectrogram Model for processing spectrogram data.
    Consists of multiple blocks of convolutional layers, followed by global average pooling and fully connected layers.
    """
    def __init__(self, num_classes=6):
        super(Spectrogram_Model, self).__init__()
        self.block1 = Block(in_channels=3, out_channels=16, pool_type='max', pool_size=(2, 2))
        self.block2 = Block(in_channels=16, out_channels=32, pool_type='avg', pool_size=(2, 2))
        self.block3 = Block(in_channels=32, out_channels=64, pool_type='max', pool_size=(2, 2))
        self.block4 = Block(in_channels=64, out_channels=128, pool_type='avg', pool_size=(2, 2))
        self.block5 = Block(in_channels=128, out_channels=256, pool_type='max', pool_size=(2, 2))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.block1(x)  # Output: (16, H/2, W/2)
        x = self.block2(x)  # Output: (32, H/4, W/4)
        x = self.block3(x)  # Output: (64, H/8, W/8)
        x = self.block4(x)  # Output: (128, H/16, W/16)
        x = self.block5(x)  # Output: (256, H/32, W/32)

        x = self.gap(x)  # Global Average Pooling to (256, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (256)
        x = self.fc(x)  # Fully connected layer
        x = self.log_softmax(x)  # Apply LogSoftmax
        return x

class EnhancedEEGNetAttention(nn.Module):
    """
    Enhanced EEGNet with Attention mechanism for processing EEG data.
    The model includes convolutional layers, depthwise separable convolutions, attention, and fully connected layers.
    """
    def __init__(self, num_channels=37, num_samples=3000, num_classes=6):
        super(EnhancedEEGNetAttention, self).__init__()
        
        # Model parameters
        self.num_channels = num_channels
        self.num_samples = num_samples
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 64), padding='same', bias=False)  # Increased filters to 32
        self.batchnorm1 = nn.BatchNorm2d(32)
        
        # Depthwise convolution block
        self.depthwiseConv = nn.Conv2d(32, 64, kernel_size=(num_channels, 1), groups=32, bias=False)  # Use num_channels
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=0.5)
        
        # Second convolutional block (separable convolution)
        self.separableConv1 = nn.Conv2d(64, 128, kernel_size=(1, 16), padding='same', bias=False)  # Increased filters to 128
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        self.separableConv2 = nn.Conv2d(128, 256, kernel_size=(1, 8), padding='same', bias=False)  # Increased filters to 256
        self.batchnorm4 = nn.BatchNorm2d(256)
        
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=0.5)
        
        # Third convolutional block (additional complexity)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 4), padding='same', bias=False)  # Added another convolutional layer
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.activation2 = nn.ELU()
        self.avg_pool3 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout3 = nn.Dropout(p=0.5)
        
        # Attention layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=512, num_heads=8)  # Increased embed_dim to 512
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(512 * (self.num_samples // 128), 512)  # Adjusted units based on the reduced sample size after pooling
        self.dropout4 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dense3 = nn.Linear(256, num_classes)
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # Add L2 regularization (weight decay)
        self.weight_decay = 1e-3  # Example value, tune as necessary

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        x = self.separableConv1(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        
        x = self.separableConv2(x)
        x = self.batchnorm4(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        # Additional convolutional block
        x = self.conv2(x)
        x = self.batchnorm5(x)
        x = self.activation2(x)
        x = self.avg_pool3(x)
        x = self.dropout3(x)
        
        # Attention mechanism
        x = x.squeeze(2).permute(2, 0, 1)  # Reshape for attention layer
        x, _ = self.attention_layer(x, x, x)
        x = x.permute(1, 2, 0).unsqueeze(2)  # Reshape back to original format
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout4(x)
        x = self.dense2(x)
        x = self.dropout5(x)
        x = self.dense3(x)
        
        return self.log_softmax(x)

class MultimodalModel(nn.Module):
    """
    Multimodal model that combines the outputs of EEG and Spectrogram models.
    The combined output is passed through fully connected layers to predict the final class.
    """
    def __init__(self, eeg_model, spectrogram_model, num_classes=6):
        super(MultimodalModel, self).__init__()
        self.eeg_model = eeg_model
        self.spectrogram_model = spectrogram_model

        # Combining the outputs of the two models
        combined_output_size = eeg_model.dense1.out_features + spectrogram_model.fc.out_features

        self.fc1 = nn.Linear(combined_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, eeg_data, spectrogram_data):
        eeg_output = self.eeg_model(eeg_data)
        spectrogram_output = self.spectrogram_model(spectrogram_data)

        combined = torch.cat((eeg_output, spectrogram_output), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)
        x = self.log_softmax(x)

        return x

    def forward_spectrogram(self, spectrogram_data):
        return self.spectrogram_model(spectrogram_data)
        
        