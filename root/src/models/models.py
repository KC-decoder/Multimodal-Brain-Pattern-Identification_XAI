import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import vit_b_16
from utils.cfg_utils import CFG
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


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
from utils.cfg_utils import CFG, _Logger, _seed_everything


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

class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.scale = attention_dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        #print(f"Query shape: {Q.shape}")
        #print(f"Key shape: {K.shape}")
        #print(f"Value shape: {V.shape}")

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        #print(f"Scores shape: {scores.shape}")

        attention_weights = F.softmax(scores, dim=-1)
        #print(f"Attention weights shape: {attention_weights.shape}")

        output = torch.matmul(attention_weights, V)
        #print(f"Output shape: {output.shape}")

        return output, attention_weights

class EEGNetAttentionDeep(nn.Module):
    def __init__(self, nb_classes, Chans=37, Samples=3000,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, F3=32, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNetAttentionDeep, self).__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 3 (additional block to deepen the network)
        self.conv2 = nn.Conv2d(F2, F3, (1, 16), padding='same', bias=False)
        self.batchnorm4 = nn.BatchNorm2d(F3)
        self.avg_pool3 = nn.AvgPool2d((1, 8))
        self.dropout3 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Attention Mechanism
        self.attention_layer = Attention(F3, F3)  # Update this line to match dimensions

        # Calculate the output size after conv and pooling layers
        self.output_samples = self._get_output_size()
        self.flattened_size = F3 * self.output_samples

        # Classification layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(self.flattened_size, 128)
        self.dense2 = nn.Linear(128, nb_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _get_output_size(self):
        # Forward pass of dummy input to calculate output size
        with torch.no_grad():
            x = torch.zeros(1, 1, self.Chans, self.Samples)
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.depthwiseConv(x)
            x = self.batchnorm2(x)
            x = self.activation(x)
            x = self.avg_pool1(x)
            x = self.dropout1(x)
            x = self.separableConv(x)
            x = self.batchnorm3(x)
            x = self.activation(x)
            x = self.avg_pool2(x)
            x = self.dropout2(x)
            x = self.conv2(x)
            x = self.batchnorm4(x)
            x = self.activation(x)
            x = self.avg_pool3(x)
            x = self.dropout3(x)
            b, c, h, w = x.size()
            return h * w

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        x = self.conv2(x)
        x = self.batchnorm4(x)
        x = self.activation(x)
        x = self.avg_pool3(x)
        x = self.dropout3(x)

        # Attention mechanism
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)  # Reshape to (batch_size, feature_dim, sequence_length)
        x, _ = self.attention_layer(x.permute(0, 2, 1))  # Apply attention and permute back
        x = x.permute(0, 2, 1).contiguous()  # Permute back to original shape
        x = x.view(b, c, h, w)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.log_softmax(x)
        return x



class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=37, Samples=3000, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))

        # Adjust the output samples based on pooling operations
        output_samples = Samples // 32
        final_output_size = F2 * output_samples

        # Classification layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(final_output_size, nb_classes)  # Adjusting linear layer input size
        self.log_softmax = nn.LogSoftmax(dim=1)  # Use LogSoftmax for output

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout(x)

        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.log_softmax(x)
        return x



class DeepConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=37, Samples=3000, dropoutRate=0.5):
        super(DeepConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 10), bias=False)  # Temporal convolution
        self.conv2 = nn.Conv2d(25, 25, kernel_size=(Chans, 1), bias=False)  # Spatial convolution
        self.bn1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop1 = nn.Dropout(p=dropoutRate)

        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop2 = nn.Dropout(p=dropoutRate)

        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 10), bias=False)
        self.bn3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop3 = nn.Dropout(p=dropoutRate)

        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 10), bias=False)
        self.bn4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop4 = nn.Dropout(p=dropoutRate)

        # Final number of features before the fully connected layer
        # Actual flattened size: 1600, so update fc1 accordingly
        self.fc1 = nn.Linear(200 * 8, nb_classes)

        # LogSoftmax layer for KL-divergence loss
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = F.elu(self.bn1(self.conv2(self.conv1(x))))
        #print(f"After conv1 and conv2 shape: {x.shape}")
        x = self.drop1(self.pool1(x))
        #print(f"After pool1 shape: {x.shape}")
        x = F.elu(self.bn2(self.conv3(x)))
        #print(f"After conv3 shape: {x.shape}")
        x = self.drop2(self.pool2(x))
        #print(f"After pool2 shape: {x.shape}")
        x = F.elu(self.bn3(self.conv4(x)))
        #print(f"After conv4 shape: {x.shape}")
        x = self.drop3(self.pool3(x))
        #print(f"After pool3 shape: {x.shape}")
        x = F.elu(self.bn4(self.conv5(x)))
        #print(f"After conv5 shape: {x.shape}")
        x = self.drop4(self.pool4(x))
        #print(f"After pool4 shape: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten
        #print(f"Flattened output shape: {x.shape}")  # Check the shape of the flattened output
        x = self.fc1(x)
        x = self.log_softmax(x)  # Apply LogSoftmax for KL-divergence loss
        return x

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
        


class EEGNetResidual(nn.Module):
    def __init__(self, nb_classes, Chans=37, Samples=3000, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNetResidual, self).__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Add L2 regularization (weight decay)
        self.weight_decay = 1e-3 # Example value, tune as necessary

        # Residual connection layers with appropriate downsampling
        self.residual_conv = nn.Conv2d(F1 * D, F2, kernel_size=1, stride=(1, 2), bias=False)  # Adjust stride
        self.residual_batchnorm = nn.BatchNorm2d(F2)
        self.residual_pool = nn.AvgPool2d((1, 4))  # Additional pooling to match dimensions

        # Classification layer
        output_samples = Samples // 32
        final_output_size = F2 * output_samples
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(final_output_size, nb_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        # Save the residual before block 2 and downsample
        residual = x

        # Block 2
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        # Apply 1x1 convolution, batch norm, and additional pooling to the residual to match the dimensions
        residual = self.residual_conv(residual)
        residual = self.residual_batchnorm(residual)
        residual = self.residual_pool(residual)

        # Add the residual
        x = x + residual

        # Classification
        x = self.flatten(x)
        x = self.dense(x)
        x = self.log_softmax(x)
        return x

class EEGNetResidualLSTM(nn.Module):
    def __init__(self, nb_classes, Chans=37, Samples=3000, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, LSTM_units=64, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNetResidualLSTM, self).__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        # Add L2 regularization (weight decay)
        self.weight_decay = 1e-4 # Example value, tune as necessary
        self.Samples = Samples

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Residual connection layers with appropriate downsampling
        self.residual_conv = nn.Conv2d(F1 * D, F2, kernel_size=1, stride=(1, 2), bias=False)  # Adjust stride
        self.residual_batchnorm = nn.BatchNorm2d(F2)
        self.residual_pool = nn.AvgPool2d((1, 4))  # Additional pooling to match dimensions

        # LSTM layer
        self.lstm = nn.LSTM(input_size=F2, hidden_size=LSTM_units, batch_first=True)

        # Classification layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(LSTM_units * (Samples // 32), nb_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        # Save the residual before block 2 and downsample
        residual = x

        # Block 2
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        # Apply 1x1 convolution, batch norm, and additional pooling to the residual to match the dimensions
        residual = self.residual_conv(residual)
        residual = self.residual_batchnorm(residual)
        residual = self.residual_pool(residual)

        # Add the residual
        x = x + residual

        # Prepare for LSTM
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # Change shape to (batch, height, width, channels)
        x = x.view(b, h * w, c)  # Flatten height and width to fit into LSTM input

        # LSTM layer
        x, _ = self.lstm(x)  # Only keep the output, ignore the hidden states

        # Classification
        x = x.contiguous().view(b, -1)  # Flatten the LSTM output
        x = self.dense(x)
        x = self.log_softmax(x)
        return x


class EEGNetTransformer(nn.Module):
    def __init__(self, nb_classes, Chans=37, Samples=3000, 
                 dropoutRate=0.5, kernLength=64, F1=16, 
                 D=4, F2=32, num_heads=8, num_transformer_layers=4,
                 norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNetTransformer, self).__init__()
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples

        # Block 1 - Added another convolution layer for more feature extraction
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwiseConv1 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 2 - Added another separable convolution layer for more complex feature extraction
        self.separableConv1 = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 3 - Another separable convolution layer for further processing
        self.separableConv2 = nn.Conv2d(F2, F2 * 2, (1, 8), padding='same', bias=False)
        self.batchnorm4 = nn.BatchNorm2d(F2 * 2)
        self.activation3 = nn.ELU()
        self.avg_pool3 = nn.AvgPool2d((1, 4))
        self.dropout3 = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Flatten the output after the convolutions
        output_samples = Samples // 64
        self.flatten = nn.Flatten()

        # Transformer encoder layer (Replaces the residual connection)
        d_model = F2 * 2 * output_samples  # Input size to Transformer
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropoutRate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

        # Fully connected layers after Transformer
        self.dense1 = nn.Linear(d_model, 256)
        self.dense2 = nn.Linear(256, 128)
        self.fc_output = nn.Linear(128, nb_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv1(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.separableConv1(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.separableConv2(x)
        x = self.batchnorm4(x)
        x = self.activation3(x)
        x = self.avg_pool3(x)
        x = self.dropout3(x)

        # Flatten for Transformer input
        b, c, h, w = x.shape
        x = x.view(b, c * h * w)  # Flatten the spatial dimensions

        # Transformer layer
        x = x.unsqueeze(1)  # Add sequence length dimension (required by transformer)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)  # Remove sequence length dimension

        # Fully connected layers
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.fc_output(x)
        x = self.log_softmax(x)

        return x

class EEGSeizureDetectionModel(nn.Module):
    def __init__(self, nb_classes=6, Chans=37, Samples=3000, dropoutRate=0.5):
        super(EEGSeizureDetectionModel, self).__init__()
        
        # 1D Convolution Layer: For spatial feature extraction across EEG channels
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))  # Conv across samples with kernel size 64
        self.batchnorm1 = nn.BatchNorm2d(16)  # Normalize after conv layer
        self.activation1 = nn.ELU()  # Activation function
        self.pool1 = nn.AvgPool2d((1, 4))  # Reduce sample size, keep channel size
        
        # Second 1D Convolution Layer for more abstract spatial features
        self.conv2 = nn.Conv2d(16, 32, (1, 32), padding=(0, 16))
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 4))  # Pooling layer

        # Flattening for feeding into Bidirectional LSTM
        self.flatten_size = Chans * (Samples // 16) * 32
        self.flatten = nn.Flatten()

        # Bi-directional LSTM for temporal sequence learning
        self.lstm = nn.LSTM(input_size=self.flatten_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2, 64)  # LSTM output to FC
        self.dropout = nn.Dropout(dropoutRate)
        self.fc2 = nn.Linear(64, nb_classes)  # Final classification layer
        
    def forward(self, x):
        # 1D Convolution + Pooling Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.pool1(x)

        # 1D Convolution + Pooling Block 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.pool2(x)

        # Flatten for LSTM input
        b, c, h, w = x.shape  # (batch_size, channels, chans, samples)
        x = x.view(b, c * h * w)

        # LSTM layer
        x, (hn, cn) = self.lstm(x.unsqueeze(1))  # (batch_size, sequence_length, features)

        # Fully connected layers
        x = x[:, -1, :]  # Take output from last LSTM time step
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    
    
    
class SpectrogramViT(nn.Module):
    def __init__(self, image_size=(400, 300), num_classes=6):
        super(SpectrogramViT, self).__init__()
        
        # Initialize a pre-trained Vision Transformer with ImageNet weights
        self.vit = vit_b_16(pretrained=True)
        
        # Calculate the number of patches based on the new image size
        new_patch_size = (image_size[0] // 16, image_size[1] // 16)
        num_patches = new_patch_size[0] * new_patch_size[1]
        
        # Adjust the positional embeddings to match the new number of patches
        self.vit.encoder.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, 768)
        )
        
        # Get the input features from the classifier head
        in_features = self.vit.heads[0].in_features
        
        # Adjust the classification head to output logits for KLDivLoss
        self.vit.heads = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for KLDivLoss
        )

    def forward(self, x):
        return self.vit(x)
    
    
    

"""



Dilated Inception WaveNet

 
""" 

    
    
    
    
    
    
class _WaveBlock(nn.Module):
    """WaveNet block.

    Args:
        kernel_size: kernel size, pass a list of kernel sizes for
            inception
    """

    def __init__(
        self,
        n_layers: int, 
        in_dim: int,
        h_dim: int,
        kernel_size: Union[int, List[int]],
        conv_module: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.dilation_rates = [2**l for l in range(n_layers)]

        self.in_conv = nn.Conv2d(in_dim, h_dim, kernel_size=(1, 1)) 
        self.gated_tcns = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for layer in range(n_layers):
            c_in, c_out = h_dim, h_dim
            self.gated_tcns.append(
                _GatedTCN(
                    in_dim=c_in,
                    h_dim=c_out,
                    kernel_size=kernel_size,
                    dilation_factor=self.dilation_rates[layer],
                    conv_module=conv_module,
                )
            )
            self.skip_convs.append(nn.Conv2d(h_dim, h_dim, kernel_size=(1, 1)))

        # Initialize parameters
        nn.init.xavier_uniform_(self.in_conv.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.in_conv.bias)
        for i in range(len(self.skip_convs)):
            nn.init.xavier_uniform_(self.skip_convs[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(self.skip_convs[i].bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Shape:
            x: (B, C, N, L), where C denotes in_dim
            x_skip: (B, C', N, L), where C' denotes h_dim
        """
        # Input convolution
        x = self.in_conv(x)

        x_skip = x
        for layer in range(self.n_layers):
            x = self.gated_tcns[layer](x)
            x = self.skip_convs[layer](x)

            # Skip-connection
            x_skip = x_skip + x 

        return x_skip


class _GatedTCN(nn.Module):
    """Gated temporal convolution layer.

    Parameters:
        conv_module: customized convolution module
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        kernel_size: Union[int, List[int]],
        dilation_factor: int,
        dropout: Optional[float] = None,
        conv_module: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        # Model blocks
        if conv_module is None:
            self.filt = nn.Conv2d(
                in_channels=in_dim, out_channels=h_dim, kernel_size=(1, kernel_size), dilation=dilation_factor
            )
            self.gate = nn.Conv2d(
                in_channels=in_dim, out_channels=h_dim, kernel_size=(1, kernel_size), dilation=dilation_factor
            )
        else:
            self.filt = conv_module(
                in_channels=in_dim, out_channels=h_dim, kernel_size=kernel_size, dilation=dilation_factor
            )
            self.gate = conv_module(
                in_channels=in_dim, out_channels=h_dim, kernel_size=kernel_size, dilation=dilation_factor
            )

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L), where L denotes the input sequence length
            h: (B, h_dim, N, L')
        """
        x_filt = F.tanh(self.filt(x))
        x_gate = F.sigmoid(self.gate(x))
        h = x_filt * x_gate
        if self.dropout is not None:
            h = self.dropout(h)

        return h


class _DilatedInception(nn.Module):
    """Dilated inception layer.

    Note that `out_channels` will be split across #kernels.
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: List[int], 
        dilation: int
    ) -> None:
        super().__init__()

        # Network parameters
        n_kernels = len(kernel_size)
        assert out_channels % n_kernels == 0, "`out_channels` must be divisible by #kernels."
        h_dim = out_channels // n_kernels

        # Model blocks
        self.convs = nn.ModuleList()
        for k in kernel_size:
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=h_dim, 
                    kernel_size=(1, k),
                    padding="same",
                    dilation=dilation),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L), where C = in_channels
            h: (B, C', N, L'), where C' = out_channels
        """
        x_convs = []
        for conv in self.convs:
            x_conv = conv(x)
            x_convs.append(x_conv)
        h = torch.cat(x_convs, dim=1)

        return h
    
    
    
class DilatedInceptionWaveNet(nn.Module):
    """WaveNet architecture with dilated inception conv."""

    def __init__(self,) -> None:
        super().__init__()

        kernel_size = [2, 3, 6, 7]

        # Model blocks 
        self.wave_module = nn.Sequential(
            _WaveBlock(12, 1, 16, kernel_size, _DilatedInception),
            _WaveBlock(8, 16, 32, kernel_size, _DilatedInception),
            _WaveBlock(4, 32, 64, kernel_size, _DilatedInception),
            _WaveBlock(1, 64, 64, kernel_size, _DilatedInception),
        )
        self.output = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, CFG.N_CLASSES)
        ) 

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, L, C)
        """
        x = inputs["x"]
        bs, length, in_dim = x.shape
        x = x.transpose(1, 2).unsqueeze(dim=2)  # (B, C, N, L), N is redundant

        x_ll_1 = self.wave_module(x[:, 0:1, :])
        x_ll_2 = self.wave_module(x[:, 1:2, :])
        x_ll = (F.adaptive_avg_pool2d(x_ll_1, (1, 1)) + F.adaptive_avg_pool2d(x_ll_2, (1, 1))) / 2

        x_rl_1 = self.wave_module(x[:, 2:3, :])
        x_rl_2 = self.wave_module(x[:, 3:4, :])
        x_rl = (F.adaptive_avg_pool2d(x_rl_1, (1, 1)) + F.adaptive_avg_pool2d(x_rl_2, (1, 1))) / 2

        x_lp_1 = self.wave_module(x[:, 4:5, :])
        x_lp_2 = self.wave_module(x[:, 5:6, :])
        x_lp = (F.adaptive_avg_pool2d(x_lp_1, (1, 1)) + F.adaptive_avg_pool2d(x_lp_2, (1, 1))) / 2

        x_rp_1 = self.wave_module(x[:, 6:7, :])
        x_rp_2 = self.wave_module(x[:, 7:8, :])
        x_rp = (F.adaptive_avg_pool2d(x_rp_1, (1, 1)) + F.adaptive_avg_pool2d(x_rp_2, (1, 1))) / 2

        x = torch.cat([x_ll, x_rl, x_lp, x_rp], axis=1).reshape(bs, -1)
        output = self.output(x)

        return output
