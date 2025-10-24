import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import torch.nn as nn
import torchvision
import csv
import numpy as np
from torchsummary import summary
import torch.nn.init as init
import random
import cv2
import ast
import platform

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.decoders import NDArrayDecoder


############################################
#       MODELS ARCHITECTURES          #
############################################

#----------------- CBAM MODULE -----------------#
class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        reduced_planes = max(1, in_planes // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return attention * x


# ----------------- Large Autoencoder -----------------#

class LargeModel(nn.Module):
    def __init__(self):
        super(LargeModel, self).__init__()

        # Shared Encoder with CBAM
        self.encoder = nn.Sequential(
            nn.Conv2d(NUMBER_FRAMES_INPUT, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(512),
            nn.MaxPool2d(2, 2),
        )

        # 4 decoder heads with CBAM
        self.decoders = nn.ModuleList([LargeDecoder() for _ in range(len(FUTURE_TARGET_RILEVATION))])

    def forward(self, x):
        latent = self.encoder(x)
        return [decoder(latent) for decoder in self.decoders]
    
class LargeDecoder(nn.Module):
    def __init__(self):
        super(LargeDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(512),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(256),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(128),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(64),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(32),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(16),

            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Final prediction
        )

    def forward(self, x):
        return self.dec(x)

class MediumModel(nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()

        # Shared Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(NUMBER_FRAMES_INPUT, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(8),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(32),
            nn.MaxPool2d(2, 2),  # 1/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(64),
            nn.MaxPool2d(2, 2),  # 1/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(128),
            nn.MaxPool2d(2, 2),  # 1/8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(256),
            nn.MaxPool2d(2, 2),  # 1/16
        )

        # 4 decoder heads
        self.decoders = nn.ModuleList([MediumDecoder() for _ in range(len(FUTURE_TARGET_RILEVATION))])

    def forward(self, x):
        latent = self.encoder(x)
        return [decoder(latent) for decoder in self.decoders]
    
class MediumDecoder(nn.Module):
    def __init__(self):
        super(MediumDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(256),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(128),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(64),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(32),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(16),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Final output
        )

    def forward(self, x):
        return self.dec(x)

# ----------------- Small Autoencoder -----------------#
    
class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()

        # Reduced Shared Encoder with CBAM
        self.encoder = nn.Sequential(
            nn.Conv2d(NUMBER_FRAMES_INPUT, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(8),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(32),
            nn.MaxPool2d(2, 2),  # Down to 1/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(64),
            nn.MaxPool2d(2, 2),  # Down to 1/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(128),
            nn.MaxPool2d(2, 2),  # Down to 1/8
        )

        # 4 decoder heads with reduced CBAMDecoder
        self.decoders = nn.ModuleList([SmallDecoder() for _ in range(len(FUTURE_TARGET_RILEVATION))])

    def forward(self, x):
        latent = self.encoder(x)
        return [decoder(latent) for decoder in self.decoders]
    
class SmallDecoder(nn.Module):
    def __init__(self):
        super(SmallDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(128),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(64),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(32),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            CBAM(16),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Output layer
        )

    def forward(self, x):
        return self.dec(x)
    
# ----------------- Diffusion Model -----------------#

class DiffusionModel(nn.Module):
    def __init__(self, input_channels=(NUMBER_FRAMES_INPUT + 2), output_channels=1, features=[16, 32, 64, 128]):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        for feature in features:
            self.encoder.append(DoubleConv(input_channels, feature, nn.ReLU))
            input_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, nn.ReLU)

        # Decoder path
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature, nn.ReLU))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], output_channels, kernel_size=1)

    def forward(self, x_t, past_frames, timestep):
        x = torch.cat([x_t, past_frames, timestep], dim=1)  # Concatenate inputs along the channel dimension
        skip_connections = []

        # Encoder forward pass
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder forward pass
        skip_connections = skip_connections[::-1]  # Reverse skip connections
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Transposed convolution (upsampling)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = self._crop(skip_connection, x)  # Crop skip connection to match the size of x
            x = torch.cat((skip_connection, x), dim=1)  # Concatenate skip connection
            x = self.decoder[i + 1](x)  # DoubleConv layer

        return self.final_conv(x)  # Final output layer

    def _crop(self, skip, x):
        """Crop the skip connection to match the size of x."""
        _, _, h, w = x.shape
        skip = torchvision.transforms.functional.center_crop(skip, [h, w])
        return skip
    
class DoubleConv(nn.Module):
    """Two convolutional layers with batch normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels, activation_fn):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            activation_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            activation_fn()
        )
    
    def forward(self, x):
        return self.conv(x)