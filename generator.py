# Define models
import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_downsampling=True, add_activation=True, **kwargs):
        super().__init__()
        if is_downsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(channels, channels, add_activation=True, kernel_size=3, padding=1),
            ConvolutionalBlock(channels, channels, add_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=6):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.downsampling_layers = nn.ModuleList([
            ConvolutionalBlock(num_features, num_features * 2, is_downsampling=True, kernel_size=3, stride=2, padding=1),
            ConvolutionalBlock(num_features * 2, num_features * 4, is_downsampling=True, kernel_size=3, stride=2, padding=1),
        ])
        self.residual_layers = nn.Sequential(*[ResidualBlock(num_features * 4) for _ in range(num_residuals)])
        self.upsampling_layers = nn.ModuleList([
            ConvolutionalBlock(num_features * 4, num_features * 2, is_downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvolutionalBlock(num_features * 2, num_features * 1, is_downsampling=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])
        self.last_layer = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = self.residual_layers(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        return torch.sigmoid(self.last_layer(x))