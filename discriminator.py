import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.layer1 = nn.Sequential(  # (B, in_channels, H, W) → (B, 64, H/2, W/2)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(  # (B, 64, H/2, W/2) → (B, 128, H/4, W/4)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(  # (B, 128, H/4, W/4) → (B, 256, H/8, W/8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(  # (B, 256, H/8, W/8) → (B, 512, H/8, W/8)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")  # PatchGAN output

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x