##############################################
#             MODEL DEFINITION               #
##############################################

#Based on: https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf

import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, input_size=48, input_channels=3, feature_dim=256):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x H/2 x W/2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x H/4 x W/4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x H/8 x W/8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256 x H/16 x W/16
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=256 * (input_size//16) * (input_size//16), out_features=feature_dim) 
            #Activation function?
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256 * (input_size // 16) * (input_size // 16)),  # Input: feature_dim, Output: 256 x H/16 x W/16
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, input_size // 16, input_size // 16)),  # Reshape back to feature map
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 128 x H/8 x W/8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x H/4 x W/4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32 x H/2 x W/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: input_channels x H x W
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)