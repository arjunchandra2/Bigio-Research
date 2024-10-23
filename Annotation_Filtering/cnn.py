import torch
import torch.nn as nn

class FilteringCNN(nn.Module):
    def __init__(self, input_size=48, input_channels=3):
        super(FilteringCNN, self).__init__()
        
        # Convolutional layers for feature extraction
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x H/2 x W/2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x H/4 x W/4
            nn.ReLU(inplace=True),
            nn.Flatten(),  
            nn.Linear(64 * (input_size // 4) * (input_size // 4), 1),  # Final layer for binary output
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        return self.network(x)
