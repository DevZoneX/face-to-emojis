import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1x48x48, Output: 32x48x48
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x24x24
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128x12x12
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)  # Flatten layer, 128 feature maps of size 6x6
        self.fc2 = nn.Linear(512, 8)  # 7 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
