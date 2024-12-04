import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # Convolutional layers with batch normalization and dropout
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Output: 64x48x48
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128x48x48
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 128x24x24
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Output: 256x24x24
        self.bn3 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 256x12x12
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Output: 512x12x12
        self.bn4 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: 512x6x6

        # Adding another convolutional layer (conv5)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # Output: 1024x6x6
        self.bn5 = nn.BatchNorm2d(1024)
        self.pool4 = nn.MaxPool2d(2, 2)  # Output: 1024x3x3

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: 1024x1x1

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 8)  # 8 emotion classes

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)  # Dropout with probability of 0.5

    def forward(self, x):
        # Apply convolutional layers with batch normalization and ReLU activations
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)

        # Apply new convolutional layer with batch normalization
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool4(x)  # Apply pooling after the new convolution

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(-1, 1024)  # Flatten the output to (batch_size, 1024)
        
        # Fully connected layer with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first fully connected layer
        x = self.fc2(x)
        
        return x

