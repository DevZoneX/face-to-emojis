import torch
import torch.nn as nn
import torch.optim as optim
from model import EmotionCNN
from preprocess import get_train_loader


# Load the training and test datasets
print('Loading data...')
train_loader, test_loader = get_train_loader(train_batch_size=8)


# Instantiate the model, define the loss and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, criterion, optimizer, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# Train the model
print('Training the model...')
train_model(model, train_loader, criterion, optimizer, num_epochs=1)

# Save the model
torch.save(model.state_dict(), 'models/'+'emotion_cnn.pth')