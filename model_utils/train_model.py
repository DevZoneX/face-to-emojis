import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import EmotionCNN
from preprocess import get_train_loader

# Load the training dataset
print('Loading data...')
train_loader = get_train_loader(train_batch_size=64)

# Instantiate the model, define the loss and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to train the model


def train_model(model, train_loader, criterion, optimizer, num_epochs=30):
    loss_history = []  # List to store loss values
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
        loss_history.append(epoch_loss)  # Append the loss to the history
        print(
            f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    return loss_history  # Return the loss history for saving

# Method to save the model checkpoint


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Method to load the model checkpoint


def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.to(device)
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}, starting training from scratch.")

# Method to save loss history to a file


def save_loss_history(loss_history, file_path):
    with open(file_path, 'w') as f:
        for loss in loss_history:
            f.write(f"{loss}\n")
    print(f"Loss history saved to {file_path}")

# Main logic: choose to train from scratch or load a pre-trained model


def main():
    model_path = 'models/emotion_cnn.pth'
    loss_history_file = 'monitor/train_loss_history.txt'  # File to save loss history

    # Ask the user whether to load a saved model or train from scratch
    load_pretrained = input(
        "Do you want to load the pre-trained model and continue training? (yes/no): ").strip().lower()

    if load_pretrained == 'yes':
        load_model(model, model_path)
    else:
        print("Starting training from scratch...")

    # Train the model
    print('Training the model...')
    loss_history = train_model(
        model, train_loader, criterion, optimizer, num_epochs=10)

    # Save the model
    save_model(model, model_path)

    # Save the loss history to a file
    save_loss_history(loss_history, loss_history_file)


if __name__ == "__main__":
    main()
