import torch
from model import EmotionCNN
from preprocess import get_test_loader
from model import EmotionCNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_loader = get_test_loader(test_batch_size=64)

model = EmotionCNN().to(device)
model.load_state_dict(torch.load('models/emotion_cnn.pth'))

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Evaluate the model
print('Evaluating the model...')
evaluate_model(model, test_loader)