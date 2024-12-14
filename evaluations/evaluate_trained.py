import torch
from sklearn.metrics import confusion_matrix, classification_report
from model_utils.preprocess import get_test_loader
from model_utils.model import EmotionCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loader(test_batch_size=64)

model = EmotionCNN().to(device)
model.load_state_dict(torch.load('models/emotion_cnn.pth', map_location=device))

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    return all_labels, all_predictions, accuracy

print('Evaluating the model...')
all_labels, all_predictions, accuracy = evaluate_model(model, test_loader)

emotion_labels = test_loader.dataset.classes

print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_predictions)
print(cm)

print("\nClassification Report:")
report = classification_report(all_labels, all_predictions, target_names=emotion_labels)
print(report)