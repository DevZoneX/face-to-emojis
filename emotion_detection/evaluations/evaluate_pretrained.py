import os
import cv2
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from deepface import DeepFace

test_data_path = os.path.join(os.getcwd(), 'data', 'test')
emotion_labels = ['happy', 'sad', 'anger', 'disgust', 'neutral', 'surprise', 'fear']

true_labels = []
predicted_labels = []

correct_counts = defaultdict(int)
total_counts = defaultdict(int)

# Loop through each emotion folder
for emotion in emotion_labels:

    emotion_folder = os.path.join(test_data_path, emotion)

    for img_name in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, img_name)

        img = cv2.imread(img_path)

        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        predicted_emotion = result[0]['dominant_emotion'].lower()

        # Append true and predicted labels for evaluation
        true_labels.append(emotion)
        predicted_labels.append(predicted_emotion)

        # Update counts for accuracy
        if predicted_emotion == emotion:
            correct_counts[emotion] += 1
        total_counts[emotion] += 1

# Compute and display per-emotion accuracy
print("\nPer-Emotion Accuracy:")
for emotion in emotion_labels:
    if total_counts[emotion] > 0:
        accuracy = correct_counts[emotion] / total_counts[emotion] * 100
        print(f"Accuracy for {emotion}: {accuracy:.2f}%")
    else:
        print(f"No images found for {emotion}.")

# Compute overall accuracy
global_accuracy = sum(correct_counts.values()) / sum(total_counts.values()) * 100
print(f"\nOverall Accuracy: {global_accuracy:.2f}%")

# Compute confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels, labels=emotion_labels)
print(cm)

# Compute F1-score, precision, and recall
print("\nClassification Report:")
report = classification_report(true_labels, predicted_labels, labels=emotion_labels, target_names=emotion_labels)
print(report)