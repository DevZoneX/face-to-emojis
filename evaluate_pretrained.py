import os
import cv2
import numpy as np
from collections import defaultdict
from deepface import DeepFace

test_data_path = 'data3/test'
emotion_labels = ['happy', 'sad', 'angry', 'neutral', 'disgust', 'neutral', 'surprise', 'fear']

correct_counts = defaultdict(int)
total_counts = defaultdict(int)

for emotion in emotion_labels:
    emotion_folder = os.path.join(test_data_path, emotion)
    
    for img_name in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, img_name)
        img = cv2.imread(img_path)
    
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        predicted_emotion = result[0]['dominant_emotion']

        if predicted_emotion.lower() == emotion:
            correct_counts[emotion] += 1
        total_counts[emotion] += 1

for emotion in emotion_labels:
    if total_counts[emotion] > 0:
        accuracy = correct_counts[emotion] / total_counts[emotion] * 100
        print(f"Accuracy for {emotion}: {accuracy:.2f}%")
    else:
        print(f"No images found for {emotion}.")

global_accuracy = sum(correct_counts.values()) / sum(total_counts.values()) * 100
print(f"Overall accuracy: {global_accuracy:.2f}%")