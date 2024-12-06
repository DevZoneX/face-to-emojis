from flask import Flask, render_template, Response, jsonify, request
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from model_utils.model import EmotionCNN
from deepface import DeepFace
import mediapipe as mp
import json
import pandas as pd
from datasets import load_dataset
import numpy as np

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = "trained"

emotion_labels = ["Angry", "Contempt", "Disgust",
                  "Fear", "Happy", "Neutral", "Sad", "Surprise"]


df = pd.read_csv("datasets/df_measure_dist.csv")
data_measures = df[["Mouth_Opening", "Left_Eye_Opening",
                    "Right_Eye_Opening", "Smile_Width"]].to_numpy()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

MOUTH_POINTS = [13, 14, 78, 308]
EYE_POINTS = {
    "left": [145, 159],
    "right": [374, 386]
}
EYEBROW_POINTS = [55, 105]

df_emotion_to_emoji = pd.read_csv("datasets/df_emotion_to_emoji.csv")
df_emotion = df_emotion_to_emoji.drop(columns=["emoji", "name"])
data_emotions = df_emotion.to_numpy()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera_index = 0
current_emotion = {}

trained_model = EmotionCNN().to(device)
trained_model.load_state_dict(torch.load(
    'models/emotion_all_cnn3.pth', map_location=device, weights_only=True))
trained_model.eval()


def calculate_distance(point1, point2):
    """Calcule la distance euclidienne entre deux points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def analyze_face_landmarks(image, landmarks):
    """Analyse les caractéristiques faciales basées sur les landmarks."""
    h, w, _ = image.shape
    results = {}

    coords = [(int(landmark.x * w), int(landmark.y * h))
              for landmark in landmarks.landmark]

    mouth_opening = calculate_distance(
        coords[MOUTH_POINTS[0]], coords[MOUTH_POINTS[1]])
    results["mouth_opening"] = mouth_opening

    left_eye_opening = calculate_distance(
        coords[EYE_POINTS["left"][0]], coords[EYE_POINTS["left"][1]])
    right_eye_opening = calculate_distance(
        coords[EYE_POINTS["right"][0]], coords[EYE_POINTS["right"][1]])
    results["eye_opening"] = {
        "left": left_eye_opening, "right": right_eye_opening}

    smile_width = calculate_distance(
        coords[MOUTH_POINTS[2]], coords[MOUTH_POINTS[3]])
    results["smile_width"] = smile_width

    return results


def load_model(model_type):
    """Selects the model based on user choice."""
    if model_type == "trained":
        return trained_model
    elif model_type == "pretrained":
        return None


def predict_emotion(face_roi):
    """Predicts emotion based on the selected model."""
    global model_type
    if model_type == "trained":
        face_img = Image.fromarray(face_roi).convert('L')
        face_tensor = transform(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = trained_model(face_tensor)
            probabilities = F.softmax(output, dim=1)

        return {emotion_labels[i]: round(probabilities[0, i].item() * 100, 2) for i in range(len(emotion_labels))}, probabilities.squeeze(0).numpy()

    elif model_type == "pretrained":
        if len(face_roi.shape) == 2:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        result = DeepFace.analyze(
            face_roi, actions=['emotion'], enforce_detection=False)[0]['emotion']
        emotion_data = {emotion_labels[i]: float(
            round(result[emotion_labels[i].lower()], 2)) for i in range(len(emotion_labels))}

        return None  # !!!!!!! il faut un vecteur de probabilité ici


def get_vector_measures(rgb_frame, frame):
    result = face_mesh.process(rgb_frame)
    vect_measure = [10, 9, 9, 40]

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        analysis = analyze_face_landmarks(frame, face_landmarks)
        vect_measure = [analysis['mouth_opening'], analysis['eye_opening']['left'],
                        analysis['eye_opening']['right'], analysis['smile_width']]

    return vect_measure


def generate_frames():
    global current_emotion
    global vector_measure
    global proba
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Error: Could not access camera with index {camera_index}")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x +
                                 w] if model_type == "pretrained" else gray[y:y+h, x:x+w]

                current_emotion, proba = predict_emotion(face_roi)
                vector_measure = get_vector_measures(rgb_frame, frame)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def get_emojis():
    global current_emotion
    global vector_measure
    global proba

    distances_measures = np.linalg.norm(data_measures - vector_measure, axis=1)
    distances_emotions = np.linalg.norm(data_emotions - proba, axis=1)

    proba_measures = np.exp(-distances_measures) / \
        np.sum(np.exp(-distances_measures))
    proba_emotions = np.exp(-distances_emotions) / \
        np.sum(np.exp(-distances_emotions))

    alpha_emotions = 0.5
    proba_final = alpha_emotions * proba_emotions + \
        (1 - alpha_emotions) * proba_measures

    idx = np.argmax(proba_final)

    suggested_emojis = [{
        "name": df_emotion_to_emoji.iloc[idx]['name'],
        "emoji": df_emotion_to_emoji.iloc[idx]['emoji']
    }]
    return suggested_emojis


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion_feed')
def emotion_feed():
    global current_emotion
    if current_emotion:

        # Add the suggested emojis to the response
        current_emotion['emojis'] = get_emojis()

        return jsonify(current_emotion)

    return jsonify({"error": "No emotion data available"})

@app.route('/set_model', methods=['POST'])
def set_model():
    global model_type
    data = request.get_json()

    if data and 'model_type' in data:
        model_type = data['model_type']
        return jsonify({"message": f"Model set to {model_type} successfully."})
    return jsonify({"error": "Invalid request"})


if __name__ == '__main__':
    app.run(debug=True, port=5002)
