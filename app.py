from flask import Flask, render_template, Response, jsonify, request
from app_utils.utils import get_vector_measures
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from model_utils.model import EmotionCNN
from deepface import DeepFace
import pandas as pd
import numpy as np
import cv2
import torch

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = "EmotionCNN"
emotion_labels = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad",
                  "Surprise"] if model_type == "EmotionCNN" else ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


df = pd.read_csv("datasets/df.csv")
measures_features = ["mouth_opening", "left_eye_opening",
                     "right_eye_opening", "smile_width"]
emotions_features = ["anger", "contempt", "disgust", "fear", "joy", "neutral", "sadness", "surprise"] if model_type == "EmotionCNN" else ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
data_measures = df[measures_features].to_numpy()
data_emotions = df[emotions_features].to_numpy()
vector_measure = [10, 9, 9, 40]
probas = np.array([0.125]*8) if model_type == "EmotionCNN" else np.array([0.125]*7)


def get_emojis():
    global current_emotion
    global vector_measure
    global probas

    distances_measures = np.linalg.norm(data_measures - np.array(vector_measure), axis=1)
    distances_emotions = np.linalg.norm(data_emotions - probas, axis=1)

    proba_measures = np.exp(distances_measures) / np.sum(np.exp(distances_measures))
    proba_emotions = np.exp(distances_emotions) / np.sum(np.exp(distances_emotions))

    alpha_emotions = 0.6
    proba_final = alpha_emotions * proba_emotions + (1 - alpha_emotions) * proba_measures

    idx = np.argmin(proba_final)

    max_values = {
        "mouth_opening": 60,
        "left_eye_opening": 25,
        "right_eye_opening": 25,
        "smile_width": 90 
    }

    suggested_emojis = [{
        "name": df.iloc[idx]['name'],
        "emoji": df.iloc[idx]['emoji'],
        "measures": {
            "mouth_opening": round((vector_measure[0] / max_values["mouth_opening"]) * 100, 2),
            "left_eye_opening": round((vector_measure[1] / max_values["left_eye_opening"]) * 100, 2),
            "right_eye_opening": round((vector_measure[2] / max_values["right_eye_opening"]) * 100, 2),
            "smile_width": round((vector_measure[3] / max_values["smile_width"]) * 100, 2),
        }
    }]


    return suggested_emojis



current_mode = "Camera"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

trained_model = EmotionCNN().to(device)
trained_model.load_state_dict(torch.load(
    'models/emotion_all_cnn3.pth', map_location=device, weights_only=True))
trained_model.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
current_emotion = {}

def predict_emotion(face_roi):
    """Predicts emotion based on the selected model."""
    global model_type

    if model_type == "EmotionCNN":
        # Process face ROI for EmotionCNN model
        face_img = Image.fromarray(face_roi).convert('L')
        face_tensor = transform(face_img).unsqueeze(0).to(device)
        with torch.no_grad():

            output = trained_model(face_tensor)
            probabilities = F.softmax(output, dim=1)

        return {emotion_labels[i]: round(probabilities[0, i].item() * 100, 2) for i in range(len(emotion_labels))}, probabilities.squeeze(0).numpy()

    elif model_type == "DeepFace":
        # Ensure the face ROI is in RGB format
        if len(face_roi.shape) == 2:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        # Analyze emotion using DeepFace
        result = DeepFace.analyze(
            face_roi, actions=['emotion'], enforce_detection=False)[0]['emotion']

        # Extract emotion probabilities
        emotion_data = {emotion_labels[i]: float(
            round(result[emotion_labels[i].lower()], 2)) for i in range(len(emotion_labels))}

        # Normalize the probabilities to create a vector (if necessary)
        probabilities = np.array([result[label.lower()]
                                 for label in emotion_labels])
        probabilities /= probabilities.sum()

        return emotion_data, probabilities


def generate_frames():
    """
    Capture frames from the camera, detect faces, and process emotion prediction for the most prominent face.
    """
    global current_emotion
    global vector_measure
    global probas
    camera_index = 0

    # Initialize camera
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Error: Could not access camera with index {camera_index}")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame from the camera.")
            break

        # Convert frame to RGB and grayscale for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4)

        if len(faces) > 0:
            # Choose the largest face based on area (w * h)
            largest_face = max(
                faces, key=lambda f: f[2] * f[3])  # (x, y, w, h)

            # Extract the largest face's coordinates
            x, y, w, h = largest_face

            # Get the face ROI (gray or color depending on the model)
            face_roi = frame[y:y+h, x:x + w] if model_type == "DeepFace" else gray[y:y+h, x:x+w]

            if current_mode == "Camera":
                # Predict emotion for the largest face
                current_emotion, probas = predict_emotion(face_roi)

                # Update vector measures (additional processing, if needed)

                vector_measure = get_vector_measures(rgb_frame, frame)

            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Encode frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode the frame to JPEG format.")
            continue

        frame = buffer.tobytes()

        # Yield the encoded frame for the stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    if current_mode == "Camera":
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion_feed', methods=['GET'])
def emotion_feed():
    global current_emotion

    current_emotion['emojis'] = get_emojis()

    return jsonify(current_emotion)


@app.route('/set_model', methods=['POST'])
def set_model():
    global model_type
    global emotion_labels
    global probas
    global emotions_features
    global data_emotions
    data = request.get_json()

    if data and 'model_type' in data:
        model_type = data['model_type']
        emotion_labels = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"] if model_type == "EmotionCNN" else [
            "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        
        probas = np.array([0.125]*8) if model_type == "EmotionCNN" else np.array([0.125]*7)
        emotions_features = ["anger", "contempt", "disgust", "fear", "joy", "neutral", "sadness", "surprise"] if model_type == "EmotionCNN" else ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        data_emotions = df[emotions_features].to_numpy()

        return jsonify({"message": f"Model set to {model_type} successfully."})
    return jsonify({"error": "Invalid request"})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    global current_emotion
    global probas
    global vector_measure

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Read uploaded file
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert the image for processing
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect emotion using existing model logic
        current_emotion, probas = predict_emotion(gray_img)
        vector_measure = get_vector_measures(img, img)

        current_emotion['emojis'] = get_emojis()

        return jsonify(current_emotion)

    return jsonify({"error": "File processing failed"}), 500


@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    data = request.get_json()
    if data and 'mode' in data:
        if data['mode'] == "Upload":
            # Stop streaming by signaling the route to stop
            current_mode = "Upload"
            return jsonify({"message": "Switched to Upload Mode"})
        elif data['mode'] == "Camera":
            current_mode = "Camera"
            return jsonify({"message": "Switched to Camera Mode"})
    return jsonify({"error": "Invalid request"}), 400


if __name__ == '__main__':
    app.run(port=5002, debug=False)