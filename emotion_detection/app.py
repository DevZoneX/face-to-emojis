from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from model_utils.model import EmotionCNN
from deepface import DeepFace

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = "trained"

emotion_labels = ["Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera_index = 0
current_emotion = {}

trained_model = EmotionCNN().to(device)
trained_model.load_state_dict(torch.load('models/emotion_cnn.pth', map_location=device, weights_only=True))
trained_model.eval()

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
        
        return {emotion_labels[i]: round(probabilities[0, i].item() * 100, 2) for i in range(len(emotion_labels))}
    
    elif model_type == "pretrained":
        if len(face_roi.shape) == 2:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
    
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)[0]['emotion']
        emotion_data = {emotion_labels[i]: float(round(result[emotion_labels[i].lower()] , 2)) for i in range(len(emotion_labels))}

        return emotion_data



def generate_frames():
    global current_emotion
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Error: Could not access camera with index {camera_index}")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w] if model_type == "pretrained" else gray[y:y+h, x:x+w]
                
                current_emotion = predict_emotion(face_roi)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_emojis():
    global current_emotion

    emoji_map = {
        "Angry": "😡",
        "Contempt": "😒",
        "Disgust": "🤢",
        "Fear": "😨",
        "Happy": "😊",
        "Neutral": "😐",
        "Sad": "😢",
        "Surprise": "😲"
    }

    suggested_emojis = []

    for emotion in emotion_labels:
        if emotion in current_emotion and current_emotion[emotion] > 50:
            suggested_emojis.append({
                "name": emotion,
                "emoji": emoji_map.get(emotion, "")
            })
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

if __name__ == '__main__':
    app.run(debug=True, port=5002)