from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from model import EmotionCNN  # Ensure your model class is in this file

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained emotion model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN().to(device)
model.load_state_dict(torch.load('models/emotion_cnn.pth', map_location=device, weights_only=True))
model.eval()

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Define image transformations (same as for the model input)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Initialize the OpenCV face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Specify the index of the external webcam (change based on your setup)
camera_index = 0  # Adjust this if needed

# Store current emotion data globally
current_emotion = {}  # Store the current detected emotion percentages

def predict_emotion(face_roi):
    """
    This function takes a face region of interest (ROI), processes it,
    and returns the predicted emotion percentages for all labels.
    """
    face_img = Image.fromarray(face_roi)  # Convert to PIL Image
    face_tensor = transform(face_img).unsqueeze(0).to(device)  # Transform and add batch dimension
    with torch.no_grad():
        output = model(face_tensor)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities

    emotion_percentages = {emotion_labels[i]: round(probabilities[0, i].item() * 100, 2) for i in range(len(emotion_labels))}
    return emotion_percentages  # Return percentages for all emotions

def generate_frames():
    global current_emotion  # Use global variable to update current emotion data

    # Open the external webcam
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        print(f"Error: Could not access camera with index {camera_index}")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # For each detected face, predict emotion percentages
            for (x, y, w, h) in faces:
                # Extract the face region of interest (ROI)
                face_roi = gray[y:y+h, x:x+w]

                # Predict emotion percentages
                current_emotion = predict_emotion(face_roi)  # Update current emotion

                # Draw a rectangle around the face (optional, you can remove this too if you don't want the rectangle)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Remove cv2.putText, so no text is displayed on the frame itself

            # Encode the frame into a JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


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
        return jsonify(current_emotion)
    return jsonify({"error": "No emotion data available"})


if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change port number here