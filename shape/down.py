import urllib.request

# Define the URL of the model file and the destination path
model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
model_path = "C:/Users/gbrlg/Desktop/Cours/INF CV/Projet/face_landmarker_v2_with_blendshapes.task"

# Download the file
urllib.request.urlretrieve(model_url, model_path)
print("Model downloaded successfully.")
