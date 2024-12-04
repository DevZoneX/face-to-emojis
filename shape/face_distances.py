import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datasets import load_dataset
import json

# Initialisation du module FaceMesh de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)


ds = load_dataset("valhalla/emoji-dataset")
emoji_of_interest = [0, 2, 113, 1335, 1446, 2006, 2017,
                     2039, 2050, 2072, 2350, 2416, 2450, 2539, 2594]
image = [ds['train'][i]['image'] for i in emoji_of_interest]


df_measure_dist = {
    "idx": [0, 2, 113, 1335, 1446, 2006, 2017, 2039, 2050, 2072, 2350, 2416, 2450, 2539, 2594],
    "Mouth_Opening": [20, 0, 0, 20, 20, 0, 0, 0, 0, 15, 0, 20, 15, 0, 0],
    "Left_Eye_Opening": [9, 4, 9, 9, 9, 9, 4, 9, 9, 9, 9, 5, 10, 5, 9],
    "Right_Eye_Opening": [9, 9, 9, 9, 4, 9, 4, 9, 9, 9, 9, 5, 10, 5, 9],
    "Smile_Width": [60, 50, 55, 55, 55, 40, 40, 50, 40, 60, 45, 60, 40, 45, 45]}


df = pd.DataFrame(df_measure_dist)

data_matrix = df[["Mouth_Opening", "Left_Eye_Opening",
                  "Right_Eye_Opening", "Smile_Width"]].to_numpy()

# Points d'intérêt pour les analyses
MOUTH_POINTS = [13, 14, 78, 308]  # Lèvres supérieures et inférieures
EYE_POINTS = {
    "left": [145, 159],  # Paupière supérieure et inférieure de l'œil gauche
    "right": [374, 386]  # Paupière supérieure et inférieure de l'œil droit
}
EYEBROW_POINTS = [55, 105]  # Distance entre les sourcils


def calculate_distance(point1, point2):
    """Calcule la distance euclidienne entre deux points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))


def analyze_face_landmarks(image, landmarks):
    """Analyse les caractéristiques faciales basées sur les landmarks."""
    h, w, _ = image.shape  # Dimensions de l'image
    results = {}

    # Conversion des landmarks en coordonnées
    coords = [(int(landmark.x * w), int(landmark.y * h))
              for landmark in landmarks.landmark]

    # Ouverture de la bouche
    mouth_opening = calculate_distance(
        coords[MOUTH_POINTS[0]], coords[MOUTH_POINTS[1]])
    results["mouth_opening"] = mouth_opening

    # Ouverture des yeux
    left_eye_opening = calculate_distance(
        coords[EYE_POINTS["left"][0]], coords[EYE_POINTS["left"][1]])
    right_eye_opening = calculate_distance(
        coords[EYE_POINTS["right"][0]], coords[EYE_POINTS["right"][1]])
    results["eye_opening"] = {
        "left": left_eye_opening, "right": right_eye_opening}

    # Sourire (distance entre les coins de la bouche)
    smile_width = calculate_distance(
        coords[MOUTH_POINTS[2]], coords[MOUTH_POINTS[3]])
    results["smile_width"] = smile_width

    # Sourcils froncés (distance entre les sourcils)
    eyebrow_distance = calculate_distance(
        coords[EYEBROW_POINTS[0]], coords[EYEBROW_POINTS[1]])
    results["eyebrow_distance"] = eyebrow_distance

    return results


# Traitement de la vidéo en direct (Webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Conversion en RGB (requis par MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            analysis = analyze_face_landmarks(frame, face_landmarks)
            mouth_opening = analysis['mouth_opening']
            left_eye_opening = analysis['eye_opening']['left']
            right_eye_opening = analysis['eye_opening']['right']
            smile_width = analysis['smile_width']
            eyebrow_distance = analysis['eyebrow_distance']

            vect_measure = [mouth_opening, left_eye_opening,
                            right_eye_opening, smile_width]

            distances = np.linalg.norm(data_matrix - vect_measure, axis=1)
            min_distance_idx = np.argmin(distances)
            # indice = df.loc[min_distance_idx, "idx"]
            # print(f"L'idx avec la plus petite distance est : {indice}")

            # Affichage des résultats
            cv2.putText(frame, f"Mouth opening: {mouth_opening:.2f}", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Left Eye: {left_eye_opening:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Eye: {right_eye_opening:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Smile width: {smile_width:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Eyebrow distance: {eyebrow_distance:.2f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Image la plus proche', cv2.cvtColor(
                (np.array(image[min_distance_idx])), cv2.COLOR_RGB2BGR))

    # Affichage de l'image
    cv2.imshow('Face Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
