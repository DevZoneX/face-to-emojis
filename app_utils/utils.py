import numpy as np
import mediapipe as mp


def compute_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def analyze_face_landmarks(image, landmarks):
    MOUTH_POINTS = [13, 14, 78, 308]
    EYE_POINTS = {
        "left": [145, 159],
        "right": [374, 386]
    }
    EYEBROW_POINTS = [55, 105]

    h, w, _ = image.shape
    results = {}

    coords = [(int(landmark.x * w), int(landmark.y * h))
              for landmark in landmarks.landmark]

    mouth_opening = compute_distance(
        coords[MOUTH_POINTS[0]], coords[MOUTH_POINTS[1]])
    results["mouth_opening"] = mouth_opening

    left_eye_opening = compute_distance(
        coords[EYE_POINTS["left"][0]], coords[EYE_POINTS["left"][1]])
    right_eye_opening = compute_distance(
        coords[EYE_POINTS["right"][0]], coords[EYE_POINTS["right"][1]])
    results["eye_opening"] = {
        "left": left_eye_opening, "right": right_eye_opening}

    smile_width = compute_distance(
        coords[MOUTH_POINTS[2]], coords[MOUTH_POINTS[3]])
    results["smile_width"] = smile_width

    return results


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)


def get_vector_measures(frame):
    result = face_mesh.process(frame)
    vector_measure = [10, 9, 9, 40]

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        analysis = analyze_face_landmarks(frame, face_landmarks)
        vector_measure = [analysis['mouth_opening'], analysis['eye_opening']['left'],
                          analysis['eye_opening']['right'], analysis['smile_width']]

    return vector_measure
