import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    height, width = rgb_image.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        # Dessin des landmarks colorés sur l'image annotée
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

        # Conversion des landmarks en coordonnées d'image
        landmarks = []
        for landmark in face_landmarks:
            x, y = int(landmark.x * width), int(landmark.y * height)
            landmarks.append((x, y))

        # Dessiner les contours du visage, des yeux, et de la bouche
        contours = [
            mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
            mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
            mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
            mp.solutions.face_mesh.FACEMESH_LIPS,
            mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,  # Sourcil gauche
            mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW  # Sourcil droit
        ]

        for contour in contours:
            for connection in contour:
                start_idx, end_idx = connection
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                cv2.line(mask, start_point, end_point, (255), 1)

    return annotated_image, mask


def create_detector():
    base_options = python.BaseOptions(
        model_asset_path='shape/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


def mask_resized(mask, shape):
    x, y, w, h = cv2.boundingRect(mask)
    mask_roi = mask[y:y+h, x:x+w]
    mask_resized = cv2.resize(mask_roi, shape)
    return mask_resized


def circular_mask(image):
    height, width = image.shape
    center = (width // 2, height // 2)
    # Limite le rayon pour que le visage reste dans les dimensions
    max_radius = min(center)

    # Appliquer la transformation polaire pour un effet de déformation circulaire
    polar_image = cv2.linearPolar(
        image, center, max_radius, cv2.WARP_FILL_OUTLIERS)

    # Re-transformation pour maintenir les traits mais avec une apparence plus ronde
    result = cv2.linearPolar(
        polar_image, center, max_radius, cv2.WARP_INVERSE_MAP)
    return result


def image_distance(face1, mask_resized):
    diff = cv2.absdiff(face1, mask_resized)
    euclidean_distance = np.sqrt(np.sum(diff**2))
    return euclidean_distance


def shape_similarity(image1, image2):
    _, binary1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    contours1, _ = cv2.findContours(
        binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(
        binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    similarity_score = cv2.matchShapes(
        contours1[0], contours2[0], cv2.CONTOURS_MATCH_I1, 0.0)
    return similarity_score


def hu_moment_similarity(image1, image2, threshold=0.1):
    _, binary1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    moments1 = cv2.HuMoments(cv2.moments(binary1)).flatten()
    moments2 = cv2.HuMoments(cv2.moments(binary2)).flatten()

    hu_distance = np.sum(np.abs(moments1 - moments2))
    return hu_distance


def compute_edges(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray_image, 100, 500)

    return edges
