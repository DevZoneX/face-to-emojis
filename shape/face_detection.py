import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functions import create_detector, draw_landmarks_on_image, compute_edges, image_distance, mask_resized, hu_moment_similarity, shape_similarity, circular_mask
from datasets import load_dataset

ds = load_dataset("valhalla/emoji-dataset")
emoji_of_interest = [1, 3, 114, 669, 780, 1002, 1336, 1447, 1996, 2007, 2018,
                     2040, 2051, 2073, 2351, 2373, 2406, 2417, 2418, 2451, 2528, 2540, 2562, 2595]
emoji_of_interest = [i - 1 for i in emoji_of_interest]
image = [ds['train'][i]['image'] for i in emoji_of_interest]
image_edge = [compute_edges(ds['train'][i]['image'])
              for i in emoji_of_interest]
text = [ds['train'][i]['text'] for i in emoji_of_interest]
data = [(image_edge[i], text[i]) for i in range(len(emoji_of_interest))]

data_emoji = np.array(data, dtype=object)

shape = image_edge[0].shape

detector = create_detector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Échec de capture de l'image.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        annotated_frame, mask = draw_landmarks_on_image(
            rgb_frame, detection_result)

        face_mask_resized = mask_resized(mask, shape)
        # face_mask_resized = circular_mask(face_mask_resized)
        diff = np.array([hu_moment_similarity(image_edge[i], face_mask_resized)
                         for i in range(len(image_edge))])

        indice = np.argmin(diff)
        smallest_value = diff[indice]
        print(f"Valeur : {smallest_value:.2f}, Indice : {indice}")

        print()
        bgr_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Face Landmarks Annotated', bgr_annotated_frame)
        # cv2.imshow('Contours du Visage', mask)
        cv2.imshow('Contours du Visage Resized', face_mask_resized)
        cv2.imshow('Image la plus proche', cv2.cvtColor(
            (np.array(image[indice])), cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
