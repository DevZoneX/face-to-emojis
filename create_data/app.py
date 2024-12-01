import cv2
import mediapipe as mp
import numpy as np
import os
from functions import create_detector, draw_landmarks_on_image, mask_resized

# Create detector and video capture
detector = create_detector()
cap = cv2.VideoCapture(0)

# Shape for resizing the mask
shape = (256, 256)

# Folder where the images will be saved
save_folder = "captured_faces"
os.makedirs(save_folder, exist_ok=True)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

image_counter = 0  # Counter for naming saved images

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Échec de capture de l'image.")
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        # Draw landmarks and obtain the face mask
        annotated_frame, mask = draw_landmarks_on_image(rgb_frame, detection_result)

        # Resize the face mask
        face_mask_resized = mask_resized(mask, shape)
        
        # Convert back to BGR for OpenCV display
        bgr_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Display the results
        cv2.imshow('Face Landmarks Annotated', bgr_annotated_frame)
        cv2.imshow('Contours du Visage Resized', face_mask_resized)

    # Save image on space bar press
    if cv2.waitKey(1) & 0xFF == ord(' '):  # Spacebar key to save image
        image_filename = os.path.join(save_folder, f"face_{image_counter}.jpg")
        cv2.imwrite(image_filename, bgr_annotated_frame)  # Save the image
        print(f"Saved: {image_filename}")
        image_counter += 1

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
