import cv2

# Check camera indices from 0 to 5
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
    else:
        print(f"No camera at index {i}")
