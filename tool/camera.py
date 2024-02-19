import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Camera index {i}: Not OK")
    else:
        print(f"Camera index {i}: OK")
    cap.release()
