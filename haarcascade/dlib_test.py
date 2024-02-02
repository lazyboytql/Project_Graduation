import dlib
import cv2
import numpy as np
import pickle

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(r'D:\shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(r'D:\dlib_face_recognition_resnet_model_v1.dat')


known_faces = []


def recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = shape_predictor(gray, face)
        face_descriptor = face_recognition_model.compute_face_descriptor(frame, shape)

  
        known_faces.append({
            'name': f'Linh_{len(known_faces) + 1}',
            'descriptor': face_descriptor,
            'additional_info': {
                'range_min': 0.002,
                'range_max': 0.1
            }
        })

      
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = recognize_faces(frame)
    cv2.imshow('Face_Recognition_Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Danh sách mô tả khuôn mặt:")
for i, face_info in enumerate(known_faces):
    print(f"Người {i + 1}: {face_info['descriptor']} | Name: {face_info['name']} | Range: {face_info['additional_info']['range_min']} - {face_info['additional_info']['range_max']}")

with open('Linh.pkl', 'wb') as file:
    pickle.dump(known_faces, file)

cap.release()
cv2.destroyAllWindows()
