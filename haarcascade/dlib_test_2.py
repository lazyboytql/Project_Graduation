import dlib
import cv2
import numpy as np
import pickle

class FaceRecognition:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(r'D:\shape_predictor_68_face_landmarks.dat')
        self.face_recognition_model = dlib.face_recognition_model_v1(r'D:\dlib_face_recognition_resnet_model_v1.dat')
        self.known_faces = []

    def calculate_face_descriptor(self, frame, face):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.shape_predictor(gray, face)
        return self.face_recognition_model.compute_face_descriptor(frame, shape)
    def recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            face_descriptor = self.calculate_face_descriptor(frame, face)

            match = self.match_face(face_descriptor)

            if match is None:
                self.known_faces.append({
                    'name': f'Linh_{len(self.known_faces) + 1}',
                    'descriptor': face_descriptor,
                    'additional_info': {
                        'range_min': 0.002,
                        'range_max': 0.1
                    }
                })

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        return frame

    def match_face(self, current_descriptor):
        for known_face in self.known_faces:
            distance = np.linalg.norm(np.array(known_face['descriptor']) - np.array(current_descriptor))
            if known_face['additional_info']['range_min'] < distance < known_face['additional_info']['range_max']:
                return known_face['name']
        return None

    def display_recognition_info(self):
        print("Danh sách mô tả khuôn mặt:")
        for i, face_info in enumerate(self.known_faces):
            print(f"Người {i + 1}: {face_info['descriptor']} | Name: {face_info['name']} | Range: {face_info['additional_info']['range_min']} - {face_info['additional_info']['range_max']}")

    def save_known_faces(self, filename='Linh.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self.known_faces, file)

if __name__ == "__main__":
    face_recognition = FaceRecognition()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = face_recognition.recognize_faces(frame)
        face_recognition.display_recognition_info()
        cv2.imshow('Face_Recognition_Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_recognition.save_known_faces()
    cap.release()
    cv2.destroyAllWindows()
