import dlib
import cv2
import numpy as np
import pickle

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(r'D:\shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(r'D:\dlib_face_recognition_resnet_model_v1.dat')

# Khởi tạo danh sách để lưu thông tin về các khuôn mặt đã biết
known_faces = []

# Thêm info từng file pkl của từng khuôn mặt
for pkl_file in ['Linh.pkl']:
    with open(pkl_file, 'rb') as file:
        face_data = pickle.load(file)
        known_faces.extend(face_data)

# Hàm nhận diện 
def recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = shape_predictor(gray, face)
        face_descriptor = face_recognition_model.compute_face_descriptor(frame, shape)

        # known_faces_array có hình dạng (N, 128)
        known_faces_array = np.array([np.array(descriptor['descriptor']) for descriptor in known_faces])
        # So sánh với khuôn mặt đã biết
        match = np.argmin(np.linalg.norm(known_faces_array - face_descriptor, axis=1))
        # Vẽ hình chữ nhật xung quanh khuôn mặt và hiển thị tên
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, known_faces[match]['name'], (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = recognize_faces(frame)
    cv2.imshow('Face_recognition_main', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
