import cv2
import numpy as np
import pickle
import ctypes
import psycopg2
import dlib 

notification_displayed = False


def display_notification(name, class_):
    ctypes.windll.user32.MessageBoxW(0, f"Chấm công thành công!\nHọ và Tên: {name}\nLớp: {class_}", "Thông báo", 1)
    global notification_displayed
    notification_displayed = True


conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="Nokia_2730",
    host="localhost",
    port="5433"
)


cur = conn.cursor()


def add_data_to_database(name, class_):
    sql = "INSERT INTO check_in_history (name, class) VALUES (%s, %s)"
    cur.execute(sql, (name, class_))
    conn.commit()

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(r'D:\shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(r'D:\dlib_face_recognition_resnet_model_v1.dat')

known_faces = []

for pkl_file in ['Linh.pkl', 'Thinh.pkl']:
    with open(pkl_file, 'rb') as file:
        face_data = pickle.load(file)
        known_faces.extend(face_data)

def recognize_faces(frame):
    global notification_displayed

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = shape_predictor(gray, face)
        face_descriptor = face_recognition_model.compute_face_descriptor(frame, shape)

        known_faces_array = np.array([np.array(descriptor['descriptor']) for descriptor in known_faces])
        match = np.argmin(np.linalg.norm(known_faces_array - face_descriptor, axis=1))

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, known_faces[match]['name'], (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not notification_displayed and match is not None:
            display_notification(known_faces[match]['name'], known_faces[match]['class'])
            add_data_to_database(known_faces[match]['name'], known_faces[match]['class'])

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

cur.close()
conn.close()
cap.release()
cv2.destroyAllWindows()
