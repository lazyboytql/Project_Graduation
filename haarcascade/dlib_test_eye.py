import dlib
import cv2
import pickle

# Load the pre-trained facial landmark detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'D:\shape_predictor_68_face_landmarks.dat')


cap = cv2.VideoCapture(0)


eye_parameters = {}

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for i, face in enumerate(faces):

        shape = predictor(gray, face)


        left_eye = shape.part(36).x, shape.part(36).y, shape.part(39).x, shape.part(39).y
        right_eye = shape.part(42).x, shape.part(42).y, shape.part(45).x, shape.part(45).y

        # Print eye parameters to terminal
        print(f"Face {i+1} - Left Eye: {left_eye}, Right Eye: {right_eye}")

        # Store eye parameters in dictionary
        eye_parameters[f"Face_{i+1}"] = {"Left_Eye": left_eye, "Right_Eye": right_eye}

        # Draw rectangles around the eyes
        cv2.rectangle(frame, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (0, 255, 0), 2)

    # Display the frame with rectangles around the eyes
    cv2.imshow("Eye Detection", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Export eye parameters to a pickle file
with open("eye_parameters.pickle", "wb") as pickle_file:
    pickle.dump(eye_parameters, pickle_file)
