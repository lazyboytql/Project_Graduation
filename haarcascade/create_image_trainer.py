import cv2
import os


train_data_folder = 'training_data'
test_data_folder = 'testing_data'


if not os.path.exists(train_data_folder):
    os.makedirs(train_data_folder)

if not os.path.exists(test_data_folder):
    os.makedirs(test_data_folder)


cap = cv2.VideoCapture(0)


img_counter = 0


is_training_data = True

while True:
    ret, frame = cap.read()

    cv2.imshow('Capture Images', frame)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        is_training_data = not is_training_data

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif cv2.waitKey(1) & 0xFF == ord('s'):
        img_name = f"{img_counter}.png"
        if is_training_data:
            img_path = os.path.join(train_data_folder, img_name)
        else:
            img_path = os.path.join(test_data_folder, img_name)

        cv2.imwrite(img_path, frame)
        print(f"{img_name} saved")

        img_counter += 1


cap.release()
cv2.destroyAllWindows()
