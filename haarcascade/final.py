import cv2
import numpy as np
import os

train_data_folder = r'D:\Project_Graduation\training_data'
test_data_folder = r'D:\Project_Graduation\testing_data'

# Tạo một đối tượng SVM
svm = cv2.ml.SVM_create()

# Tham số của SVM
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)

# Khai báo đối tượng HOG với tham số 
win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

# Tải data huấn luyện và kiểm tra
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(int('face' in filename))  # Label 1  khuôn mặt, 0 là không có
    return images, labels

# Load data huấn luyện và kiểm tra
train_images, train_labels = load_images(train_data_folder)
test_images, test_labels = load_images(test_data_folder)

# Thay đổi kích thước hình ảnh trước khi dùng HOG
resized_train_images = [cv2.resize(img, win_size) for img in train_images]
resized_test_images = [cv2.resize(img, win_size) for img in test_images]

# Chuẩn bị data cho SVM
train_data = np.array([hog.compute(img) for img in resized_train_images], dtype=np.float32)
train_labels = np.array(train_labels)

# Huấn luyện SVM
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)


cap = cv2.VideoCapture(0)

# Load Haar được đào tạo trước để nhận diện 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
# Chuyển khung hình sang thang độ xám cho Haar
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận diện bằng haar
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
       
        face_roi = gray_frame[y:y + h, x:x + w]

        
        resized_face = cv2.resize(face_roi, win_size)
        face_data = np.array([hog.compute(resized_face)], dtype=np.float32)

        # Predict using SVM
        _, result = svm.predict(face_data)

        # Vẽ hộp giới hạn nếu là khuôn mặt (Label 1)
        if int(result[0][0]) == 1:
            cv2.putText(frame, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow('Face Detection', frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
