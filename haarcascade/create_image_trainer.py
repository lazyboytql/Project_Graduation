import cv2
import os

# Thiết lập các thư mục cho dữ liệu huấn luyện và kiểm tra
train_data_folder = 'training_data'
test_data_folder = 'testing_data'

# Tạo thư mục nếu chúng không tồn tại
if not os.path.exists(train_data_folder):
    os.makedirs(train_data_folder)

if not os.path.exists(test_data_folder):
    os.makedirs(test_data_folder)

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Biến đếm số ảnh đã chụp
img_counter = 0

# Biến để xác định liệu ảnh nên thuộc tập huấn luyện hay kiểm tra
is_training_data = True

while True:
    ret, frame = cap.read()

    # Hiển thị frame
    cv2.imshow('Capture Images', frame)

    # Nhấn 't' để chuyển đổi giữa tập huấn luyện và kiểm tra
    if cv2.waitKey(1) & 0xFF == ord('t'):
        is_training_data = not is_training_data

    # Nhấn 'q' để thoát
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Nhấn 's' để lưu ảnh vào thư mục tương ứng
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        img_name = f"{img_counter}.png"
        if is_training_data:
            img_path = os.path.join(train_data_folder, img_name)
        else:
            img_path = os.path.join(test_data_folder, img_name)

        cv2.imwrite(img_path, frame)
        print(f"{img_name} saved")

        img_counter += 1

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
