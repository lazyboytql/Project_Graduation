import pickle

# Mở tệp pickle để đọc
with open(r'D:\Project_Graduation\known_faces.pkl', 'rb') as file:
    loaded_known_faces = pickle.load(file)

# In ra danh sách mô tả khuôn mặt đã biết từ tệp
print("Danh sách mô tả khuôn mặt đã biết:")
for i, face_descriptor in enumerate(loaded_known_faces):
    print(f"Người {i + 1}: {face_descriptor}")
