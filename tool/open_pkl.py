import pickle


with open(r'D:\Project_Graduation\Thinh.pkl', 'rb') as file:
    loaded_known_faces = pickle.load(file)


print("Danh sách mô tả khuôn mặt đã biết:")
for i, face_descriptor in enumerate(loaded_known_faces):
    print(f"Người {i + 1}: {face_descriptor}")
