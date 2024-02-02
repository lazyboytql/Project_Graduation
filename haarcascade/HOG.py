import cv2
import numpy as np
import os

train_data_folder = r'D:\Project_Graduation\training_data'
test_data_folder = r'D:\Project_Graduation\testing_data'

svm = cv2.ml.SVM_create()


svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)


win_size = (64, 64)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(int('face' in filename))  
    return images, labels


train_images, train_labels = load_images(train_data_folder)
test_images, test_labels = load_images(test_data_folder)


resized_train_images = [cv2.resize(img, win_size) for img in train_images]
resized_test_images = [cv2.resize(img, win_size) for img in test_images]


train_data = np.array([hog.compute(img) for img in resized_train_images], dtype=np.float32)
train_labels = np.array(train_labels)


svm.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)


test_data = np.array([hog.compute(img) for img in resized_test_images], dtype=np.float32)
_, result = svm.predict(test_data)


accuracy = np.mean((result.flatten() == np.array(test_labels)).astype(int))
print(f'Accuracy: {accuracy * 100}%')

# Giải phóng bộ nhớ
cv2.destroyAllWindows()
