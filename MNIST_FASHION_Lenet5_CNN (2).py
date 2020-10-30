import tensorflow as tf
from tensorflow import keras


import numpy as np
import matplotlib.pyplot as plt

print("Tensorflow Version:", tf.__version__)

#Lấy dữ liệu từ mnist
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Các lớp chứa dữ liệu
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Khám phá dữ liệu

train_images.shape

len(train_labels)

test_images.shape

len(test_labels)

#tiền xử lý dữ liệu

#Điều chỉnh giá trị trong phạm vi từ 0-1
train_images = train_images / 255.0

test_images = test_images / 255.0

#Hiển thị 25 ảnh đầu tiên để huấn luyện và hiển thị các lớp dưới mỗi hình ảnh

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# Xây dựng model - LeNet inspired CNN architecture
model = keras.models.Sequential([
    keras.layers.Conv2D(20, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.Conv2D(50, (5, 5), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#Xem chi tiết model
model.summary()

#Tái cấu trúc model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

#Đầu ra của model sẽ là vector 1 chiều với kích cỡ 10


# chuyển đội đại diện hiện tại của mỗi label sang "1 đại diện duy nhất"
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

#Xem cấu trúc mới của tập train và test

print('train_images shape:', train_images.shape)
print('test_images shape:', test_images.shape)
print('train_labels shape:', train_labels.shape)
print('test_labels shape:', test_labels.shape)

#Xem một đại diện duy nhất
train_labels[0]

#Biên dịch model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Huấn luyện model
model.fit(train_images, train_labels, epochs=5)

#Đánh giá sự chính xác
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#Dự đoán
predictions = model.predict(test_images)
predictions[0]

#Nhãn với giá trị cao nhất

np.argmax(predictions[0])

#Kiểm tra nhãn xem nó có đúng với test không
test_labels[0]
