import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#load tập train và tập test từ mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)



#Chuẩn hóa dữ liệu
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255



model = tf.keras.Sequential()
# Định hình đầu vào của input
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary() #Xem tổng quan model



#biến dịch model
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])



#Huấn luyện model
model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])



''' #Đánh giá qua độ chỉnh xác của tập test
# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])
'''


path=input("Moi ban nhap duong dan anh: ")
out=model.predict(path)
plt.text(2,4,out) 
plt.imshow(path)