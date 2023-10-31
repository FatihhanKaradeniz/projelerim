import cv2
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


img = cv2.imread('sekiz.yazim.jpeg', cv2.IMREAD_GRAYSCALE)
resized = cv2.resize(img, (28, 28))
mnist_format = np.reshape(resized, (1, 28 * 28))
mnist_format = mnist_format.astype('float32') / 255.0


network = Sequential()
network.add(Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(Dense(10, activation='softmax'))
network.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])


network.fit(train_images, train_labels, epochs=5, batch_size=128)


prediction = network.predict(mnist_format)
predicted_label = np.argmax(prediction)

print("Tahmin edilen rakam:", predicted_label)




