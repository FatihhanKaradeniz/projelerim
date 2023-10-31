import cv2
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
from keras import models
from keras import layers

img = cv2.imread('el.yazim.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28))
mnist_format = np.reshape(resized, (1, 28 * 28))
mnist_format = mnist_format.astype('float32') / 255.0
network=models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
print("train_labels:",set(train_labels),"\t test_labels:",set(test_labels))
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# BU KODLARDA BAŞKA BİR DENEME YAPTIM DOĞRU OLMUŞTUR UMARIM