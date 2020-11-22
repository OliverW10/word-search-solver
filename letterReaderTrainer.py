from tensorflow import keras
import numpy as np
import cv2
import random
import string
import os
import time

def loadDatasetImg(toLoad = 10000): # lowest file size (beacuse of png compression)
    fileNames = os.listdir("trainSetImg/")
    l = len(fileNames)
    images = np.empty((l, 32, 32, 3))
    labels = np.empty(l)
    readimg_start = time.time()
    for i in range(min(toLoad, l)):
        images[i] = cv2.imread(f"trainSetImg/{fileNames[i]}")
    print("load time: " + str(time.time() - readimg_start))
    return images, labels

def loadDatasetNpAll(*args): # slightly higher file size but much quicker loading
    readimg_start = time.time()
    images = np.load("trainSetNpOne/images.npy")
    labels = np.load("trainSetNpOne/labels.npy")
    print("load time: " + str(time.time() - readimg_start))
    return images, labels
    
if __name__ == "__main__":
    start_time = time.time()
    print("Img: ")
    images1, labels1 = loadDatasetImg()
    print("Np Grouped: ")
    images3, labels3 = loadDatasetNpAll()
    print("total time: " + str(time.time()-start_time))
    print(len(images1), len(images3))
    # cv2.imshow("test", images3[5])
    # print(string.ascii_letters[labels3[5]])

    images3 = images3 / 255

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(52)
    ])
    
    model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    
    model.fit(images3[100:], labels3[100:], epochs=100)

    test_loss, test_acc = model.evaluate(images3[:100],  labels3[:100], verbose=2)

    print('\nTest accuracy:', test_acc)
    # https://www.tensorflow.org/guide/keras/save_and_serialize
    model.save("testModel2")

    probability_model = keras.Sequential([model, keras.layers.Softmax()])

    predictions = probability_model.predict(images3[:100])

    for i in range(5):
        print(np.argmax(predictions[i]))
        print(labels3[i])
        print("\n")
