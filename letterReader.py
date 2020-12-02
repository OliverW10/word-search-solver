import numpy as np
from tensorflow import keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import cv2

import random

# run an already trained model on a image
class LetterReader:
    def __init__(self, modelPath):
        self.model = keras.models.load_model(modelPath)
        self.probability_model = keras.Sequential([self.model, keras.layers.Softmax()])
    def readLetters(self, imgs):
        # classify a list of images
        for i, img in enumerate(imgs):
            img = self.preProcess(img)
        #cv2.imwrite(f"{random.randint(0, 1000)}.png", imgs[0])
        predictions = self.probability_model.predict(imgs)
        choices = np.argmax(predictions, 1)
        confs = []
        for i in range(len(choices)):
            confs.append(predictions[i][choices[i]])
        return choices, confs
        
    def readLetter(self, img):
        # classify a single image
        
        img = self.preProcess(img)
        predictions = self.probability_model.predict(np.expand_dims(img, 0))
        choice = np.argmax(predictions[0])
        return choice, predictions[0][choice]


    def preProcess(self, img, extra = False):
        img = np.array(img)
        img = img / 255
        size = img.shape # height first
        
        if size != (32, 32):
            img = cv2.resize(img, (32, 32))

        if len(img.shape) == 3:
            img = self.get_grayscale(img)

        if extra:
            img = self.remove_noise(img, 3)
            img = self.thresholding(img, 5, 4) # cant do this without first greyscaling
        return img

    # https://nanonets.com/blog/ocr-with-tesseract/
    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self, image, rad = 5):
        return cv2.medianBlur(image, rad)
     
    #thresholding
    def thresholding(self, image, rad = 5, static = 3): # size has to be odd
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, rad, static)

    #dilation
    def dilate(self, image, size): # size has to be odd
         kernel = np.ones((size, size),np.uint8)
         return cv2.dilate(image, kernel, iterations = 1)
            
    #erosion
    def erode(self, image, size): # size has to be odd
         kernel = np.ones((size, size),np.uint8)
         return cv2.erode(image, kernel, iterations = 1)

if __name__ == "__main__":
    from letterReaderTrainer import *

    testReader = LetterReader("testModel1")
    testImg, testLabel = loadDatasetNpAll()[0][0], loadDatasetNpAll()[1][0]
    testImg = cv2.imread("trainSetImg/a-5400.png")
    testLabel = "a"
    print(f"should get {testLabel}")
    print(string.ascii_letters[testReader.readLetter(testImg)])