from tensorflow import keras
import numpy as np
import cv2

# run an already trained model on a image
class LetterReader:
    def __init__(self, modelPath):
        self.model = keras.models.load_model(modelPath)

    def readLetters(self, imgs):
        # classify a list of images
        for i in imgs:
            i = self.preProcess(i)
    
    def readLetter(self, img):
        # classify a single image
        img = self.preProcess(img)
        self.model
    def preProcess(self, img):
        size = img.shape # height first
        # ratio = size[0] / size[1]
        img = cv2.resize(img, (32, 32))
        
        img = get_grayscale(img)
        img = remove_noise(img, 3)
        img = thresholding(img, int(int(size[0]*0.05)/2)*2+1, 4) # cant do this without first greyscaling
        return img

    # https://nanonets.com/blog/ocr-with-tesseract/
    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self, image, rad = 5):
        return cv2.medianBlur(image, rad)
     
    #thresholding
    def thresholding(self, image, rad = 11, static = 3): # size has to be odd
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, rad, static)

    #dilation
    def dilate(self, image, size): # size has to be odd
         kernel = np.ones((size, size),np.uint8)
         return cv2.dilate(image, kernel, iterations = 1)
            
    #erosion
    def erode(self, image, size): # size has to be odd
         kernel = np.ones((size, size),np.uint8)
         return cv2.erode(image, kernel, iterations = 1)

if __name__ == "__self__":
    testReader = LetterReader("testModel1")
