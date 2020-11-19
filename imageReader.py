import cv2
import numpy as numpy
import string
import os
import random

class ImageProcessing:
	minContourSize = 0.001
	# realised that i can just put a box on the screen to line up grid to to make vision a lot easier
	def loadImg(fileName):
		return cv2.imread(fileName)

	def processImage(img, draw = False):
		newImg = ImageProcessing.preProcessImg(img)
		letters, newImg = ImageProcessing.findLetters(newImg)
		grid = []

		if draw:
			return grid, newImg
		else:
			return grid, newImg # just returns so it always returns two values

	def boxOverlap(box1, box2):
		# finds the total area of overlap between two rects (x, y, w, h)
		if box1[0] < box2[0] + box2[2] and box1[0] + box1[2] > box2[1] and box1[1] < box2[1] + box2[3] and box1[1] + box1[3] > box2[1]:
			width = min(box1[0]+box1[2], box2[0]+box2[2]) - max(box1[0], box2[0])
			height = min(box1[1]+box1[3], box1[1]+box2[3]) - max(box1[1], box2[1])
			return width * height
		else:
			return 0

	def preProcessImg(img):
		size = img.shape # height first
		print(size)
		img = ImageProcessing.get_grayscale(img)
		img = ImageProcessing.remove_noise(img, int(int(size[0]*0.002)/2)*2+1)
		img = ImageProcessing.thresholding(img, int(int(size[0]*0.075)/2)*2+1, 4) # cant do this without first greyscaling
		return img

	def findLetters(img):
		# first find all contours
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for i, cnt in enumerate(contours):
			if cv2.contourArea(cnt) > img.shape[0]:
				x,y,w,h = cv2.boundingRect(cnt)
				cv2.rectangle(img,(x,y),(x+w,y+h),255,2)
				print(cv2.contourArea(cnt), (x, y, w, h))
		letters = []

		return letters, img

	# https://nanonets.com/blog/ocr-with-tesseract/
	# get grayscale image
	def get_grayscale(image):
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# noise removal
	def remove_noise(image, rad = 5):
		return cv2.medianBlur(image, rad)
	 
	#thresholding
	def thresholding(image, rad = 11, static = 3): # size has to be odd
		return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, rad, static)

	#dilation
	def dilate(image, size): # size has to be odd
		kernel = np.ones((size, size),np.uint8)
		return cv2.dilate(image, kernel, iterations = 1)
		
	#erosion
	def erode(image, size): # size has to be odd
		kernel = np.ones((size, size),np.uint8)
		return cv2.erode(image, kernel, iterations = 1)

	def annotate(image, words, fourPoint):
		pass


if __name__ == "__main__":
	fileNames = os.listdir("tests")
	imageNames = []
	for i in fileNames: # could probrobly be done in one line
		if i.lower().endswith(".png") or i.lower().endswith(".jpg"):
			imageNames.append(i)

	print(imageNames)
	for i in range(5):
		img = cv2.imread("tests/"+imageNames[i]) # "tests/originals/4.png"
		grid, img = ImageProcessing.processImage(img, True)
		cv2.imwrite(f"{random.randint(0, 10000)}img.png", img)
		img = cv2.resize(img, None, fx = 0.3, fy = 0.3)
		cv2.imshow("Img", img)
		cv2.waitKey(0)