import cv2
import numpy as np
import string
import os
import random
from letterReader import LetterReader

class ImageProcessing:
	# both as a multiple of the image size
	minContourSize = 0.8
	maxContourSize = 2

	def loadImg(fileName):
		return cv2.imread(fileName)

	def processImage(img, pos, draw = False):
		newImg = ImageProcessing.preProcessImg(img)
		croppedImg = ImageProcessing.cropToPos(newImg, pos)
		letters, newImg = ImageProcessing.findLetters(newImg)
		print(letters)
		grid = []

		if draw:
			return grid, croppedImg
		else:
			return grid, croppedImg # just returns so it always returns two values

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
		img = ImageProcessing.get_grayscale(img)
		img = ImageProcessing.remove_noise(img, int(int(size[0]*0.002)/2)*2+1)
		img = ImageProcessing.thresholding(img, int(int(size[0]*0.075)/2)*2+1, 4) # cant do this without first greyscaling
		return img

	def findLetters(img):
		# first find all contours that big enough to be letters
		lettersContours = []
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for i, cnt in enumerate(contours):
			if img.shape[0] * ImageProcessing.maxContourSize > cv2.contourArea(cnt) > img.shape[0] * ImageProcessing.minContourSize:
				x,y,w,h = cv2.boundingRect(cnt)
				cv2.rectangle(img,(x,y),(x+w,y+h),255,2)
				lettersContours.append(cnt)

		# use the among of found contuors to determin the size of the grid
		gridSize = round(len(lettersContours) ** 0.5)
		print("gridSize: ", gridSize)

		# go through every contour and get the section of img
		# done so that they can be keras'ed as a batch
		letterImgs = []#np.empty([len(lettersContours), 32, 32])
		for i, cnt in enumerate(lettersContours):
			x,y,w,h = cv2.boundingRect(cnt)
			letterImgs.append( img[y:y+h, x:x+w] )
		
		# get the letter for each contour and its confidence
		letReader = LetterReader("testModel1")
		letters, confs = letReader.readLetters(letterImgs)
		print(letters)

		# last pass of removing false-positives


		# position all letters in grid

		''' 
		psudocode:
		sort all letters by y pos
		for each row in grid
			take first gridSize of letters
			sort by x pos and add to grid
			remove those letters from letters list 
		'''
		return letters, img

	def cropToPos(img, pos):
		# crop and transform to the four corners given
		return img # not done yet



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

	def annotate(img, words):
		pass

if __name__ == "__main__":
	fileNames = os.listdir("tests/fulls")
	imageNames = []
	for i in fileNames: # could probrobly be done in one line
		if i.lower().endswith(".png") or i.lower().endswith(".jpg"):
			imageNames.append(i)

	# print(imageNames)
	for i in range(5):
		img = cv2.imread("tests/fulls/"+imageNames[i]) # "tests/originals/4.png"
		grid, img = ImageProcessing.processImage(img, [[0, 0], [0, 1], [1, 0], [1,1]], True)
		# cv2.imwrite(f"{random.randint(0, 10000)}img.png", img)
		img = cv2.resize(img, None, fx = 0.3, fy = 0.3)
		cv2.imshow("Img", img)
		cv2.waitKey(0)