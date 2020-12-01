import cv2
import numpy as np
import string
import os
import random
from letterReader import LetterReader
import image_to_numpy


class ImageProcessing:
	# both as a multiple of the image size
	minContourSize = 0.0001
	maxContourSize = 0.1

	maxBoxSize = 1
	minBoxSize = 0

	def loadImg(fileName):
		return image_to_numpy.load_image_file(fileName)

	def processImage(img, pos, debug = False):
		newImg = ImageProcessing.preProcessImg(img)
		croppedImg = ImageProcessing.cropToPos(newImg, pos)
		smallImg = cv2.resize(croppedImg, None, fx = 0.1, fy = 0.1)
		cv2.imshow("cropped image", smallImg)
		cv2.waitKey()
		cv2.destroyAllWindows()
		grid, letters, letterImg = ImageProcessing.findLetters(croppedImg, debug)

		if debug:
			return grid, letters, letterImg
		else:
			return grid, letters, letterImg

	def boxOverlap(box1, box2):
		# finds the total area of overlap between two rects (x, y, w, h)
		if ImageProcessing.boxCollide(box1, box2):
			# assumes boxes dont have negative dimentions
			actualWidth = min(box1[0], box2[0]) + max(box1[0]+box1[2], box2[0]+box2[2])
			actualHeight = min(box1[1], box2[1]) + max(box1[1]+box1[3], box2[1]+box2[3])
			totalWidth = box1[2] + box2[2]
			totalHeight = box1[3] + box2[3]
			return (totalWidth-actualWidth) * (totalHeight-actualHeight)
		else:
			return 0

	def boxCollide(rect1, rect2):
		return not (rect2[0] > rect1[0]+rect1[2]
        or rect2[0]+rect2[2] < rect1[0]
        or rect2[1] > rect1[1]+rect1[3]
        or rect2[1]+rect2[3] < rect1[1])

	def preProcessImg(img):
		size = img.shape # height first
		img = ImageProcessing.get_grayscale(img)
		img = ImageProcessing.remove_noise(img, int(int(size[0]*0.003)/2)*2+1)
		img = ImageProcessing.thresholding(img, int(int(size[0]*0.075)/2)*2+1, 4) # cant do this without first greyscaling
		return img

	def findLetters(img, debug = False):
		if debug:
			drawImg = img.copy()

		# first find all contours that big enough to be letters
		lettersContours = []
		contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		for i, cnt in enumerate(contours):
			x,y,w,h = cv2.boundingRect(cnt) 
			if ImageProcessing.maxContourSize > cv2.contourArea(cnt)/(img.shape[0]*img.shape[1]) > ImageProcessing.minContourSize:
				# drawImg = cv2.putText(drawImg, str(round((cv2.contourArea(cnt)/img.shape[0]), 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX , 1, 0, 1, cv2.LINE_AA) 
				# cv2.rectangle(drawImg,(x,y),(x+w,y+h), 20, 3)
				lettersContours.append(cnt)

		if len(lettersContours) < 5:
			print("\n\n\n NO LETTERS FOUND \n\n")
		# use the among of found contuors to determin the size of the grid
		gridSize = round(len(lettersContours) ** 0.5)
		print("\ngridSize: ", gridSize)

		# go through every contour and get the section of img around it
		# done first so that they can be keras'ed as a batch
		letterImgs = np.empty([len(lettersContours), 32, 32])
		letterPositions = np.empty([len(lettersContours), 4])
		badCnts = []
		avgSize = 0
		for i, cnt in enumerate(lettersContours):
			# find a square centered around the contour
			x,y,w,h = cv2.boundingRect(cnt)
			midX = x+w/2
			midY = y+h/2
			maxSize = max(w, h) * (1.1)
			x = int(midX - maxSize/2)
			y = int(midY - maxSize/2)
			w = int(maxSize)
			h = int(maxSize)
			if debug:
				cv2.rectangle(drawImg,(x,y),(x+w,y+h), 20, 3)

			# crop the image to the square
			crop = img[y:y+h, x:x+w]
			if crop.shape[0] > 5 and crop.shape[1] > 5: # filter out tiny contours that somehow slipped through
				cntSize = cv2.contourArea(cnt) / (img.shape[0] * img.shape[1])
				# keep track of the average
				if i-len(badCnts) == 0:
					avgSize = cntSize
				else:
					avgSize = ( (avgSize * i-len(badCnts)) + cntSize ) / (i-len(badCnts)+1)

				# save the box position and the image
				letterPositions[i] = [x, y, w, h]
				letterImgs[i] = cv2.resize(crop, (32, 32))
			else:
				print("\nweird shaped contour ????")
				print(x, y, w, h)
				badCnts.append(i)
		print("average cnt size: ", avgSize)

		# find any overlaps between letterContours and remove the one whose size is furthest from the average
		for i1, rect1 in enumerate(letterPositions):
			for i2, rect2 in enumerate(letterPositions):
				if i1 != i2:
					if ImageProcessing.boxCollide(rect1, rect2):
						cnt1Size = cv2.contourArea(lettersContours[i1]) / (img.shape[0] * img.shape[1])
						cnt2Size = cv2.contourArea(lettersContours[i2]) / (img.shape[0] * img.shape[1])
						if abs(cnt1Size - avgSize) < abs(cnt1Size - avgSize):
							badCnts.append(i1)
						else:
							badCnts.append(i2)
		
		# get the letter for each contour and its confidence
		letReader = LetterReader("testModel1")
		letters, confs = letReader.readLetters(letterImgs)

		# combine the letters, their positions and the confidence and removes all badCnts
		lettersPlus = []
		for i in range(len(letters)):
			isBad = i in badCnts
			if not isBad:
				lettersPlus.append( (string.ascii_letters[letters[i]], letterPositions[i], confs[i]) )
				if debug:
					drawImg = cv2.putText(drawImg, lettersPlus[-1][0], (int(letterPositions[i][0]), int(letterPositions[i][1])), cv2.FONT_HERSHEY_SIMPLEX , 2, 0, 2, cv2.LINE_AA) 
			# elif debug:
				# drawImg = cv2.putText(drawImg, lettersPlus[-1][0], (int(letterPositions[i][0]), int(letterPositions[i][1])), cv2.FONT_HERSHEY_SIMPLEX , 2, 0.5, 2, cv2.LINE_AA) 

		# position all letters in grid
		grid = []
		YsortedLetters = sorted(lettersPlus, key = lambda x:x[1][1])
		for row in range(gridSize):
			rowLettersPlus = sorted( YsortedLetters[row*gridSize : (row+1)*gridSize] , key = lambda x:x[1][0] )
			rowLetters = [letter[0] for letter in rowLettersPlus]
			# print(rowLetters)
			grid.append(rowLetters)
			# del YsortedLetters[row*gridSize : (row+1)*gridSize]
		# print(grid)

		if debug:
			return grid, lettersPlus, drawImg
		else:
			return grid, lettersPlus, np.zeros((10, 10))

	def cropToRect(img, rect):
		# crop and transform to the four corners given
		img[y:y+h, x:x+w]
		return img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

	def cropToPos(img, pos):
		srcTri = np.array( [pos[0], pos[1], pos[2]] ).astype(np.float32) * np.array([img.shape[1], img.shape[0]]).astype(np.float32)
		dstTri = np.array( [[0, 0], [1, 0], [1, 1]] ).astype(np.float32) * np.array([img.shape[1], img.shape[0]]).astype(np.float32)

		warp_mat = cv2.getAffineTransform(srcTri, dstTri)
		warp_dst = cv2.warpAffine(img, warp_mat, (img.shape[1], img.shape[0]))
		print(type(warp_dst), warp_dst.shape, warp_dst.dtype)
		print(type(img), img.shape, img.dtype)
		return warp_dst

	def getAvgCol(img):
		# for greyscale images
		return np.sum(img)/(img.shape[0]*img.shape[1])

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

	def annotate(img, wordsPos, warpPos):
		pass

if __name__ == "__main__":
	fileNames = os.listdir("tests")
	imageNames = []
	for i in fileNames: # could probrobly be done in one line
		if i.lower().endswith(".png") or i.lower().endswith(".jpg"):
			imageNames.append(i)

	# print(imageNames)
	for i in range(5):
		img = cv2.imread("tests/"+imageNames[i]) # "tests/originals/4.png"
		grid, letters, img = ImageProcessing.processImage(img, [[0, 0], [1, 0], [1, 1]], True)
		cv2.imwrite("results/"+imageNames[i], img)
		img = cv2.resize(img, None, fx = 0.3, fy = 0.3)
		
		cv2.imshow("Img", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()