import cv2
import numpy as np
import string
import os
import random
from letterReader import LetterReader
import image_to_numpy


class ImageProcessing:
	# both as a multiple of the image size
	minContourSize = 0.5
	maxContourSize = 10

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

	def boxOverlap(box1, box2, amount):
		# finds the total area of overlap between two rects (x, y, w, h)
		if box1[0] < box2[0] + box2[2] and box1[0] + box1[2] > box2[1] and box1[1] < box2[1] + box2[3] and box1[1] + box1[3] > box2[1]:
			if amont:
				width = min(box1[0]+box1[2], box2[0]+box2[2]) - max(box1[0], box2[0])
				height = min(box1[1]+box1[3], box1[1]+box2[3]) - max(box1[1], box2[1])
				return width * height
			else:
				1
		else:
			return 0

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
			if ImageProcessing.maxContourSize > cv2.contourArea(cnt)/img.shape[0] > ImageProcessing.minContourSize:
				# and w > 3 and h > 3 and w < h*2 and h < w*10
				# drawImg = cv2.putText(drawImg, str(round((cv2.contourArea(cnt)/img.shape[0]), 2)), (x, y), cv2.FONT_HERSHEY_SIMPLEX , 1, 0, 1, cv2.LINE_AA) 
				# cv2.rectangle(drawImg,(x,y),(x+w,y+h), 20, 3)
				lettersContours.append(cnt)

		# use the among of found contuors to determin the size of the grid
		gridSize = round(len(lettersContours) ** 0.5)
		print("\n\ngridSize: ", gridSize)

		# go through every contour and get the section of img
		# done so that they can be keras'ed as a batch
		letterImgs = np.empty([len(lettersContours), 32, 32])
		letterPositions = np.empty([len(lettersContours), 4])
		badCnts = []
		avgSize = 0
		for i, cnt in enumerate(lettersContours):
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
			crop = img[y:y+h, x:x+w]
			if crop.shape[0] > 3 and crop.shape[1] > 3:
				if i-len(badCnts) == 0:
					avgSize = cv2.contourArea(cnt)
				else:
					avgSize = ( (avgSize * i-len(badCnts)-1) + cv2.contourArea(cnt) ) / i-len(badCnts)
				letterPositions[i] = [x, y, w, h]
				letterImgs[i] = cv2.resize(crop, (32, 32))
			else:
				print("\nweird shaped contour ????")
				print(x, y, w, h)
				badCnts.append(i)
		print("average cnt size: ", avgSize)

		# find any overlaps between letterContours and remove the one whose size is furthest from the average

		
		# get the letter for each contour and its confidence
		letReader = LetterReader("testModel1")
		letters, confs = letReader.readLetters(letterImgs)

		# combine the letters, their positions and the confidence
		lettersPlus = []
		for i in range(len(letters)):
			isBad = i in badCnts
			lettersPlus.append( (string.ascii_letters[letters[i]], letterPositions[i], confs[i], isBad) )
			if debug:
				drawImg = cv2.putText(drawImg, lettersPlus[-1][0], (int(letterPositions[i][0]), int(letterPositions[i][1])), cv2.FONT_HERSHEY_SIMPLEX , 2, 0, 2, cv2.LINE_AA) 

		# last pass of removing false-positives
		# for n in badCnts:
		# 	del lettersPlus[n]
		lettersPlus = list(filter(lambda x:x[3] == False, lettersPlus))

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
		print("average colour ",ImageProcessing.getAvgCol(img))

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