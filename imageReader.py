import cv2
import numpy as numpy
import pytesseract
import string
import os

pytesseract.pytesseract.tesseract_cmd = r".\Tesseract-OCR\tesseract.exe" # the r stops the backslashes from being escape characters, i think
config = f"--psm 12" #  -c tessedit_char_blacklist={string.punctuation}{string.digits}
class ImageProcessing:
	# realised that i can just put a box on the screen to line up grid to to make vision a lot easier
	def processImage(img, draw = False):
		newImg = ImageProcessing.preProcessImg(img)
		# print(data)
		text = pytesseract.image_to_string(newImg, config=config)
		boxes = pytesseract.image_to_boxes(newImg, config=config).splitlines()
		data = pytesseract.image_to_data(newImg, config=config, output_type = pytesseract.Output.DICT)

		if len(newImg.shape) == 2:
			h, w = newImg.shape
		else:
			h, w, c = newImg.shape

		# print(data)
		possibleBoxes = []
		for b in boxes:
			b = b.split(" ")
			if len(b[0]) == 1 and b[0] in string.ascii_letters:
				possibleBoxes.append(b)

		# find words
		n_boxes = len(data['text'])
		wordBoxes = []
		for i in range(n_boxes):
			if int(data['conf'][i]) > 90 and len(data["text"]) > 1:
				(x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
				# newImg = cv2.rectangle(newImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(newImg, str(data["conf"][i]),(x, y), font, 0.5,(20, 20, 20),1,cv2.LINE_AA)
				wordBoxes.append([int(x), int(y), int(w), int(h)])

		# remove letters that appear in words
		finalBoxes = []
		for letterBox in possibleBoxes:
			isWord = False
			for wordBox in wordBoxes:
				if ImageProcessing.boxOverlap((int(letterBox[1]), int(letterBox[2]), int(letterBox[3]), int(letterBox[4])), wordBox) > 0:
					isWord = True

			if not isWord:
				finalBoxes.append(letterBox)

		for b in finalBoxes:
			if draw:
				newImg = cv2.rectangle(newImg, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
		grid = []

		if draw:
			return grid, fourPoint, newImg
		else:
			return grid, fourPoint

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
		ratio = size[0] / size[1]
		img = cv2.resize(img, (1000, int(1000*ratio))) # , fx = 0.5, fy = 0.5
		
		img = ImageProcessing.get_grayscale(img)
		img = ImageProcessing.remove_noise(img, 3)
		img = ImageProcessing.thresholding(img, int(int(size[0]*0.05)/2)*2+1, 4) # cant do this without first greyscaling
		return img

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
		_, img = ImageProcessing.processImage(img, True)
		img = cv2.resize(img, None, fx = 0.5, fy = 0.5)
		cv2.imshow("Img", img)
		cv2.waitKey(0)