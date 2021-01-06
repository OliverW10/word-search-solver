import numpy as np
import cv2
from copy import deepcopy
import math

class Annotator:
	def unCropPos(pos, p1, p2, imgSize):
		newX = pos[0]*(p2[0]-p1[0])
		newY = pos[1]*(p2[1]-p1[1])
		return newX+p1[0], newY+p1[1]

	def annotate(img, gridPlus, cropPos, words):
		drawImg = img.copy()
		p1 = (int(cropPos[0][0] * img.shape[1]), int(cropPos[0][1] * img.shape[0])) # top left
		p2 = (int(cropPos[2][0] * img.shape[1]), int(cropPos[2][1] * img.shape[0])) # bottom right
		for word in words.keys():
			if len(words[word]) >= 1:
				wordPos = max(words[word], key=lambda x:x["conf"])["position"] # chooses word with highest confidence
				letterRects = [ gridPlus[wordPos[0][0]][wordPos[0][1]][1], gridPlus[wordPos[1][0]][wordPos[1][1]][1] ]
				letterSize = (letterRects[0][2]+letterRects[0][3] + letterRects[1][2]+letterRects[1][3])/5 # by 5 beacuse the rect it bigger than the letter
				print("letter size", letterSize)

				letterPoints = [ [letterRects[0][0]+letterRects[0][2]/2, letterRects[0][1]+letterRects[0][3]/2] , [letterRects[1][0]+letterRects[1][2]/2, letterRects[1][1]+letterRects[1][3]/2]]
				
				angle = -math.atan2(letterPoints[1][1]-letterPoints[0][1], letterPoints[1][0]-letterPoints[0][0])
				print(angle, "angle")

				sideLines = [[letterPoints[0][0] + math.sin(angle)*letterSize, letterPoints[0][1] + math.cos(angle)*letterSize], # first line
				[letterPoints[1][0] + math.sin(angle)*letterSize, letterPoints[1][1] + math.cos(angle)*letterSize],
				[letterPoints[1][0] - math.sin(angle)*letterSize, letterPoints[1][1] - math.cos(angle)*letterSize], # second line
				[letterPoints[0][0] - math.sin(angle)*letterSize, letterPoints[0][1] - math.cos(angle)*letterSize]]

				linePoints = [Annotator.unCropPos(sideLines[i], p1, p2, img.shape) for i in range(4)]

				pts = np.array(linePoints, np.int32)
				pts = pts.reshape((-1,1,2))
				cv2.polylines(drawImg, [pts], False, (0,255,255), int(img.shape[0]/500))
			else:
				print(f"word {word} not found")
		# for l in lettersPlus:
		# 	x = int(lerp(cropPos[0][0], cropPos[2][0], l[1][0]) * img.shape[1])
		# 	y = int(lerp(cropPos[0][1], cropPos[2][1], l[1][1]) * img.shape[0])
		# 	img = cv2.putText(img, l[0], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX , 2, 0, 2, cv2.LINE_AA) 
		# 	# cv2.rectangle(img,  (l[1][0], l[1][1]), (l[1][0]+l[1][2], l[1][1]+l[1][3]),  20, 3)
		cv2.rectangle(drawImg, p1, p2, (0, 255, 0), 3)
		return drawImg

	def drawGrid(grid, size = (800, 800)):
		img = np.zeros(size)
		img.fill(255)
		for i in range(len(grid)):
			for j in range(len(grid)):
				cv2.putText(img, grid[j][i], (int(((i+0.5)/len(grid))*size[0]), int(((j+0.5)/len(grid))*size[1])), cv2.FONT_HERSHEY_SIMPLEX, size[0]*0.001, 0)
		return img


if __name__ == "__main__":
	from imageReader import ImageProcessing
	from solvers import Solvers
	import csv

	words = {}
	with open('/home/olikat/word-search-solver/test_images/labels.csv', newline='') as csvfile:
		wordreader = csv.reader(csvfile)
		for row in wordreader:
			words[int(row[0])] = row[1:]

	img = ImageProcessing.loadImg("test_images/1.0.jpg")
	_, gridPlus, multiGrid = ImageProcessing.processImage(img, [[0, 0], [1, 0], [1, 1]])
	foundWords = Solvers.wordSearch(multiGrid, words[1])
	annotatedImg = Annotator.annotate(img, gridPlus, [[0, 0], [1, 0], [1, 1]], foundWords)
	cv2.imshow("annoted image", cv2.resize(annotatedImg, None, None, fx=0.4, fy=0.4))
	cv2.waitKey()