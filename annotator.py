import numpy as np
import cv2

class Annotator:
	def unCropPos(pos, p1, p2, imgSize):
		newX = pos[0]*(p2[0]-p1[0])
		newY = pos[1]*(p2[1]-p1[1])
		return newX+p1[0], newY+p1[1]

	def annotate(img, gridPlus, cropPos, words):
		drawImg = img.copy()
		# cropPos = ImageProcessing.fixCropPos()
		p1 = (int(cropPos[0][0] * img.shape[1]), int(cropPos[0][1] * img.shape[0])) # top left
		p2 = (int(cropPos[2][0] * img.shape[1]), int(cropPos[2][1] * img.shape[0])) # bottom right
		for word in words.keys():
			if len(words[word]) >= 1:
				wordPos = max(words[word], key=lambda x:x["conf"])["position"] # chooses word with highest confidence
				# print("wordPos",wordPos)
				letterRects = [ gridPlus[wordPos[0][0]][wordPos[0][1]][1], gridPlus[wordPos[1][0]][wordPos[1][1]][1] ]
				letterPoints = [ [letterRects[0][0]+letterRects[0][2]/2, letterRects[0][1]+letterRects[0][3]/2] , [letterRects[1][0]+letterRects[1][2]/2, letterRects[1][1]+letterRects[1][3]/2]]
				# print("letter points", letterPoints)
				linePoints = [Annotator.unCropPos(letterPoints[i], p1, p2, img.shape) for i in range(2)]
				# print("line points", linePoints)
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