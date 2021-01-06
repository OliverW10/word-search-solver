import csv
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from imageReader import ImageProcessing
from solvers import Solvers
from annotator import Annotator
sys.path.pop(0)
import os
import time
import cv2

if __name__ == "__main__":
	words = {}
	with open('/home/olikat/word-search-solver/test_images/labels.csv', newline='') as csvfile:
		wordreader = csv.reader(csvfile)
		for row in wordreader:
			words[row[0]] = row[1:]

	fileNames = os.listdir("/home/olikat/word-search-solver/test_images")
	imageNames = []
	for i in fileNames:
		if i.lower().endswith(".png") or i.lower().endswith(".jpg"):
			imageNames.append(i)

	avgFound = 0
	avgTime = 0
	for i in imageNames:
		img = ImageProcessing.loadImg("/home/olikat/word-search-solver/test_images/"+i)
		wordSearchNum = i[0]
		start_time = time.time()
		grid, _, grids = ImageProcessing.processImage(img, [[0, 0], [1, 0], [1, 1]], False)
		res = Solvers.wordSearch(grids, words[wordSearchNum])
		total_time = time.time() - start_time
		totalWords = len(words[wordSearchNum])
		foundWords = 0
		for word in res.keys():
			if len(res[word]) >= 1:
				foundWords += 1
		print(f"{round(foundWords/totalWords, 5)*100}% on {i} in {total_time}")
		avgFound += (foundWords/totalWords)/len(imageNames)
		avgTime += total_time/len(imageNames)
	print(f"average of {round(avgFound, 5)*100}% in {avgTime}")
