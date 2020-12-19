import string
import random
import copy
import time

# use kivy for app, https://kivy.org/#home
# use pytesseract for ocr https://pypi.org/project/pytesseract/

class Solvers:
	allWords = ["hello", "test", "python", "app"] # should be dataset of all words
	def wordSearch1(grid, words):
		# idea 1: create strings of all the vertical and horizontal and just use 'if word in string'

		lines = []
		for i in range(len(grid)): # add all 
			lines.append([item[i] for item in grid])
		print(lines)

		for n, word in enumerate(words):
			print(word, n)


	def wordSearch2(grid, words):
		# loop through grid looking for first letter of word, then around that for second and on

		# make all letters lowercase
		lowerRow = lambda row:list(map(lambda x:x.lower(), row))
		grid = list(map(lambda x:lowerRow(x), grid))
		directions = [[1, 0], [0, 1], [1, 1], [1, -1], [-1, 0], [-1, -1], [-1, 1], [0, -1]]

		foundWords = {}
		for n, word in enumerate(words):
			foundWords[word] = []
			print(word)
			firsts = Solvers.findLetters(grid, word[0]) # find where all the first letters are
			print(firsts)
			for pos in firsts:
				# print(f"\n\n next pos {pos} ")
				for direction in directions: # for each first letter try each direction
					# print("\n next dir")
					stillGoing = True
					for i in range(1, len(word)): # iterate forwards checking if the letter is correct
						x = pos[0] + direction[0]*i
						y = pos[1] + direction[1]*i
						if 0 <= x < len(grid) and 0 <= y < len(grid): # first checks if its inside the grid to prevent out-of-range-ing
							if grid[x][y] != word[i]:
								stillGoing = False
								# print("wrong letter")
						else:
							stillGoing = False
							# print("out of grid")

					if stillGoing:
						# print(f"found word {word} {str([pos, ( pos[0] + direction[0]*len(word), pos[1] + direction[1]*len(word) ) ])}")
						foundWords[word].append((pos, [ pos[0] + direction[0]*(len(word)-1), pos[1] + direction[1]*(len(word)-1) ] ))
		return foundWords

	def findLetters(grid, toFind):
		# helper function for wordSearch2()
		# returns all positions the letter is at
		poss = []
		for x, row in enumerate(grid):
			for y, letter in enumerate(row):
				if letter.lower() == toFind.lower():
					poss.append([x, y])
		return poss

	def wordSearch(grid, words, algo = 2):
		if algo == 1:
			return Solvers.wordSearch1(grid, words)
		if algo == 2:
			return Solvers.wordSearch2(grid, words)


if __name__ == "__main__":

	def generateWordSearch(size, words, backwards = False): # only for debugging so not in a class
		# for some reason generates overlapping words so testing with this might not be a good idea just yet
		grid = []

		# fill grid with random letters
		for i in range(size):
			grid.append([])
			for j in range(size):
				grid[-1].append(random.choice(string.ascii_lowercase))

		# define allowed directions
		allDirections = [[1, 0], [0, 1], [1, 1], [1, -1]]
		if backwards:
			directions.append([[-1, 0], [-1, -1], [-1, 1], [0, -1]])

		# current dosent support overlapping
		usedSpots = []
		for n, word in enumerate(words):
			direction = random.choice(allDirections)
			pos = random.randint(0, size-1), random.randint(0, size-1)

			overlapping = False
			wordTrys = 0
			while not (0 < pos[0] + direction[0]*len(word) < size and 0 < pos[1] + direction[1]*len(word) < size) or overlapping == True: # pick a position that isnt outside the grid and isnt overlapping other words
				pos = random.randint(0, size-1), random.randint(0, size-1)
				overlapping = False
				for num, letter in enumerate(word): # check whether any of the letters in the word overlap other words
					if [pos[0] + num*direction[0], pos[1] + num*direction[1]] in usedSpots:
						overlapping = True
				wordTrys += 1

			# put word in found postion
			print(word+"  "+str(pos))
			for num, letter in enumerate(word):
				grid[pos[0] + num*direction[0]][pos[1] + num*direction[1]] = letter
				usedSpots.append([pos[0] + num*direction[0], pos[1] + num*direction[1]])
		return grid


	testWords = ["test", "hi"]

	testGrid = [["a"]*10 for x in range(10)]# generateWordSearch(100, testWords)
	testGrid[2][5], testGrid[2][6], testGrid[2][7], testGrid[2][8] = "t", "e", "s", "t"
	testGrid[7][0], testGrid[6][0] = "h", "i"
	testGrid[5][0], testGrid[6][1] = "h", "h"
	for i in testGrid:
		for j in i:
			print(j, end = " ")
		print("")
	print("\n"*2)

	# start_time = time.time()
	# for i in range(1000):
		# Solvers.wordSearch(testGrid, testWords)
	# print("time: ", (time.time()-start_time)/1000)
	print(Solvers.wordSearch(testGrid, testWords))