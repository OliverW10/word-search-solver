import string
import random
import copy
import time
import numpy as np
import math


class Solvers:
    allLetters = string.ascii_letters + " "

    def wordSearch(
        grid, words, printNotFound=False
    ):  # seaerches a single grid for the words
        # loop through grid looking for first letter of word, then around that for second and on

        directions = [
            [1, 0],
            [0, 1],
            [1, 1],
            [1, -1],
            [-1, 0],
            [-1, -1],
            [-1, 1],
            [0, -1],
        ]

        foundWords = {}
        for n, word in enumerate(words):
            word = word.strip()
            foundWords[word] = []
            # print(word)
            firsts = Solvers.findLetter(
                grid, word[0]
            )  # find where all the first letters are
            # print(firsts)
            for pos in firsts:
                # print(f"\n\n next pos {pos} ")
                for direction in directions:  # for each first letter try each direction
                    # print("\n next dir")
                    stillGoing = True
                    conf = Solvers.hasLetter(grid[int(pos[0])][int(pos[1])], word[0])
                    for i in range(
                        1, len(word)
                    ):  # iterate forwards checking if the letter is correct
                        x = pos[0] + direction[0] * i
                        y = pos[1] + direction[1] * i
                        if 0 <= x < len(grid) and 0 <= y < len(
                            grid
                        ):  # first checks if its inside the grid to prevent out-of-range-ing
                            if (
                                Solvers.hasLetter(grid[x][y], word[i]) == 0
                            ):  # checks if the letter is incorrect
                                stillGoing = False
                                break
                            else:
                                conf += Solvers.hasLetter(
                                    grid[x][y], word[i]
                                )  # means the position is in the grid and has the correct letter
                        else:
                            stillGoing = False
                            break

                    if stillGoing:
                        # print(f"found word {word} at {str([pos, ( pos[0] + direction[0]*(len(word)-1), pos[1] + direction[1]*(len(word)-1) ) ])} with {conf} out of {len(word)*len(grid[0][0])} matches")
                        foundWords[word].append(
                            {
                                "position": (
                                    pos,
                                    [
                                        pos[0] + direction[0] * (len(word) - 1),
                                        pos[1] + direction[1] * (len(word) - 1),
                                    ],
                                ),
                                "conf": (conf / len(word)) / len(grid[0][0]),
                            }
                        )
        if printNotFound:
            print(
                f"words not found: {list(word for word in foundWords.keys() if len(foundWords[word])==0)}"
            )
        return foundWords

    def findLetter(grids, toFind):  # takes a grid of numbers and a letter as a string
        poss = []
        for x, row in enumerate(grids):
            for y, position in enumerate(row):
                if (
                    Solvers.hasLetter(position, toFind) >= 1
                ):  # creates a list of bools of if whether it is toFind
                    # print(f"found letter {toFind} (or {Solvers.allLetters.index(toFind)}) in {position}, {Solvers.hasLetter(position, toFind)} times")
                    poss.append([x, y])
        return poss

    def hasLetter(possibilities, letter):
        return sum(
            Solvers.allLetters[int(letterIdx)].lower() == letter.lower()
            for letterIdx in possibilities
        )


class PositionSolver:
    MAX_DIST_PER = 0.05 # maximum distance to search for letters, in percent of image size
    MAX_ANGLE = math.pi*0.4 # max allowed misalightment from set angle, in radians

    def __init__(self, imageSize):
        self.MAX_DIST = self.MAX_DIST_PER*math.sqrt(imageSize[0]*imageSize[1])
        self.sMAX_DIST = math.sqrt(self.MAX_DIST) # to avoid a square root when finding distance

    def wordSearch(self, lettersPlus, words):
        """
        will be slower by less suseptable to mistakes in gridSize
        psudocode
        for each word
                find each instance of the first letter
                        check for any of the second letters near by
                        if its found get the angle
                        find any of the third letter near by that roughly follow the angle
                        repeat for each letter
                        can use how much it follows the angle in confidence aswell
        """
        self.letters = lettersPlus

        foundWords = {}
        for word in words:
            foundWords[word] = self.findWord(word)

        return foundWords

    def findWord(self, targetWord):
        possibleWords = []

        # find all instances of first letter
        firsts = [[l] for l in self.letters if targetWord[0] in l.allLetters]

        # find all of second letters that are nearby the first at any angle
        for first in firsts:
            nxt = nextLetter(targetWord[1], first.position, anyAngle = True)
            for second in nxt:
                possibleWords.append([first, second])

        # continue for all other letters of the word with a similar angle
        for letterNum in range(2, len(targetWord)):
            for word in possibleWords:
                wordAngle = math.atan2(word[0].position[1] - word[1].position[1] ,  word[0].position[0] - word[1].position[0])
                res = nextLetter(targetWord[letterNum], word[-1].position, angle=wordAngle)
                if res != False:
                    word.append(res)

            # removes words which didnt continue
            possibleWords = [word for word in possibleWords if (len(word) == letterNum+1)]

        # gets confidences for each possible word
        wordConfs = []
        for i, word in enumerate(possibleWords):
            # foe ach word sum the count of what letter is meant to be in each spot
            wordConfs.append(sum( [let.allLetters.count(targetWord[i]) for let in word] ) /len(word))

        # gets the index of the most confident word
        bestIdx = wordConfs.index( max(wordConfs) )
        return possibleWords[bestIdx]

    def nextLetter(self, targetLetter, prevPos, angle = 0, anyAngle = False):
        '''
        searches for a letter that meets requirements
        
        finds letters that could be targetLetter and are:
        within self.MAX_DIST from pos
        within self.MAX_ANGLE_DIFF from angle
        '''
        goodLetters = []
        # checks every letter
        for letter in self.letters:
            # checks if its the target letter
            if targetLetter in letter.allLetters:
                # check sif its withiin distance
                if (letter.position[0]-prevPos[0])**2 + (letter.position[1]-prevPos[1])**2 < self.sMAX_DIST:

                    if not anyAngle:
                        nowAngle = math.atan2(letter.position[1]-prevPos[1], letter.position[0]-prevPos[0])
                        if abs(newAngle-prevAngle) < self.MAX_ANGLE or abs(newAngle-prevAngle)-math.pi*2 < self.MAX_ANGLE:
                            goodLetters.append(letter)
                    else:
                        goodLetters.append(letter)

        # if it finds more than one letter that meets the requirements it takes the one its most confident about
        # NOTE: dont know if allLetters is string letters of numbers, check later
        if len(goodLetters) >= 1:
            return max(goodLetters, key=lambda x:x.allLetters.count(targetLetter))
        else:
            return False

if __name__ == "__main__":

    def generateWordSearch(
        size, words, backwards=False
    ):  # only for debugging so not in a class
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
            pos = random.randint(0, size - 1), random.randint(0, size - 1)

            overlapping = False
            wordTrys = 0
            while (
                not (
                    0 < pos[0] + direction[0] * len(word) < size
                    and 0 < pos[1] + direction[1] * len(word) < size
                )
                or overlapping == True
            ):  # pick a position that isnt outside the grid and isnt overlapping other words
                pos = random.randint(0, size - 1), random.randint(0, size - 1)
                overlapping = False
                for num, letter in enumerate(
                    word
                ):  # check whether any of the letters in the word overlap other words
                    if [
                        pos[0] + num * direction[0],
                        pos[1] + num * direction[1],
                    ] in usedSpots:
                        overlapping = True
                wordTrys += 1

            # put word in found postion
            print(word + "  " + str(pos))
            for num, letter in enumerate(word):
                grid[pos[0] + num * direction[0]][pos[1] + num * direction[1]] = letter
                usedSpots.append(
                    [pos[0] + num * direction[0], pos[1] + num * direction[1]]
                )
        return grid

    testWords = ["test", "hi"]

    testGrid = [["a"] * 10 for x in range(10)]  # generateWordSearch(100, testWords)
    testGrid[2][5], testGrid[2][6], testGrid[2][7], testGrid[2][8] = "t", "e", "s", "t"
    testGrid[7][0], testGrid[6][0] = "h", "i"
    testGrid[5][0], testGrid[6][1] = "h", "h"
    for i in testGrid:
        for j in i:
            print(j, end=" ")
        print("")
    print("\n" * 2)

    with open("./tests/output/data.npz", "rb") as f:
        imgData = np.load(f)
        imgNames = list(imgData.keys())
        print("image data loaded for ", imgNames)
        print("shape of image", imgData["20201114_181054.jpg"].shape)

        print("20201114_181054.jpg")
        print(
            Solvers.wordSearch(
                imgData["20201114_181054.jpg"],
                [
                    "alder",
                    "apple",
                    "ash",
                    "aspen",
                    "birch",
                    "buckthorn",
                    "cedar",
                    "cherry",
                    "chestnut",
                    "chinkapin",
                    "poopy",
                ],
            )
        )

    # start_time = time.time()
    # for i in range(1000):
    # Solvers.wordSearch(testGrid, testWords)
    # print("time: ", (time.time()-start_time)/1000)
