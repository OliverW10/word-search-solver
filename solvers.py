import string
import random
import copy
import time
import numpy as np


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
    def wordSearch(gridPlus, words):
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
        foundWords = {}
        for word in words:
            pass


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
