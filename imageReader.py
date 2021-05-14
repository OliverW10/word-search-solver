import cv2
import numpy as np
import string
import random
from letterReader import LetterReader
import image_to_numpy
import time

def clampPos(x, y, imgSize):
    return max(min(x, imgSize[1]), 0), max(min(y, imgSize[0]), 0)

@dataclass
class Letter:
    # class to store the data for a letter
    letter: str
    position: list
    allLetters: list

    @property
    def letterStr() -> str:
        return string.ascii_letters[int(letters[i])]

class ImageProcessing:
    # both as a multiple of the image size
    minContourSize = 0.0001
    maxContourSize = 0.1

    # increases the constant untill there are less than maxContoursNum contours
    maxContoursNum = 2000

    maxBoxSize = 1
    minBoxSize = 0

    letReader = LetterReader()

    shrinkRatio = 0.5

    def loadImg(fileName):
        return image_to_numpy.load_image_file(fileName)

    def processImage(img, pos, debug=False, progressCallback=lambda *x: x):
        # flippedImg = np.flip(img, 0)
        progressCallback(10, "Pre-processing")
        croppedImg = ImageProcessing.cropToRect(img, pos=pos)
        smallImg = cv2.resize(
            croppedImg,
            None,
            fx=ImageProcessing.shrinkRatio,
            fy=ImageProcessing.shrinkRatio,
        )
        # incriments c (the constant for adaptive threshold) untill there are less than maxContoursNum contours
        c = 10
        newImg = ImageProcessing.preProcessImg(smallImg, constant=c, debug=debug)
        print("c= 10 contours num", ImageProcessing.checkContours(newImg))
        while ImageProcessing.checkContours(newImg) > ImageProcessing.maxContoursNum:
            c += 10
            newImg = ImageProcessing.preProcessImg(smallImg, constant=c, debug=debug)
            print("c= ", c, "contours num", ImageProcessing.checkContours(newImg))

        grid, letters, allGrids = ImageProcessing.findLetters(
            newImg, debug, progressCallback
        )

        return ImageProcessing.makeGridFull(grid, letters, allGrids)

    def boxOverlap(box1, box2):
        # finds the total area of overlap between two rects (x, y, w, h)
        if ImageProcessing.boxCollide(box1, box2):
            # assumes boxes dont have negative dimentions
            actualWidth = min(box1[0], box2[0]) + max(
                box1[0] + box1[2], box2[0] + box2[2]
            )
            actualHeight = min(box1[1], box2[1]) + max(
                box1[1] + box1[3], box2[1] + box2[3]
            )
            totalWidth = box1[2] + box2[2]
            totalHeight = box1[3] + box2[3]
            return (totalWidth - actualWidth) * (totalHeight - actualHeight)
        else:
            return 0

    def boxCollide(rect1, rect2):
        return not (
            rect2[0] > rect1[0] + rect1[2]
            or rect2[0] + rect2[2] < rect1[0]
            or rect2[1] > rect1[1] + rect1[3]
            or rect2[1] + rect2[3] < rect1[1]
        )

    def preProcessImg(img, constant=10, debug=False):
        if debug:
            timeCheckpoints.append(["start preProcessImg", time.time()])
        size = img.shape  # height first
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.medianBlur(img, int(int(size[0] * 0.003) / 2) * 2 + 1, img)
        # this errors if you dont greyscale first
        cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(int(size[0] * 0.075) / 2) * 2 + 1,
            constant,
            img,
        )
        if debug:
            timeCheckpoints.append(["finished preProcessImg", time.time()])
        return img

    def checkContours(img):
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return len(contours)

    def findLetters(img, debug=False, callback=lambda *x: x):
        # cv2.imshow("findLetters image", img)
        # cv2.waitKey()
        callback(10, "Finding Possible Letters")
        if debug:
            timeCheckpoints.append(["got to findLetters", time.time()])
        if debug:
            drawImg = img.copy()

        # first find all contours that big enough to be letters
        lettersContours = []
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if (
                ImageProcessing.maxContourSize
                > cv2.contourArea(cnt) / (img.shape[0] * img.shape[1])
                > ImageProcessing.minContourSize
            ):
                lettersContours.append(cnt)

        if len(lettersContours) < 5:
            print("\n\n\n NO LETTERS FOUND \n\n")

        if debug:
            timeCheckpoints.append(["found contours", time.time()])
        callback(10, "Removing False Positives 1/2")
        # go through every contour and get the section of img around it
        # done first so that they can be keras'ed as a batch
        letterImgs = np.empty([len(lettersContours), 32, 32])
        letterPositions = np.empty([len(lettersContours), 4])
        badCnts = []
        avgSize = 0
        for i, cnt in enumerate(lettersContours):
            # find a square centered around the contour
            x, y, w, h = cv2.boundingRect(cnt)
            midX = x + w / 2
            midY = y + h / 2
            maxSize = max(w, h) * 1.0
            x = int(midX - maxSize / 2)
            y = int(midY - maxSize / 2)
            w = int(maxSize)
            h = int(maxSize)

            # crop the image to the square
            crop = img[y : y + h, x : x + w]
            if (
                crop.shape[0] > 5 and crop.shape[1] > 5
            ):  # filter out tiny contours that somehow slipped through
                cntSize = cv2.contourArea(cnt) / (img.shape[0] * img.shape[1])
                # keep track of the average
                if i - len(badCnts) == 0:
                    avgSize = cntSize
                else:
                    avgSize = ((avgSize * i - len(badCnts)) + cntSize) / (
                        i - len(badCnts) + 1
                    )

                # save the box position and the image
                letterPositions[i] = [
                    x / img.shape[1],
                    y / img.shape[0],
                    w / img.shape[1],
                    h / img.shape[0],
                ]
                letterImgs[i] = cv2.resize(crop, (32, 32))
            else:
                # print("\nweird shaped contour  ", [x, y, w, h])
                badCnts.append(i)
        # print("average cnt size: ", avgSize)
        if debug:
            timeCheckpoints.append(["finished boxing contours", time.time()])
        callback(10, "Removing False Positives 2/2")
        # find any overlaps between letterContours and remove the one whose size is furthest from the average
        for i1, rect1 in enumerate(letterPositions):
            for i2, rect2 in enumerate(letterPositions):
                if i1 != i2:
                    if ImageProcessing.boxCollide(rect1, rect2):
                        cnt1Size = cv2.contourArea(lettersContours[i1]) / (
                            img.shape[0] * img.shape[1]
                        )
                        cnt2Size = cv2.contourArea(lettersContours[i2]) / (
                            img.shape[0] * img.shape[1]
                        )
                        # print("found overlapping boxes, with areas ", cnt1Size, " and ", cnt2Size)
                        if abs(cnt1Size - avgSize) < abs(cnt2Size - avgSize):
                            badCnts.append(i2)
                        else:
                            badCnts.append(i1)

        if debug:
            timeCheckpoints.append(["finished box check", time.time()])

        # get the letter for each contour and its confidence
        letters, neighbours = ImageProcessing.letReader.readLetters(
            letterImgs, callback
        )
        if debug:
            timeCheckpoints.append(["finished letter classification", time.time()])
        # combine the letters, their positions and the confidence and removes all badCnts
        lettersPlus = []
        for i in range(len(letters)):
            if not i in badCnts:
                lettersPlus.append(
                    (
                        string.ascii_letters[int(letters[i])],
                        letterPositions[i],
                        int(letters[i]),
                        neighbours[i],
                    )
                )
                if debug:
                    xI, yI, wI, hI = letterPositions[i]
                    x, y, w, h = (
                        xI * img.shape[1],
                        yI * img.shape[0],
                        wI * img.shape[1],
                        hI * img.shape[0],
                    )
                    cv2.rectangle(
                        drawImg, (int(x), int(y)), (int(x + w), int(y + h)), 20, 3
                    )

        # use the among of found contuors to determin the size of the grid
        # print("letters num before: ", len(letterPositions))
        # print("letters num after: ", len(lettersPlus))
        gridSize = round(len(lettersPlus) ** 0.5)
        print("gridSize: ", gridSize)
        callback(10, "Forming Grid")
        # position all letters in grid
        grid = []
        gridPlus = []
        gridPossibilities = []
        YsortedLetters = sorted(lettersPlus, key=lambda x: x[1][1])
        for row in range(gridSize):
            rowLettersPlus = sorted(
                YsortedLetters[row * gridSize : (row + 1) * gridSize],
                key=lambda x: x[1][0],
            )
            rowLetters = [letter[0] for letter in rowLettersPlus]
            rowPossibilities = [letter[3] for letter in rowLettersPlus]
            # print(rowLetters)
            grid.append(rowLetters)
            gridPlus.append(rowLettersPlus)
            gridPossibilities.append(rowPossibilities)
        if debug:
            timeCheckpoints.append(["organised letters into grid", time.time()])

        return grid, gridPlus, gridPossibilities

    def makeGridFull(grid, gridPlus, gridPossibilities):
        gridSize = len(grid[0])
        emptySquare = (" ", [0, 0, 0, 0], 53, [53] * len(gridPlus[0][0][-1]))
        for i in range(gridSize):
            if i < len(grid):
                while len(grid[i]) < gridSize:
                    grid[i].append(" ")
                    gridPlus[i].append(emptySquare)
                    gridPossibilities[i].append([27] * len(gridPossibilities[0][0]))
            else:
                grid.append([" "] * gridSize)
                gridPlus.append([emptySquare] * gridSize)
                gridPossibilities.append(
                    [[27] * len(gridPossibilities[0][0])] * gridSize
                )
        return grid, gridPlus, gridPossibilities

    def cropToRect(img, pos):
        # takes the positions of the four to crop to
        posNp = np.array(pos)
        # gets the pos of the two opposet corners
        p1 = (int(posNp[0][0] * img.shape[1]), int(posNp[0][1] * img.shape[0]))
        p2 = (int(posNp[2][0] * img.shape[1]), int(posNp[2][1] * img.shape[0]))
        # clamps them into the bounds of the iamge
        p1 = clampPos(*p1, img.shape)
        p2 = clampPos(*p2, img.shape)
        # orders them so the top left one is first and bottom right is second
        cropPos = ImageProcessing.checkCropPos(p1, p2)
        return img[cropPos[2] : cropPos[3], cropPos[0] : cropPos[1]]

    def checkCropPos(p1, p2):
        # makes the first of each axis of the cropPos to be the smallest
        return [
            min(p1[0], p2[0]),
            max(p1[0], p2[0]),
            min(p1[1], p2[1]),
            max(p1[1], p2[1]),
        ]

    def cropToPos(img, pos):
        srcTri = np.array([pos[0], pos[1], pos[2]]).astype(np.float32) * np.array(
            [img.shape[1], img.shape[0]]
        ).astype(np.float32)
        dstTri = np.array([[0, 0], [1, 0], [1, 1]]).astype(np.float32) * np.array(
            [img.shape[1], img.shape[0]]
        ).astype(np.float32)

        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(img, warp_mat, (img.shape[1], img.shape[0]))
        return warp_dst


def lerp(a, b, n):
    return (n * a) + ((1 - n) * b)


if __name__ == "__main__":
    from annotator import Annotator
    import os

    fileNames = os.listdir("test_images")
    imageNames = []
    for i in fileNames:  # could probrobly be done in one line
        if i.lower().endswith(".png") or i.lower().endswith(".jpg"):
            imageNames.append(i)

    checkpointAverages = []
    imageOutputs = {}
    for imageName in [imageNames[0]]:
        print(imageName)
        img = cv2.imread("test_images/" + i)  # "tests/originals/4.png"
        timeCheckpoints = [["start", time.time()]]
        grid, letters, possibleGrids = ImageProcessing.processImage(
            img, [[0, 0], [1, 0], [1, 1]], True
        )

        cv2.imwrite("results/" + imageName, img)
        img = cv2.resize(img, None, fx=0.3, fy=0.3)

        temp = np.array(possibleGrids, dtype=np.uint8)  # possibleGrids as an np array
        imageOutputs[imageName] = temp
        # print(imageOutputs[imageName)

        # cv2.imshow("Img", img)

        # cv2.imshow("grid", Annotator.drawGrid(grid))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    np.savez("test_images/output/data.npz", **imageOutputs)
    for i in range(len(checkpointAverages)):
        print("avg", checkpointAverages[i][0], checkpointAverages[i][1] / 5)
    print("avg total time", sum(x[1] for x in checkpointAverages) / 5)
