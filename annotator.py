import numpy as np
import cv2
import math


class Annotator:
    
    @staticmethod
    def annotate(img, cropPos, words, idealDist):
        print("annotator words", words)
        drawImg = img.copy()
        print("image size", drawImg.shape)
        p1 = [cropPos[0][0] * img.shape[1], cropPos[0][1] * img.shape[0]]  # top left
        p2 = [
            cropPos[2][0] * img.shape[1],
            cropPos[2][1] * img.shape[0],
        ]  # bottom right

        # p1 = ( min(_p1[0], _p2[0]), min(_p1[1], _p2[1]) )
        # p2 = ( max(_p1[0], _p2[0]), max(_p1[1], _p2[1]) )
        def unCropNum(n):
            return int(n * (p2[0] - p1[0]))

        def unCropPos(pos):
            # translates from position in the cropped image to position in the full image
            newX = pos[0] * (p2[0] - p1[0])
            newY = pos[1] * (p2[1] - p1[1])
            return (int(newX + p1[0]), int(newY + p1[1]))

        print("p1", p1, "    p2", p2)
        for word in words.keys():
            if len(words[word]) >= 1:
                # average letter size
                letterSize = sum([x.position[2] for x in words[word]])/len(word)
                letterSize = letterSize * 0.95

                # the centers of the first and lest letters
                letterPoints = [
                    [
                        words[word][0].position[0] + words[word][0].position[2]/2,
                        words[word][0].position[1] + words[word][0].position[3]/2

                    ],
                    [
                        words[word][-1].position[0] + words[word][-1].position[2]/2,
                        words[word][-1].position[1] + words[word][-1].position[3]/2
                    ],
                ]

                angle = -math.atan2(
                    letterPoints[1][1] - letterPoints[0][1],
                    letterPoints[1][0] - letterPoints[0][0],
                )  # angle between the first letter and the last

                line1 = [
                    [
                        letterPoints[0][0] + math.sin(angle) * letterSize,
                        letterPoints[0][1] + math.cos(angle) * letterSize,
                    ],
                    [
                        letterPoints[1][0] + math.sin(angle) * letterSize,
                        letterPoints[1][1] + math.cos(angle) * letterSize,
                    ],
                ]

                line2 = [
                    [
                        letterPoints[1][0] - math.sin(angle) * letterSize,
                        letterPoints[1][1] - math.cos(angle) * letterSize,
                    ],
                    [
                        letterPoints[0][0] - math.sin(angle) * letterSize,
                        letterPoints[0][1] - math.cos(angle) * letterSize,
                    ],
                ]

                print("before", line1, line2)
                line1 = [unCropPos(line1[i]) for i in range(2)]
                line2 = [unCropPos(line2[i]) for i in range(2)]
                print("after", line1, line2)
                drawImg = cv2.line(
                    drawImg,
                    line1[0],
                    line1[1],
                    (0, 255, 255),
                    math.ceil(img.shape[0] / 500),
                )
                drawImg = cv2.line(
                    drawImg,
                    line2[0],
                    line2[1],
                    (0, 255, 255),
                    math.ceil(img.shape[0] / 500),
                )

                print("ideal dist", idealDist)
                cv2.circle(drawImg, unCropPos(letterPoints[0]), int(unCropNum(idealDist)), (255, 0, 0), int(img.shape[0]/500))

                rad = unCropNum(letterSize)
                drawImg = cv2.ellipse(
                    drawImg,
                    unCropPos(letterPoints[0]),
                    (rad, rad),
                    0,
                    math.degrees(-angle) + 90,
                    math.degrees(-angle) + 270,
                    (0, 255, 255),
                    int(img.shape[0] / 500),
                )
                drawImg = cv2.ellipse(
                    drawImg,
                    unCropPos(letterPoints[1]),
                    (rad, rad),
                    0,
                    math.degrees(-angle) + 90,
                    math.degrees(-angle) - 90,
                    (0, 255, 255),
                    int(img.shape[0] / 500),
                )
            else:
                print(f"word {word} not found")
        # for l in lettersPlus:
        # 	x = int(lerp(cropPos[0][0], cropPos[2][0], l[1][0]) * img.shape[1])
        # 	y = int(lerp(cropPos[0][1], cropPos[2][1], l[1][1]) * img.shape[0])
        # 	img = cv2.putText(img, l[0], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX , 2, 0, 2, cv2.LINE_AA)
        # 	# cv2.rectangle(img,  (l[1][0], l[1][1]), (l[1][0]+l[1][2], l[1][1]+l[1][3]),  20, 3)
        cv2.rectangle(drawImg, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 4)
        return np.flip(drawImg, 0)

    @staticmethod
    def drawGrid(grid, size=(800, 800)):
        img = np.zeros(size)
        img.fill(255)
        for i in range(len(grid)):
            for j in range(len(grid)):
                cv2.putText(
                    img,
                    grid[j][i],
                    (
                        int(((i + 0.5) / len(grid)) * size[0]),
                        int(((j + 0.5) / len(grid)) * size[1]),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    size[0] * 0.001,
                    0,
                )
        return img


if __name__ == "__main__":
    from imageReader import ImageProcessing
    from solvers import Solvers
    import csv

    words = {}
    with open(
        "/home/olikat/word-search-solver/test_images/labels.csv", newline=""
    ) as csvfile:
        wordreader = csv.reader(csvfile)
        for row in wordreader:
            words[int(row[0])] = row[1:]

    img = ImageProcessing.loadImg("test_images/1.0.jpg")
    _, gridPlus, multiGrid = ImageProcessing.processImage(img, [[0, 0], [1, 0], [1, 1]])
    foundWords = Solvers.wordSearch(multiGrid, words[1])
    annotatedImg = Annotator.annotate(
        img, gridPlus, [[0, 0], [1, 0], [1, 1]], foundWords
    )
    cv2.imshow("annoted image", cv2.resize(annotatedImg, None, None, fx=0.4, fy=0.4))
    cv2.waitKey()
