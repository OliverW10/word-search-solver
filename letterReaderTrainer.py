import numpy as np
import cv2
import random
import string
import os
import time

def loadDatasetNpAll(*args): # slightly higher file size but much quicker loading
    readimg_start = time.time()
    images = np.load("trainSetNpOne/images.npy").astype(np.float32)
    labels = np.load("trainSetNpOne/labels.npy").astype(np.float32)
    print("load time: " + str(time.time() - readimg_start))
    return images, labels
    
if __name__ == "__main__":
    start_time = time.time()
    allImages, allLabels = loadDatasetNpAll()
    trainImages, trainLabels = allImages[0:-1000], allLabels[0:-1000]
    testImages, testLabels = allImages[-1000:-1], allLabels[-1000:-1]
    print("all ", len(allImages), len(allLabels))
    print("train ", len(trainImages), len(trainLabels))
    print("test ", len(testImages), len(testLabels))
    # print("\n", testLabels)
    # cv2.imshow("test", images3[5])
    # print(string.ascii_letters[labels3[5]])

    # images = images / 255

    knn = cv2.ml.KNearest_create()
    knn.train(trainImages, cv2.ml.ROW_SAMPLE, trainLabels)
    acc = 0
    ret,result,neighbours,dist = knn.findNearest(testImages,k=10)
    correct = 0
    for i in range(len(result)):
        print(testLabels[i], "  ", result[i][0])
        if result[i] == testLabels[i]:
            correct += 1
    print(f"\naccuracy {round(100*correct/len(result), 2)}%")