import numpy as np
import cv2
import random
import string
import os
import time

def loadDatasetNpz():
    read_img_start = time.time()
    with open("./trainSetNp/dataZ.npz", "rb") as f:
        data = np.load(f)
        images = data["images"].astype(np.float32)
        labels = data["labels"].astype(np.float32)
        print("load time: " + str(time.time() - read_img_start), "for", len(labels), "samples")
        return images, labels
    
def makeKnn(dataPer=0.01):
    # dataPer is the percentage of the data to use
    allImages, allLabels = loadDatasetNpz()
    testNum = int((dataPer)*len(allImages))+1
    trainImages, trainLabels = allImages[0:-testNum], allLabels[0:-testNum]
    testImages, testLabels = allImages[-testNum:-1], allLabels[-testNum:-1]

    knn = cv2.ml.KNearest_create()
    knn.train(trainImages, cv2.ml.ROW_SAMPLE, trainLabels)

    return knn, testImages, testLabels

def testKnn():
    total_start_time = time.time()
    start_time = time.time()
    knn, testImages, testLabels = makeKnn(0.01)
    print("train time ", time.time()-start_time)

    acc = 0
    start_time = time.time()
    ret,result,neighbours,dist = knn.findNearest(testImages,k=10)
    # print("result: ", result)
    # print("neighbours: ", neighbours)
    # print("dist: ", dist)
    predict_time = time.time()-start_time
    print("predict time ", predict_time)

    start_time = time.time()
    correct = 0
    for i in range(len(result)):
        # print(string.ascii_letters[int(testLabels[i])], "  ", string.ascii_letters[int(result[i][0])])
        if result[i] == testLabels[i]:
            correct += 1
    print("test samples: ", len(testLabels))
    print("eval time ", time.time()-start_time)
    accuracy = 100*correct/len(result)
    print(f"\naccuracy {round(accuracy, 2)}%")
    print("total time ", time.time()-total_start_time)
    return accuracy, predict_time

if __name__ == "__main__":
    testKnn()