import cv2
import numpy as np
import string
import random
import imutils

# generates images to be used to train model on
# model will be used in word search solver app

def generateLetter(letter, textSize, thickness, offset, noise, blur, rotation, font, textColour, backgroundColour):
    kernel = np.ones((blur, blur),np.float32) / (blur**2)
    margin = 10 # to prevent black barring from rotation
    img = np.empty((32+margin*2, 32+margin*2), dtype = np.uint8)
    img.fill(backgroundColour)
    cv2.putText(img, letter, (6 + offset[0]+int(margin), 25 + offset[1]+int(margin)), font, textSize, textColour, thickness, cv2.LINE_AA)
    img = imutils.rotate(img, round(rotation))
    img = img[margin:-margin, margin:-margin]

    img = cv2.filter2D(img,-1,kernel)
    for x in range(len(img)):
        for y in range(len(img[x])):
            pass
            # img[x][y] += min(max(random.randint(-noise, noise), 0), 255)
    return img


if __name__ == "__main__":
    letters = string.ascii_letters
    textSize = range(1, 2)
    noise = range(0, 15)
    blur = range(1, 4)
    rotation = 10 # degrees
    # https://codeyarns.com/tech/2015-03-11-fonts-in-opencv.html
    font= [cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_ITALIC
           ]
    textColour = range(1) # all text will be black on white
    backgroundColour = range(150, 255)
    offset = range(-3, 3) # size of text as proportion of image
    thickness = range(1, 3)

    toGen = 100000
    npAll = np.empty((toGen, 32, 32), dtype = np.uint8)
    npNames = np.empty(toGen, dtype = np.int8)
    for i in range(toGen):
        l = random.choice(letters)
        im = generateLetter(letter = random.choice(l),
                            textSize = random.choice(textSize),
                            thickness = random.choice(thickness),
                            offset = (random.choice(offset), random.choice(offset)),
                            noise = random.choice(noise),
                            blur = random.choice(blur),
                            rotation = random.uniform(-rotation, rotation),
                            font = random.choice(font),
                            textColour = random.choice(textColour),
                            backgroundColour = random.choice(backgroundColour)
                            )
        
        npAll[i] = im
        npNames[i] = letters.index(l)
        if i%100 == 0:
            print(i)
            cv2.imwrite(f"trainSetImg/{l}-{i}.png", im)

    np.save(f"trainSetNpOne/images.npy", npAll)
    np.save(f"trainSetNpOne/labels.npy", npNames)
    
        
