import cv2
import numpy as np
import string
import random
import imutils
from PIL import ImageFont, ImageDraw, Image 
import os

# generates images to be used to train model on
# model will be used in word search solver app

def generateLetter(letter, offset, noise, blur, rotation, font, textColour, backgroundColour):
    kernel = np.ones((blur, blur),np.float32) / (blur**2)
    margin = 15 # to prevent black barring from rotation
    img = np.empty((32+margin*2, 32+margin*2), dtype = np.uint8)
    img.fill(backgroundColour)
    img = drawText(letter, img, font, textColour)
    img = imutils.rotate(img, round(rotation))

    img = cv2.filter2D(img,-1,kernel)
    _, img = cv2.threshold(img,240,255,cv2.THRESH_BINARY)
    # img = img[margin:-margin, margin:-margin]
    # for x in range(len(img)): # add noise
    #     for y in range(len(img[x])):
    #         pass
            # img[x][y] += min(max(random.randint(-noise, noise), 0), 255)
    return img

def drawText(text, inputImg, font, textColour):
    text_to_show = text

    # Load image in OpenCV  
    image = inputImg

    # Pass the image to PIL  
    pil_im = Image.fromarray(inputImg)  

    draw = ImageDraw.Draw(pil_im)
    # use a truetype font  
    font = ImageFont.truetype(font, 30)  

    # Draw the text  
    draw.text((inputImg.shape[1]*0.4, inputImg.shape[0]*0.4), text_to_show, font=font, fill=(textColour))  

    return np.array(pil_im, dtype = np.uint8)

def crop(img):
    img = 255-img
    cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(cnts) >= 1:
        letCnt = max(cnts, key = lambda x:cv2.contourArea(x))
        # debugImg = np.ones((32, 32))
        # cv2.drawContours(debugImg, [letCnt], 0, 0, 1)
        # cv2.imshow("Img", debugImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x,y,w,h = cv2.boundingRect(letCnt)
        maxSize = max(w, h) * 1.1
        midX = x+w/2
        midY = y+h/2
        x = abs(int(midX-maxSize/2))
        y = abs(int(midY-maxSize/2))
        w = int(maxSize)
        h = int(maxSize)
        crop = img[y:y+h, x:x+w]
        return 255-cv2.resize(crop, (32, 32)).astype(np.uint8)
    else:
        return 255-img.astype(np.uint8)

if __name__ == "__main__":
    letters = string.ascii_letters
    textSize = range(1, 2)
    noise = range(0, 15)
    blur = range(1, 3)
    rotation = 10 # degrees
    font= os.listdir("fonts/")
    print(font)
    textColour = [0]
    backgroundColour = [255]
    offset = range(-3, 3) # size of text as proportion of image

    toGen = 100000
    npAll = np.empty((toGen, 1024), dtype = np.uint8)
    npNames = np.empty(toGen, dtype = np.int8)
    for i in range(toGen):
        l = random.choice(letters)
        im = generateLetter(letter = l,
                            offset = (0, 0),
                            noise = random.choice(noise),
                            blur = random.choice(blur),
                            rotation = random.uniform(-rotation, rotation),
                            font = "fonts/"+random.choice(font),
                            textColour = random.choice(textColour),
                            backgroundColour = random.choice(backgroundColour)
                            )
        im = crop(im)
        npAll[i] = np.reshape(im, 1024)
        npNames[i] = letters.index(l)
        if i%100 == 0:
            print(i)
            cv2.imwrite(f"trainSetImg/{i}.png", im)

    np.save(f"trainSetNpOne/images.npy", npAll)
    np.save(f"trainSetNpOne/labels.npy", npNames)
    
        
