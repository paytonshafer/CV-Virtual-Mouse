import cv2
import time
import numpy as np
from hand_track_module import HandDetector
import pyautogui as autopy

#set width and height of window
wCam, hCam = 640, 480
frameR = 100 #frame reduction
smoothening = 5
plocx, plocy = 0, 0
clocx, clocy = 0, 0

#set pTime
pTime = 0

#Set camera and dimensions
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

#Initalize Hand detector
detector = HandDetector(maxHands=1)

wScr, hScr = autopy.size()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) #orignal soltion to flip whole image
    
    #get hand landmarks
    img = detector.getHands(img)
    lmlist, bbox = detector.getPos(img, draw=False)
    
    #We will need the tip of the index and middle finger
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:] #index finger 
        x2, y2 = lmlist[12][1:] #middle finger 

        #check what fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255,0,255), 2)

        #Only index finger is moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            #convert coordinates and smooth values
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            #smoothening
            clocx = plocx + (x3 - plocx)/smoothening
            clocy = plocy + (y3 - plocy)/smoothening

            #move mouse
            autopy.moveTo(clocx, clocy) #this work if whole image fliped
            #autopy.moveTo(wScr-x3, y3) #this works if whole image is NOT fliped

            plocx, plocy = clocx, clocy


        #Both fingers is click mode
        if fingers[1] == 1 and fingers[2] == 1:
            #find distance between fingers
            length, img = detector.findDistance(8, 12, img)
            #if short dist then click
            if length < 35:
                cv2.circle(img, (x1, y1), 4, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 4, (0, 255, 0), cv2.FILLED)
                autopy.click()

    #Adding frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    #Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)