import cv2
import numpy as np
import utilies


frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass




#install trackbars
cv2.namedWindow("Parameters")
cv2.resizeWindow("Paramters", 640, 240)
cv2.createTrackbar("Threshold1","Parameters", 12, 255, empty)
cv2.createTrackbar("Threshold2","Parameters", 23, 255, empty)
cv2.createTrackbar("Area","Parameters", 5000, 30000, empty)


while True:
        success, img = cap.read()
        imgContour = img.copy()

        imgBlur = cv2.GaussianBlur(img,(7,7),1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

        threshold1 = cv2.getTrackbarPos("Threshold1","Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2","Parameters")
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

        kernel = np.ones((5,5))
        imgDil = cv2.dilate(imgCanny,kernel, iterations = 1)

        utilies.getContours(imgDil, imgContour)

        imgStacked = utilies.stackImages(0.5,([img, imgGray, imgCanny],
                                            [imgDil, imgContour, imgContour]))


        cv2.imshow("Video", imgStacked)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break