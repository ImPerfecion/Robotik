import cv2
import numpy as np


#Dieser Kurs nutzt Farben als Merkmal, statt Kanten. Hier werden die Farbwerte eingegeben
#die in der ColorPickerScript Funktion ermittelt wurden.
def thresholding(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #lowerWhite = np.array([0,00,60])
    #upperWhite = np.array([ 255, 67, 255 ])
    lowerWhite = np.array([80,0,0])
    upperWhite = np.array([ 255, 160, 255 ])

    maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
    return maskWhite


#Ziel ist es, die Straße knapp vor/unter dem Wagen zu detektieren. Hierfür wird in dieser 
#Funktion die Vogelperspektive durch eine Transformationsmatrize hergestellt, die mit 
#manuell ermittelten Werten gefüttert wird.
def warpIMG(img, points,w,h,inv = False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix,(w,h))
    return imgWarp



def nothing(a):
    pass

#Hier müssen bei wT und hT die Seitenformatwerte des Videos eingestellt werden.
#def initializeTrackbars(initialTracbarVals, wT=1280, hT=720):
def initializeTrackbars(initialTracbarVals, wT=640, hT=480):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initialTracbarVals[0], wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initialTracbarVals[2], wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initialTracbarVals[3], hT, nothing)

def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop), (widthBottom, heightBottom), (wT-widthBottom, heightBottom)])
    return points

#Visualisiert die Verzerrungspunkte, damit das Verzerrungsmaß eingeschätzt werden kann
def drawPoints(img, points):
    for x in range(4):
        cv2.circle(img,(int(points[x][0]), int(points[x][1])), 15, (0,0,255), cv2.FILLED)
    return img


#In dieser Funktion werden die Kurvenwerte der binarisierten Strecke ermittelt. 
#Pixel die weiß sind, haben einen Wert von 255, überall wo sie Schwarz sind 0. 
# Die Pixelwerte werden spaltenweise aufsummiert. Eine linksgerichtete Kurve hat dann
# links des Mittelpunktes einen höheren Wert und eine rechtsgerichtete Kurve hat rechts des Mittelpunktes 
# einen höheren Wert. 
def getHistogram(img,minPer=0.1,display= False,region=1):

    #Aufaddieren der Spaltenhelligkeitswerte 
    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        #in diesem Fall wird der angezeigte Bildausschnitt proz. verkleinert
        histValues = np.sum(img[img.shape[0]//region:,:], axis=0)  
    maxValue = np.max(histValues)
    minValue = minPer*maxValue #Noise cancelling
    #Index kreieren der Werte für "nicht straße" in Kurven
    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray)) #Mittelpunkt finden
    
    #zwecks Visualisierung, wird dieser Mittelpunkt als Kreis dargestellt
    if display:
        imgHist = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255//region),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)#Mittelwert
        return basePoint, imgHist

    return basePoint

    
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
