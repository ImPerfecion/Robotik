#MurtazaVersion
#source:https://www.youtube.com/channel/UCYUjYU5FveRAscQ8V21w81A
#source:https://www.murtazahassan.com/courses/self-driving-car-using-raspberry-pi/

import cv2
import numpy as np
import utilies

curveList = []
avgVal = 10


#Funktion um die Kurvenlage durch ein verzerrtes Bild der Strecke zu determinieren
def getLaneCurve(img,display=2):
    #Kopie zum bearbeiten erstellen
    imgCopy = img.copy()
    imgResult = img.copy()

    #Schwellwert kontrastiertes Bild
    imgThres = utilies.thresholding(img)


    #Trackbars einfügen
    hT,wT,c = img.shape
    points = utilies.valTrackbars()
    #Bild krümmen
    imgWarp = utilies.warpIMG(imgThres,points,wT,hT)
    imgWarpPoints = utilies.drawPoints(imgCopy,points)

    #Krümmungswerte ermitteln
    middlePoint,imgHist = utilies.getHistogram(imgWarp,display=True,minPer=0.5,region=4)
    curveAveragePoint,imgHist = utilies.getHistogram(imgWarp,display=True,minPer=0.9)
    #curveRaw zeigt die Intensität der Kurve an. Wenn der Mittelwert 240 ist, geht in einer starken 
    #Linkskurve zum Beispiel der Wert ins negative. In einer Rechtskurve stärker ins Positive
    curveRaw = curveAveragePoint - middlePoint

    #Mittelwerte der Kurvenwerte ermitteln für flüssigere Darstellung
    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))

    #Anzeigeeinstellungen (reinkopiert)
    if display != 0:
        imgInvWarp = utilies.warpIMG(imgWarp, points, wT, hT,inv = True)
        imgInvWarp = cv2.cvtColor(imgInvWarp,cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT//3,0:wT] = 0,0,0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult,1,imgLaneColor,1,0)
        midY = 450
        cv2.putText(imgResult,str(curve),(wT//2-80,85),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
        cv2.line(imgResult,(wT//2,midY),(wT//2+(curve*3),midY),(255,0,255),5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY-25), (wT // 2 + (curve * 3), midY+25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve//50 ), midY-10),
                        (w * x + int(curve//50 ), midY+10), (0, 0, 255), 2)
        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        #cv2.putText(imgResult, 'FPS '+str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230,50,50), 3);
    if display == 2:
        #Kramt Fenster zusammen
        imgStacked = utilies.stackImages(0.7,([img,imgWarpPoints,imgWarp],
                                            [imgHist,imgLaneColor,imgResult]))
        cv2.imshow('ImageStack',imgStacked)
    elif display == 1:
        cv2.imshow('Result',imgResult)

    #cv2.imshow('Thres', imgThres)
    #cv2.imshow('Warp', imgWarp)
    #cv2.imshow('Warp Points', imgWarpPoints)
    #cv2.imshow('Histogram', imgHist)
    return curve



#Der Kurs will ein modular ansteuerbares Gerät erschaffen
#daher die main Bedingung
if __name__ == '__main__':
    #cap = cv2.VideoCapture("test2.mp4")
    cap = cv2.VideoCapture("vid1.mp4")
    #initialTrackbarVals = [0, 179, 0, 188]
    initialTrackbarVals = [102, 80, 20, 214]
    utilies.initializeTrackbars(initialTrackbarVals)
    frameCounter = 0
    while True:
        #Bild wiederholen
        frameCounter +=1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, img = cap.read() 
        img = cv2.resize(img,(480,240))
        #mit dem displaywert bestimmt man, wieviel angezeigt wird
        #0 = trackbar, 1 = Result, 2 = alles
        curve =  getLaneCurve(img,display=2)
        print(curve)
        if not success:
            break
        #cv2.imshow('Vid',img)   
        if cv2.waitKey(1) == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
