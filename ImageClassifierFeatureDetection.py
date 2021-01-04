import cv2
import numpy as np
import utilies as ut
import os

#Loop through folder and import all images
path = 'Featuresdetection/Games'

#oriented FAST and Rotatedbrief ist ein feature detection algorithmus. FAST nimmt Pixel P und 16 umliegende Punkte. diese werden
#überprüft, ob sie dunkler, ähnlich oder heller sind.
orb = cv2.ORB_create(nfeatures=1500)

#---------Import images-------------------------------
images = []
classNames = []
MyPctList = []

myList = os.listdir(path)

#find picture types in folder
for pct in myList:
    #get last 4 chars of string
    length = len(pct)
    last_chars = pct[length - 4 : ]
    if last_chars == '.jpg':
        MyPctList.append(pct)


print('Total Classes Detected', len(MyPctList))



#Import images from array
for cl in MyPctList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)    
    classNames.append(os.path.splitext(cl)[0])

print(classNames)


def findDes(images):
    desList=[]
    for img in images:
            #kp sind die Keypoints der beiden Bilder. descriptors are a 500x32 array of numbers
            kp,des = orb.detectAndCompute(img,None)
            desList.append(des)
    return desList

def findID(img, desList,thres=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    #brute force matcher. compares descriptors and uses the nearest neighbor method. 
    # k defines the number of returned features
    bf = cv2.BFMatcher()
    matchlist=[]
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k = 2)
            good = []

            #vergleicht die ermittelten Matches und hängt sie an Array an
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchlist.append(len(good))
    except:
        pass

    #print(matchlist)
    if len(matchlist)!=0:
        if max(matchlist) > thres:            
            finalVal = matchlist.index(max(matchlist))
    return finalVal


desList = findDes(images)
print(len(desList))


cap = cv2.VideoCapture(0)


while True:
    
    success, img2 = cap.read()
    imgOriginal = img2.copy()

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    id = findID(img2,desList)
    if id != -1:
        cv2.putText(imgOriginal, classNames[id], (50,50), cv2.FONT_HERSHEY_COMPLEX,2, (0,255,0),3)
    cv2.imshow('img2',imgOriginal)
    cv2.waitKey(1)

######ab hier möchte murtaza die eingespeic
# herten Bilder mit Webcam Bildern vergleichen.
#  Ich muss also Schilder ausdrucken und basteln


"""
#brute force matcher. compares descriptors and uses the nearest neighbor method. 
# k defines the number of returned features
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k = 2)

good = []

#vergleicht die ermittelten Matches und hängt sie an Array an
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print(len(good))


#Zeichnen der ermittelten matches
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

#keypoints zeichnen
imgKp1 = cv2.drawKeypoints(img1,kp1,None)
imgKp2 = cv2.drawKeypoints(img2,kp2,None)


imgStacked = ut.stackImages(0.8,[imgKp1,img1,imgKp2,img2,img3])
cv2.imshow('Stacked',img3)


cv2.waitKey(0)
 """