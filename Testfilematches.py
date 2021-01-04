import cv2
import numpy as np
import utilies as ut


img1 = cv2.imread('30schild.jpg')
img2 = cv2.imread('30schildtraining.jpg')

imgBlur1 = cv2.GaussianBlur(img1,(7,7),1)
imgBlur2 = cv2.GaussianBlur(img2,(7,7),1)

imgGray1 = cv2.cvtColor(imgBlur1, cv2.COLOR_BGR2GRAY)
imgGray2 = cv2.cvtColor(imgBlur2, cv2.COLOR_BGR2GRAY)




#oriented FAST and Rotatedbrief ist ein feature detection algorithmus. FAST nimmt Pixel P und 16 umliegende Punkte. diese werden
#überprüft, ob sie dunkler, ähnlich oder heller sind.
orb = cv2.ORB_create(nfeatures=3000)


#kp1,2 sind die Keypoints der beiden Bilder. descriptors are a 500x32 array of numbers
kp1, des1 = orb.detectAndCompute(imgGray1, None)
kp2, des2 = orb.detectAndCompute(imgGray2, None)



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
img3 = cv2.drawMatchesKnn(imgGray1,kp1,imgGray2,kp2,good,None,flags=2)

#keypoints zeichnen
imgKp1 = cv2.drawKeypoints(imgGray1,kp1,None)
imgKp2 = cv2.drawKeypoints(imgGray2,kp2,None)


imgStacked = ut.stackImages(0.8,[imgKp1,img1,imgKp2,img2,img3])
cv2.imshow('Stacked',img3)


cv2.waitKey(0)
