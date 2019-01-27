import numpy as np
import cv2
from matplotlib import pyplot as plt
im=cv2.imread("nepixart3.jpg")
avarea=[]
color=[0,0,255]
chsv=np.uint8([[color]])
colorhsv=cv2.cvtColor(chsv,cv2.COLOR_BGR2HSV)
colorhsv=colorhsv[0][0]
print colorhsv
"""while True:

        # Take each frame
        _, frame = cv2.imread()

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_color = np.array([0,100,100])
        upper_color = np.array([255,202,205])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        #gray= cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask)
        #ret,res = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        cv2.imshow('res',res)
        if cv2.waitKey(30)==27:
                break"""
cv2.imshow("im",im)
cv2.waitKey()
cv2.destroyAllWindows()
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

        # define range of blue color in HSV
lower_color = np.array([40,110,100])
upper_color = np.array([100,140,150])

        # Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_color, upper_color)

        # Bitwise-AND mask and original image
res = cv2.bitwise_and(im,im, mask= mask)
        #gray= cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',frame)
        #cv2.imshow('mask',mask)
        #ret,res = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#cv2.imshow('res',res)
#thresh=cv2.Canny(im,100,200)
thresh=cv2.Canny(res,100,200)
cv2.imshow("res",mask)
cv2.waitKey()
cv2.destroyAllWindows()
im3, contours1, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(thresh, contours1, -1 , (0,0,255), 3)
(x,y),radius = cv2.minEnclosingCircle(contours1[0])
center = (int(x),int(y))
r = int(radius)
for i in range(len(contours1)):
        avarea.append(cv2.contourArea(contours1[i]))
        (x,y),radius = cv2.minEnclosingCircle(contours1[i])
        if radius > r:
                center = (int(x),int(y))
                r = int(radius)
                ind = i
"""(x,y),radius = cv2.minEnclosingCircle(contours1[ind])
center = (int(x),int(y))
radius = int(radius)"""
# cv2.circle(im,center,radius,(0,255,0),2)
# print center[0],center[1]
#ret1,im=cap.read()
"""(x,y),radius = cv2.minEnclosingCircle(contours1[ind])
radius=int(radius)
d=2*radius
x,y=(int(x),int(y))"""
x,y,w,h=cv2.boundingRect(contours1[ind])
#cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
#cv2.circle(im,center,radius,(0,255,0),2)
cv2.imshow("im3",im)
cv2.waitKey()
img=im[y:y+h,x:x+w]

# cv2.imshow("im3",thresh)
# cv2.waitKey()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
mask = cv2.inRange(hsv, lower_color, upper_color)
img = cv2.bitwise_not(img,img,mask = mask)
thresh=cv2.Canny(img,100,200)
im3, contours1, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours1, -1 , (0,0,255), 3)
print len(contours1)
cv2.imshow("im3",thresh)
cv2.waitKey()