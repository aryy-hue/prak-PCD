import cv2
import numpy as np
def nothing(x): # Function to do nothing
    pass

cam = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H","Trackbars",0,179,nothing) #HUE
cv2.createTrackbar("L-S","Trackbars",0,255,nothing) #SATURATION
cv2.createTrackbar("L-V","Trackbars",0,255,nothing) #VALUE
cv2.createTrackbar("U-H","Trackbars",179,179,nothing)
cv2.createTrackbar("U-S","Trackbars",255,255,nothing)
cv2.createTrackbar("U-V","Trackbars",255,255,nothing)

while True:
    _,frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #BLUE GREEN RED(BGR) to HUE SATURATION VALUE(HSV)
    l_h = cv2.getTrackbarPos("L-H","Trackbars") #LOWER
    l_s = cv2.getTrackbarPos("L-S","Trackbars")
    l_v = cv2.getTrackbarPos("L-V","Trackbars")
    u_h = cv2.getTrackbarPos("U-H","Trackbars") #UPPER
    u_s = cv2.getTrackbarPos("U-S","Trackbars") 
    u_v = cv2.getTrackbarPos("U-V","Trackbars")
    lower_color = np.array([l_h,l_s,l_v]) #Set to lower_color
    upper_color = np.array([u_h,u_s,u_v]) #Set to upper_color
    mask = cv2.inRange(hsv,lower_color,upper_color)
    result = cv2.bitwise_and(frame,frame,mask=mask) #Hasil dari proses
    
    cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("result",result)
    key = cv2.waitKey(1)
    if key == 27:
        break #Jika menekan "ESC"
cam.release()
cv2.destroyAllWindows()