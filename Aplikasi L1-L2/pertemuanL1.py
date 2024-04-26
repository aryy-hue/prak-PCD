import cv2
import numpy as np

cam = cv2.VideoCapture(0)
while True:
    _,frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([66,98,100]) #HUE,SATURATION,VALUE
    upper_color = np.array([156,232,255]) #HUE,SATURATION,VALUE
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
    