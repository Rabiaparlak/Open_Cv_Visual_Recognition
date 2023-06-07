import cv2
import numpy as np
import os

Cam = cv2.VideoCapture(0)
kernel = np.ones((15,15),np.uint8)

name = "bes"


while True:
    ret, Square = Cam.read()
    Qute_Square =  Square[0:200,0:250]
    Qute_Square_HSV = cv2.cvtColor(Qute_Square,cv2.COLOR_BGR2HSV)

    Lower_Value = np.array([0,20,40])
    Upper_Value = np.array([40,255,255])

    Color_Filters_Result = cv2.inRange(Qute_Square_HSV,Lower_Value,Upper_Value)
    Color_Filters_Result = cv2.morphologyEx(Color_Filters_Result, cv2.MORPH_CLOSE, kernel)

    Result = Qute_Square.copy()

    cnts,hierarchy = cv2.findContours(Color_Filters_Result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    Max_Width = 0
    Max_Length = 0
    Max_Index = -1

    for t in range(len(cnts)):
        cnt = cnts[t]
        x, y, w, h = cv2.boundingRect(cnt)
        if (w>Max_Width and h>Max_Length):
            Max_Width=w
            Max_Length=h
            Max_Index=t

    if(len(cnts)>0):
        x,y,w,h = cv2.boundingRect(cnts[Max_Index])
        cv2.rectangle(Result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        Hand_Image = Color_Filters_Result[y:y+h,x:x+w]
        cv2.imshow("Hand_Image",Hand_Image)





    cv2.imshow("Square", Square)
    cv2.imshow("Qute_Square", Qute_Square)
    cv2.imshow("Color_Filters_Result", Color_Filters_Result)
    cv2.imshow("Result",Result)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cv2.imwrite("Data/"+name+".jpg",Hand_Image)
Cam.release()
cv2.destroyAllWindows()

