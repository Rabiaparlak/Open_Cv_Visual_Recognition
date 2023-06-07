import cv2
import numpy as np
import os

Cam = cv2.VideoCapture(0)
kernel = np.ones((15,15),np.uint8)


def ImageDif(Image1,Image2):
    Image2= cv2.resize(Image2,(Image1.shape[1],Image1.shape[0]))
    Dif_Image= cv2.absdiff(Image1,Image2)
    Dif_Number = cv2.countNonZero(Dif_Image)
    return Dif_Number

def DataUpload():
    Data_Names = []
    Data_Images = []

    Files = os.listdir("Data/")
    for File in Files:
        Data_Names.append(File.replace(".jpg",""))
        Data_Images.append(cv2.imread("Data/"+File,0))

    return Data_Names,Data_Images

def Classing(Image,Data_Names,Data_Images):
    Min_Index=0
    Min_Value = ImageDif(Image,Data_Images[0])
    for t in range(len(Data_Names)):
        Dif_Value = ImageDif(Image,Data_Images[t])
        if(Dif_Value<Min_Value):
            Min_Value=Dif_Value
            Min_Index=t
    return Data_Names[Min_Index]

Data_Names, Data_Images = DataUpload()




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
        print(Classing(Hand_Image,Data_Names,Data_Images))





    cv2.imshow("Square", Square)
    cv2.imshow("Qute_Square", Qute_Square)
    cv2.imshow("Color_Filters_Result", Color_Filters_Result)
    cv2.imshow("Result",Result)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
Cam.release()
cv2.destroyAllWindows()

