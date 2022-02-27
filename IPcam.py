#-------------------------------------#
#       调用IPcam检测
#-------------------------------------#
from keras.layers import Input
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import cul
yolo = YOLO()
product = []
# 調用IPcam
capture=cv2.VideoCapture('http://192.168.2.81:8088/video') # capture=cv2.VideoCapture("1.mp4")
#cv2.VideoCapture('http://192.168.2.81:8088/video')

fps = 0.0
while(True):
    t1 = time.time()
    # 讀取某一偵
    ref,frame=capture.read()
    # 格式轉變，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 轉變成Image
    frame = Image.fromarray(np.uint8(frame))

    # 進行檢測
    frame,product = yolo.detect_image(frame)

    # RGBtoBGR满足opencv顯示格式
    frame = cv2.cvtColor(np.array(frame),cv2.COLOR_RGB2BGR)
    
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    total= cul.cul(product) 
    for i in range(0, len(total)):
        print (total[i]+"\n")
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("video",frame)
    c= cv2.waitKey(30) & 0xff 
    if c==ord("q"):
        capture.release()
        break

yolo.close_session()    
