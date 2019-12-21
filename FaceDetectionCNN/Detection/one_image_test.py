
"""
单个图片检测任务
"""
import sys
from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from mtcnn_model import P_Net, R_Net, O_Net
from loader import TestLoader
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def gen_mtcnn_model():
    test_mode = "ONet"
    thresh = [0.9, 0.6, 0.7]
    min_face_size = 24
    stride = 2
    slide_window = False
    shuffle = False
    detectors = [None, None, None]
    prefix = ['E:\\AIcode\\face\\model\\MTCNN_model\\PNet_landmark\\PNet', 
                'E:\\AIcode\\face\\model\\MTCNN_model\\RNet_landmark\\RNet', 
                'E:\\AIcode\\face\\model\\MTCNN_model\\ONet_landmark\\ONet']
    epoch = [18, 14, 16]
    batch_size = [2048, 256, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                stride=stride, threshold=thresh, slide_window=slide_window)
    return mtcnn_detector

# all_boxes,landmarks = gen_mtcnn_model().detect_face(test_data)
# image = cv2.imread(imagepath)
# for bbox,landmark in zip(all_boxes[0],landmarks[0]):       
#     cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
#     cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255), 7)
# cv2.imshow("lala",image)
# cv2.waitKey(0)  

# gt_imdb = []

# path = "data"
# for item in os.listdir(path):
#     if('jpg' not in item):
#         continue
#     gt_imdb.append(os.path.join(path,item))

# print(gt_imdb)
# test_data = TestLoader(gt_imdb)  #test_data 是
# print("testdata",test_data)#,len(test_data))
# all_boxes,landmarks = gen_mtcnn_model().detect_face(test_data)
# # all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
# count = 0
# for imagepath in gt_imdb:
#     image = cv2.imread(imagepath)

#     for bbox,landmark in zip(all_boxes[count],landmarks[count]):
        
#         cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
#         cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255), 7)
        
#     for landmark in landmarks[count]:
#         for i in range(int(len(landmark)/2)):
#             cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
        
#     count = count + 1
#     #cv2.imwrite("result_landmark/%d.png" %(count),image)

#     cv2.imshow("lala",image)
#     cv2.waitKey(0)  



def CatchUsbVideo(window_name):
    mtcnn_model = gen_mtcnn_model()
    cv2.namedWindow(window_name)
    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(0)
    color = (0, 255, 0)
    i =0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break
        # print ("ddd",frame.shape)
        image_dict=[]
        # image = cv2.imread(frame)
        image_dict.append(frame)
        all_boxes,landmarks = mtcnn_model.detect_face(image_dict)
        for bbox,landmark in zip(all_boxes[0],landmarks[0]):
            cv2.putText(frame,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255), 7)
            #模型分类部分，输入为人脸框位置 
            x_l = int(bbox[0])
            y_l = int(bbox[1])
            x_w = int(bbox[2])
            y_h = int(bbox[3])
            f = cv2.resize(frame[x_l:x_w,y_l:y_h],(48,48))
            cv2.imwrite("E:\\AIcode\\face\\FaceDetectionCNN\\data_test\\{}.jpg".format(i),f)
        cv2.imshow("lala",frame)
        # cv2.waitKey(10)  
        i+=1
        if cv2.waitKey(10) & 0xFF == ord('q'):# 如果强制停止执行程序，结束视频放映
            break

if __name__ == '__main__':
    CatchUsbVideo("detect_image")
  
