# import os
# import torch
# from ultralytics import YOLO
# import cv2

# #MODEL PARAMETERS
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = YOLO('best.pt')
# model.to(device)
# model.conf = 0.45
# model.iou = 0.5

# cap = cv2.VideoCapture('gleb_move.mp4')

# res = model.predict(source=cap, show=True)


import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import torch
import cvzone
import m3u8
import os
import urllib.request
import time
#import deepsparse

playlist = "http://136.169.226.59/1-4/tracks-v1/mono.m3u8?token=da80b56a886144299cf4b6e2704d582d"
videoLink = os.path.dirname(playlist) + '/'
CURR_DIR = os.path.dirname(os.path.realpath(__file__))


def download_files(local_files):
    """
    super cool function that could
    sequently download ts files from m3u8 playlist
    """
    # time.sleep(0.05)
    m3u8_obj = m3u8.load(playlist)
    ts_segments_str = str(m3u8_obj.segments)
        
    for line in ts_segments_str.splitlines():
        if ".ts" in line:
            server_file_path = os.path.join(videoLink, line)
            file_name = line[line.rfind('/') + 1:line.find('?')]
            local_file_path = os.path.join(CURR_DIR, "video_files", file_name)
            if not local_file_path in local_files:
                local_files.append(local_file_path)
                urllib.request.urlretrieve(server_file_path, local_file_path)
    return local_files


model_path='./best(4).pt'
model=YOLO('best(4).pt')
# model=torch.hub.load('ultralytics/assets', 'custom', model_path, trust_repo=True, force_reload=True).eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

# model.export(format="onnx")
model.export(format='openvino')  # creates 'yolov8n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO('best(4)_openvino_model/')

#on_model = YOLO('best(4)_openvino_model/')

# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)
# cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('gleb_move.mp4')
count=0

online = False

if online:
    local_files = download_files([])
    del_file = None
    
import math  
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed

while True:
    if online:
        local_file = local_files[0]
        cap = cv2.VideoCapture(local_file)
        if del_file:
            os.remove(del_file)
            
    success, frame = cap.read()
    with torch.no_grad():
        results = ov_model(frame, verbose=False)
    speed_t = 0
    for r in results:
        boxes = r.boxes
        # k = 0
        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = round(float(box.conf[0]), 2)
            #Class name
            cls = int(box.cls[0])
            # currentClass = classNames[cls]
        


            if conf > 0.35:
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9)
                cvzone.putTextRect(frame, f'{conf}', (max(0, x1), max(35, y1)),
                            scale=0.9, thickness=1, offset=3)
                cvzone.putTextRect(frame, f'{cls}', (max(0, x1 + 30), max(35, y1)),
                            scale=0.9, thickness=1, offset=3)
                # cvzone.putTextRect(frame, f'{speed_t}', (max(0, x1 + 40), max(35, y1)),
                #             scale=0.9, thickness=1, offset=3)
                




        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

    if online:
        del_file = local_file
        local_files.pop(0)
        local_files = download_files(local_files)




