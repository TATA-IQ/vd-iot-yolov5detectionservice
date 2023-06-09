import copy
import glob
import json
import os
import sys

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.centroid_tacker import CentroidTracker

import torch
import cv2
import numpy as np
import pandas as pd
class InferenceModel():
    def __init__(self, model_path=None,gpu=False):
        if gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
           self.device="cpu" 
        self.model_path=model_path
        self.model=None
        self.augment = None
        self.object_confidence = 0.01
        self.iou_threshold = 0.01
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.half = False
    def initializeVariable(self, conf=0.01, iou_threshold=0.01, classes=None,agnostic_nms=False,max_det=1000, half=False, augment=False):
        
        self.object_confidence = conf
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.half = half
        

    def loadmodel(self):
        if os.path.exists(self.model_path):
            self.model = attempt_load(self.model_path, device=self.device)
        else:
            print("MODEL NOT FOUND")
            sys.exit()
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
    def getClasses(self):
        return self.model.names
    
    def infer(self,image,model_config=None):
        image_height, image_width, _ = image.shape
        raw_image = copy.deepcopy(image)
        img0 = copy.deepcopy(image)
        img = letterbox(img0, 640, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        print("img===>",img)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print("model====>",self.model)
        predictions = self.model(img, augment=self.augment)[0]
        print("predictions=====>",predictions)
        if model_config is not None:
            self.object_confidence=model_config["conf_thres"]
            self.iou_threshold=model_config["iou_thres"]
            self.max_det=model_config["max_det"]
            self.agnostic_nms=model_config["agnostic_nms"]
            self.augment=model_config["augment"]
            self.object_confidence=model_config["conf_thres"]
            #self.classes=model_config["conf_thres"]
        predictions = non_max_suppression(predictions, self.object_confidence, self.iou_threshold, self.classes,
                                          self.agnostic_nms, max_det=self.max_det)
        print("predictions=====>",predictions)
        listresult=[]
        for i, det in enumerate(predictions):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for x1, y1, x2, y2, conf, cls in reversed(det):
                    listresult.append({
                                     "class_name": str(self.names[int(cls.numpy())]),
                                     "score": round(float(conf.numpy()), 3),
                                     "xmin": int(x1), "ymin": int(y1),
                                     "xmax": int(x2), "ymax": int(y2)})
        return listresult
        
        
