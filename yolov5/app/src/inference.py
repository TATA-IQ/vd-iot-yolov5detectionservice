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

from console_logging.console import Console
console=Console()

class InferenceModel:
    """
    Yolov5 inference
    """
    def __init__(self, model_path=None, gpu=False,logger=None):
        """
        Initialize Yolov5 inference
        
        Args:
            model_path (str): path of the downloaded and unzipped model
            gpu=True, if the system have NVIDIA GPU compatibility
        """
        if gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.model_path = model_path
        self.model = None
        self.augment = None
        self.object_confidence = 0.01
        self.iou_threshold = 0.01
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.half = False
        self.log = logger

    def initializeVariable(
        self,
        conf=0.01,
        iou_threshold=0.01,
        classes=None,
        agnostic_nms=False,
        max_det=1000,
        half=False,
        augment=False,
    ):
        """
        This will initialize the model parameters of inference. This configuration is specific to camera group
        Args:
            conf (float): confidence of detection
            iou_threshold (float): intersection over union threshold of detection
            classes (obj): classes of the detection
            agnostic_ms (boolean): Non max supression
            half (boolean): precision of the detection
            augment (boolean): Augmentation while detection
        """
        self.object_confidence = conf
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.half = half

    def loadmodel(self):
        '''
        This will load Yolov5 model
        '''
        if os.path.exists(self.model_path):
            # self.model_path = "/home/sridhar.bondla10/gitdev_v3/vd-iot-yolov5detectionservice/yolov5/app/DeployedModel/tata_com_ppe_02_03_2022.pt"
            self.model = attempt_load(self.model_path, device=self.device)
        else:
            self.log.error("MODEL NOT FOUND")
            console.error("MODEL NOT FOUND")
            sys.exit()
        self.stride = int(self.model.stride.max())
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )

    def getClasses(self):
        """
        Get the classes of the model

        """
        return self.model.names

    def infer(self, image, model_config=None):
        """
        This will do the detection on the image
        Args:
            image (array): image in numpy array
            model_config (dict): configuration specific to camera group for detection
        Returns:
            list: list of dictionary. It will have all the detection result.
        """
        image_height, image_width, _ = image.shape
        raw_image = copy.deepcopy(image)
        img0 = copy.deepcopy(image)
        img = letterbox(img0, 640, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        # print("img===>", img)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # print("model====>", self.model)
        predictions = self.model(img, augment=self.augment)[0]
        print("predictions=====>", predictions)
        print("*"*100)
        print(model_config)
        if len(model_config)>0:
            print("model config is not none")
            self.object_confidence = model_config["conf_thres"]
            self.iou_threshold = model_config["iou_thres"]
            self.max_det = model_config["max_det"]
            self.agnostic_nms = model_config["agnostic_nms"]
            self.augment = model_config["augment"]
            # self.classes=model_config["conf_thres"]
        predictions = non_max_suppression(
            predictions,
            self.object_confidence,
            self.iou_threshold,
            self.classes,
            self.agnostic_nms,
            max_det=self.max_det,
        )
        # print("predictions=====>", predictions)
        listresult = []
        for i, det in enumerate(predictions):
            non_scaled_coords = det[:, :6].clone()
            print("non_scaled_coords===",non_scaled_coords)
            # print("det[:, :6]====>",det[:, :6])
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            # print("$"*100)
            print("======det===>",det)
            # print(type(det))
            for j,(x1, y1, x2, y2, conf, cls) in enumerate(reversed(det)):
                listresult.append(
                    {
                        "class": int(cls.numpy()),
                        "id": None,
                        "class_name": str(self.names[int(cls.numpy())]),
                        "score": round(float(conf.numpy()), 3),
                        "xmin": int(x1),
                        "ymin": int(y1),
                        "xmax": int(x2),
                        "ymax": int(y2),
                        "xmin_c": int(non_scaled_coords[j][0].item()),
                        "ymin_c": int(non_scaled_coords[j][1].item()),
                        "xmax_c": int(non_scaled_coords[j][2].item()),
                        "ymax_c": int(non_scaled_coords[j][3].item()),
                        
                    }
                )
        return listresult
    
    def deduplication(self,det_list, h,w):
        H = h
        W = w
        final_det = []
        clist_i = []
        for i in range(0, len(det_list)):
            for j in range(i+1, len(det_list)):
                if(i not in clist_i and (det_list[i][0]['class'] == det_list[j][0]['class']) and 
                   (((abs(det_list[i][0]['xmin'] - det_list[j][0]['xmax']) <= 5 or abs(det_list[i][0]['xmax'] - det_list[j][0]['xmin']) <= 5) and 
                   ((det_list[i][0]['ymax']-det_list[j][0]['ymin'])/(det_list[j][0]['ymax']-det_list[j][0]['ymin']) > 0.5 and
                    (det_list[j][0]['ymax']-det_list[i][0]['ymin'])/(det_list[i][0]['ymax']-det_list[i][0]['ymin']) > 0.5)) or 
                   (((abs(det_list[i][0]['ymin'] - det_list[j][0]['ymax']) <= 5 or abs(det_list[i][0]['ymax'] - det_list[j][0]['ymin']) <= 5)) and 
                   ((det_list[i][0]['xmax']-det_list[j][0]['xmin'])/(det_list[j][0]['xmax']-det_list[j][0]['xmin']) > 0.5 and
                    (det_list[j][0]['xmax']-det_list[i][0]['xmin'])/(det_list[i][0]['xmax']-det_list[i][0]['xmin']) > 0.5)))):
                    #print(str(i)+"_"+str(j))
                    #print(det_list[i][0])
                    #print(det_list[j][0])
                    cord =   [{'class': det_list[i][0]['class_id'],
                                "id":None,
                              'class_name': det_list[i][0]['class'],
                              'score': det_list[i][0]['score'],
                              'xmin': min(det_list[i][0]['xmin'], det_list[j][0]['xmin']),
                              'ymin': min(det_list[i][0]['ymin'], det_list[j][0]['ymin']),
                              'xmax': max(det_list[i][0]['xmax'], det_list[j][0]['xmax']),
                              'ymax': max(det_list[i][0]['ymax'], det_list[j][0]['ymax']),
                              'xmin_c': min(det_list[i][0]['xmin'], det_list[j][0]['xmin']/W),
                              'ymin_c': min(det_list[i][0]['ymin'], det_list[j][0]['ymin']/H),
                              'xmax_c': max(det_list[i][0]['xmax'], det_list[j][0]['xmax'])/W,
                              'ymax_c': max(det_list[i][0]['ymax'], det_list[j][0]['ymin'])/H}]
                    #print(cord)
                    final_det.append(cord)
                    clist_i.append(i)
                    clist_i.append(j)
        for i in range(0, len(det_list)):
            if(i not in clist_i):
                final_det.append(det_list[i])
        return final_det
    def split(self,frame,split_col,split_row,model):
        swidth_col =  int(frame.shape[1]/split_col)
        sheight_row =  int(frame.shape[0]/split_row)
        det_list = []
        h,w,_=frame.shape
        self.log.info(f"===split_col,split_row=={split_col},{split_row}")
        console.info(f"===split_col,split_row=={split_col},{split_row}")
        for i in range(0, split_row):
            for j in range(0, split_col):
                self.log.info(f"i:{i}, j:{j}")
                console.info(f"i:{i}, j:{j}")
                sub_img = frame[i*sheight_row:(i+1)*sheight_row, j*swidth_col:(j+1)*swidth_col]                
                # res=model.predict(sub_img)                
                # cv2.imwrite("/home/sridhar.bondla10/gitdev_v3/vd-iot-yolov5detectionservice/yolov5/test/"+str(i)+"_"+str(j)+".jpg",sub_img)

                image=copy.deepcopy(sub_img)
                image_height, image_width, _ = image.shape
                raw_image = copy.deepcopy(image)
                img0 = copy.deepcopy(image)
                img = letterbox(img0, 640, stride=32)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                # print("img===>", img)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # print("model====>", self.model)
                predictions = self.model(img, augment=self.augment)[0]
                self.log.info(f"predictions=====>{predictions}")
                console.info(f"predictions=====>{predictions}")
                # print(self.model_config)
                if len(self.model_config)>0:
                    self.log.info("model config is not none")
                    console.info("model config is not none")
                    self.object_confidence = self.model_config["conf_thres"]
                    self.iou_threshold = self.model_config["iou_thres"]
                    self.max_det = self.model_config["max_det"]
                    self.agnostic_nms = self.model_config["agnostic_nms"]
                    self.augment = self.model_config["augment"]
                    # self.classes=model_config["conf_thres"]
                predictions = non_max_suppression(
                    predictions,
                    self.object_confidence,
                    self.iou_threshold,
                    self.classes,
                    self.agnostic_nms,
                    max_det=self.max_det,
                )
                # print("predictions=====>", predictions)
                # listres=self.mark_res(res, i*sheight_row, j*swidth_col, frame.shape[0], frame.shape[1])
                origin_y,origin_x,H,W=i*sheight_row, j*swidth_col, frame.shape[0], frame.shape[1]
                listresult = []
                for i1, det in enumerate(predictions):
                    non_scaled_coords = det[:, :6].clone()
                    # print("non_scaled_coords===",non_scaled_coords)
                    # print("det[:, :6]====>",det[:, :6])
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                    # print("$"*100)
                    self.log.info("======det===>{det}")
                    console.info("======det===>{det}")
                    # print(type(det))
                    for j1, (x1, y1, x2, y2, conf, cls) in enumerate(reversed(det)):
                        
                        listresult.append(
                            {
                                "class": int(cls.numpy()),
                                "id": None,
                                "class_name": str(self.names[int(cls.numpy())]),
                                "score": round(float(conf.numpy()), 3),
                                "xmin": int(x1) + origin_x,
                                "ymin": int(y1) + origin_y,
                                "xmax": int(x2) + origin_x,
                                "ymax": int(y2) + origin_y,
                                "xmin_c": round((int(x1) + origin_x)/W,5),
                                "ymin_c": round((int(y1) + origin_y)/H,5),
                                "xmax_c": round((int(x2) + origin_x)/W,5),
                                "ymax_c": round((int(y2) + origin_y)/H,5),  
                                # "xmin_c": int(non_scaled_coords[j][0].item()),
                                # "ymin_c": int(non_scaled_coords[j][1].item()),
                                # "xmax_c": int(non_scaled_coords[j][2].item()),
                                # "ymax_c": int(non_scaled_coords[j][3].item()),
                                
                            }
                        )
                # return listresult
                
                
                
                # listres=self.mark_res(res, i*sheight_row, j*swidth_col, frame.shape[0], frame.shape[1])
                if len(listresult)>0:
                    det_list.append(listresult)
        return sum(self.deduplication(det_list,h,w),[])
    
    

    def infer_v2(self, image, model_config=None, split_columns=1, split_rows=1):
        """
        This will do the detection on the image
        Args:
            image (array): image in numpy array
            model_config (dict): configuration specific to camera group for detection
        Returns:
        results (list):  list of dictionary. It will have all the detection result.
        """
        # image_height, image_width, _ = image.shape
        print("image shape====",image.shape)
        self.log.info(f"image shape===={image.shape}")
        console.info(f"image shape===={image.shape}")
        # raw_image = copy.deepcopy(image)
        # img0 = copy.deepcopy(image)
        img = copy.deepcopy(image)
        # print("model config====>", model_config)
        
        self.model_config = model_config
        # if model_config is not None:
        if len(model_config) !=0:
            self.isTrack = model_config['is_track']
            self.object_confidence = model_config["conf_thres"]
            self.iou_threshold = model_config["iou_thres"]
            self.max_det = model_config["max_det"]
            self.agnostic_nms = model_config["agnostic_nms"]
            self.augment = model_config["augment"]
            # self.classes=model_config["classes"]
        
        
        listresult = self.split(img, split_columns, split_rows, self.model)
        if len(listresult)==0:
            self.log.info("no detections")
            console.info("no detections")
        else:
            self.log.info(f"listresult==={listresult}")
            console.info(f"listresult==={listresult}")
        return listresult
        