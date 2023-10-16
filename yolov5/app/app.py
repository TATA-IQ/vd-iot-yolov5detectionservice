""" yolov5"""
import subprocess
import time
import os
import socket
import base64
import numpy as np
from zipfile import ZipFile

import cv2
import requests
from fastapi import FastAPI
import uvicorn

from src.inference import InferenceModel
from sftpdownload.download import SFTPClient
from config_parser.parser import ParseConfig
from querymodel.imageModel import Image_Model
app = FastAPI()
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('192.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP
conf=ParseConfig()
apis=conf.getAPIs()
sftp=conf.getSFTPConfig()
#subprocess.getoutput("docker ps -aqf name=containername")
hostid=subprocess.getoutput("cat /etc/hostname")
cgroupout=subprocess.getoutput("cat /proc/self/cgroup")
print(cgroupout)
print("===>",cgroupout.split(":")[2][8:20])
hostid=cgroupout.split(":")[2][8:20]
#hostid="ce09255b0b13"
query={"container_id":hostid}
print("query====>",query)
responseContainer=requests.get(apis["container"],json=query)
print(responseContainer.json())
while True:    
    responseContainer=requests.get(apis["container"],json=query)
    print(query,responseContainer.json())
    try:
        model_id=responseContainer.json()["data"][0]["model_id"]
        query={"model_id":model_id}        
        break
    except:
        time.sleep(5)
        continue
responseModel=requests.get(apis["container_model"],json=query)
print("found====>",responseModel.json())
model_path=responseModel.json()["data"][0]["model_path"]
time.sleep(10)
print("downloading from server")
sf=SFTPClient(sftp["host"], sftp["port"], sftp["username"], sftp["password"])

sf.downloadyolov5(model_path,"model/test.zip")
print("downloaded from server")

model_nm=model_path.split(".")[0]
with ZipFile("model/test.zip", 'r') as zObject:
     zObject.extractall(path="model/temp")


# class ImageModel():
#     image: str


#model_path=""

print("====Model Loaded====")
root_path="/model"

ip=get_local_ip()
url="http://"+ip+":"+str(responseModel.json()["data"][0]["model_port"])+"/detect"
print("URL=====>",url)
model_id=responseModel.json()["data"][0]["model_id"]
print("updating for model id===>",model_id)

responseupdate=requests.post(apis["update_endpoint"],json={"model_end_point":url,"model_id":model_id})
print(responseupdate.json())
model_port=responseModel.json()["data"][0]["model_port"]

modelmaster=requests.get(apis["model_master_config"],json={"model_id":model_id}).json()
gpu=False
print("====Model Master Response======")
print(modelmaster)
model_config=requests.get(apis["model_config"],json={"model_id":model_id})
if modelmaster["data"][0]["device"]=="gpu":
    gpu=True
print("*******",os.listdir("model/temp/"))
model_list=os.listdir("model/temp/")
im=InferenceModel(model_path="model/temp/"+model_list[0], gpu=gpu)
im.loadmodel()

print("Running api on GPU {}".format(gpu))

def strToImage(imagestr):
    imagencode=imagestr.encode()
    decodedimage=base64.b64decode(imagencode)
    nparr=np.frombuffer(decodedimage,np.uint8)
    img_np=cv2.imdecode(nparr,flags=1)
    return img_np


@app.get("/test")
def test_fetch():
    return {"data":responseModel.json()}
@app.post("/detect")
def detection(data:Image_Model):
    image=strToImage(data.image)
    print(image)
    
    if data.model_config is None:
        res=im.infer(image)
    else:
        res=im.infer(image,data.model_config)
    print("======inference done**********")
    print(res)
    print(type(res))
    return {"data":res}
@app.get("/classes")
def get_classes():
    res=im.getClasses()
    return {"data":res}
        

if __name__ == "__main__":
    print("=====inside main************")
    uvicorn.run(app, host="0.0.0.0", port=model_port)


# app = FastAPI()
# @app.get("/test")
# async def ClassMaster_fetch(ImageModel):
#     return {"data":"Hi From TesorFlow "+responseModel.json()["data"][0]["model_usecase"]}

#         ##im.infer(ImageModel.image)



