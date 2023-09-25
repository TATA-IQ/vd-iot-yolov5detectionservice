import cv2
from src.inference import InferenceModel
model_path="DeployedModel/tata_com_ppe_02_03_2022.pt"
image_path=""
ifm=InferenceModel(model_path)
ifm.loadmodel()
img=cv2.imread("test.jpg")
res=ifm.infer(img)
classnm=ifm.getClasses()
print("=====")
print(classnm)
#print(res)