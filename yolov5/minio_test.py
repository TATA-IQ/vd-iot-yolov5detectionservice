from minio import Minio
from minio.error import S3Error
import urllib3
import cv2
from zipfile import ZipFile
import os
import numpy as np
from datetime import datetime
import shutil


class DownloadModel:
    """
    Class to download model from miniodb and remove it after validation is done.
    """

    def __init__(self, bucket_name):
        """
        args: bucket_name-> Bucket name of miniodb
        """
        self.bucket_name = bucket_name
        self.client = Minio(
            endpoint="172.16.0.170:9000",
            access_key="XHXPHOuchiuxkQpYUYIM",
            secret_key="JEJG4oNT3rpTe80VlFXWcxkE8fWUVJfxP1XHdmNd",
            secure=False,)
        self.client = Minio(
            endpoint="172.16.0.178:9000",
            access_key="C3dkhEzuNU0JPt3UEwSN",
            secret_key="s8YRgOAnwmT3X3OWBSxcJYIViSDE65uIedB4fLsB",
            secure=False,
        )
    def get_buckets(self,):
        print(self.client)
        return self.client.list_buckets()

    def save_data(self, object_name, local_path):
        """
        args:object_name-> full path of file from miniodb
        args: local_path-> path to save model locally
        """
        obj_name = object_name.split("/")[-1]
        print("obj_name", obj_name)
        save_path = os.path.join(local_path, obj_name)
        try:
            self.client.fget_object(self.bucket_name, object_name, save_path)
            print(f"{object_name} is saved into {save_path}")
        except S3Error as e:
            buckets = self.client.list_buckets()
            print("available buckets ",buckets)
            print(e)
            print(f"{object_name} {e.message} ")

    def save_model_files(self, object_name, local_path):
        """
        args:object_name-> full path of file from miniodb
        args:local_path-> path to save model  locally
        """
        obj_name = object_name.split("/")[-1]
        # print(obj_name)
        save_path = os.path.join(local_path, obj_name)
        try:
            self.client.fget_object(self.bucket_name, object_name, save_path)
            print(f"{object_name} is saved into {save_path}")
        except S3Error as e:
            print(e)
            print(f"{object_name} {e.message} ")

    def unzip(self, path, unzippath, modelname):
        """
        args:path-> path of the downloaded model in zip file
        unzippath:
        modelname:
        """
        print("Zip path===>", path)
        with ZipFile(path, "r") as zObject:
            zObject.extractall(path=unzippath)
        os.remove(path)

    def removeData(self, path):
        shutil.rmtree(path)


# bucket_name = "yolov5"
# # object_name = "minio_images.zip"
# # object_name = "test(2).mp4"
# # object_name = "cam4/cam4_2023_06_13_18_17_01_739876.jpg"
object_name = "/object_detection/usecase1/model_id_2/ppe.zip"  #object_path
local_path = "/home/sridhar.bondla/api_database/app/minIO_db/minio_data1/"

a = DownloadModel("yolov5")
print(a)
print(a.get_buckets())
# a.save_data(object_name,local_path)
# a.save_model_files(object_name,local_path)
