from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
class Image_Model(BaseModel):
    image_name: Union[str, None] = None
    image: Union[str, None] = None
    model_config:Union[dict[str,str], None] = None
