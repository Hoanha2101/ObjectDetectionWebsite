import io
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import models, transforms
from utils import *
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware
from configs import *
from main import *
from utils import load_session
from PIL import Image, ImageDraw
from fastapi.responses import JSONResponse
import base64

api = FastAPI()

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.05
    iou_thres = 0.3
cfg = CFG()

session = load_session(PATH_MODEL)

@api.post("/detect")
async def detect(file: UploadFile):
    image_bytes = await file.read()
    
    image = Image.open(io.BytesIO(image_bytes))
    
    image_array = np.array(image)
    
    pred = prediction(
            session=session,
            image=image_array,
            cfg=cfg )
    
    image = Image.fromarray(image_array)
    
    image_detected = visualize(image, pred)
    
    _, image_encoded = cv2.imencode('.jpg', cv2.cvtColor(np.array(image_detected), cv2.COLOR_RGB2BGR))
    
    image_base64 = base64.b64encode(image_encoded).decode('utf-8')
    
    return JSONResponse(content={"image_detected": image_base64})

