import io
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import models, transforms
from utils import *
import pycuda.driver as cuda
import pycuda.autoinit
from __init__ import TensorrtBase
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware
from configs import *
from main import *
from utils import load_session
from PIL import Image, ImageDraw
from fastapi.responses import JSONResponse
import base64
import time

api = FastAPI()

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.05
    iou_thres = 0.3
cfg = CFG()
session = load_session(PATH_MODEL)

@api.post("/detectONNX")
async def detect_onnx(file: UploadFile):
    time_start = time.time()
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
    
    time_end = time.time()
    time_infer = time_end - time_start
    
    return JSONResponse(content={"image_detected": image_base64,
                                 "time_infer" : round(time_infer,3)})


input_names = ['images']
output_names = ['output0']
batch = 1
net = TensorrtBase("weight/best.trt",
                   input_names=input_names,
                   output_names=output_names,
                   max_batch_size=batch,
                   )
binding_shape_map = {'images': (1, 448, 448, 3)}
conf_thres = 0.05
iou_thres = 0.1

@api.post("/detectTRT")
async def detect_trt(file: UploadFile):
    time_start = time.time()
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    image_ = image.copy()
    image, ratio, (padd_left, padd_top) = resize_and_pad(image, new_shape=(448,448))
    input_image = preprocess_image_TRT(image)
    
    net.cuda_ctx.push()
    input_image = input_image.transpose((0, 3, 1, 2))
    images = np.ascontiguousarray(input_image).astype(np.float32)
    inf_in_list = [images]
    inputs, outputs, bindings, stream = net.buffers
    if binding_shape_map:
        net.context.set_optimization_profile_async(0, stream.handle)
        for binding_name, shape in binding_shape_map.items():
            net.context.set_input_shape(binding_name, shape)
        for i in range(len(inputs)):
            inputs[i].host = inf_in_list[i]
            cuda.memcpy_htod_async(inputs[i].device, inputs[i].host, stream)
        stream.synchronize()
        net.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)  

        for i in range(len(outputs)):
            cuda.memcpy_dtoh_async(outputs[i].host, outputs[i].device, stream)
        stream.synchronize()
        trt_outputs = [out.host.copy() for out in outputs]
    out = trt_outputs[0].reshape(batch,-1)
    net.cuda_ctx.pop()
    
    arr = np.array(out)
    
    new_arr = arr.reshape(1, 12348, 10)
    
    pred = postprocess(new_arr, conf_thres, iou_thres)[0]
    paddings = np.array([padd_left, padd_top, padd_left, padd_top])
    pred[:,:4] = (pred[:,:4] - paddings) / ratio
    
    for box in pred:
        xmin, ymin, xmax, ymax, conf, cls  = box
        color_hex = IDX2COLORs[int(cls)]
        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        cv2.rectangle(image_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color_bgr, 2)
        cv2.putText(image_, IDX2TAGs[int(cls)], (int(xmin),int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX , 0.4, color_bgr, 1, cv2.LINE_AA)
        
    _, image_encoded = cv2.imencode('.jpg', cv2.cvtColor(np.array(image_), cv2.COLOR_RGB2BGR))
    
    image_base64 = base64.b64encode(image_encoded).decode('utf-8')
    
    time_end = time.time()
    time_infer = time_end - time_start
    
    return JSONResponse(content={"image_detected": image_base64,
                                 "time_infer" : round(time_infer,3)})
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)