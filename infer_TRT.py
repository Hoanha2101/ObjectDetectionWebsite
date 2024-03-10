import numpy as np
import cv2
from utils import *
import pycuda.driver as cuda
import pycuda.autoinit
import time
from __init__ import TensorrtBase
from tensorflow.keras.preprocessing import image as I_convert
from preprocess import *
from postprocess import *
from main import *

path = "sample/1.jpg"

def preprocess_image_TRT(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = I_convert.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    return img_tensor

image = cv2.imread(path)
img_ = image.copy()
image, ratio, (padd_left, padd_top) = resize_and_pad(image, new_shape=(448,448))
input_image = preprocess_image_TRT(image)

input_names = ['images']
output_names = ['output0']
batch = 1
net = TensorrtBase("weight/best.trt",
                   input_names=input_names,
                   output_names=output_names,
                   max_batch_size=batch,
                   )
binding_shape_map = {'images': (1, 448, 448, 3)}
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
print(new_arr)
conf_thres = 0.05

iou_thres = 0.1

pred = postprocess(new_arr, conf_thres, iou_thres)[0]
paddings = np.array([padd_left, padd_top, padd_left, padd_top])
pred[:,:4] = (pred[:,:4] - paddings) / ratio

for box in pred:
    xmin, ymin, xmax, ymax, conf, cls  = box
    color_hex = IDX2COLORs[int(cls)]
    color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
    cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color_bgr, 2)
    cv2.putText(img_, IDX2TAGs[int(cls)], (int(xmin),int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX , 0.5, color_bgr, 1, cv2.LINE_AA)
cv2.imshow("Image", img_)
cv2.waitKey(0)
cv2.destroyAllWindows()
