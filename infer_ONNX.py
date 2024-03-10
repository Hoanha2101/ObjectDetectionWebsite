import onnxruntime
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np

def load_image_cls_bottle(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (448, 448))
    img_tensor = image.img_to_array(img)
    img_tensor = np.transpose(img_tensor, (2, 0, 1))  # Transpose dimensions from NHWC to NCHW
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    return img_tensor

# Load the ONNX model
ort_session = onnxruntime.InferenceSession("weight/best.onnx")

# Path to the input image
path_img = "sample/1.jpg"

# Load and preprocess the image
img = load_image_cls_bottle(path_img)

# Run inference
outputs = ort_session.run(["output0"], {'images': img})

print(outputs[0][0][0])
