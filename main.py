from utils import *
from preprocess import *
from postprocess import *
from PIL import Image, ImageDraw
from configs import *

def prediction(session, image, cfg):
    image, ratio, (padd_left, padd_top) = resize_and_pad(image, new_shape=cfg.image_size)
    img_norm = normalization_input(image)
    pred = infer(session, img_norm)
    pred = postprocess(pred, cfg.conf_thres, cfg.iou_thres)[0]
    paddings = np.array([padd_left, padd_top, padd_left, padd_top])
    pred[:,:4] = (pred[:,:4] - paddings) / ratio
    return pred

def visualize(image, pred):
    img_ = image.copy()
    drawer = ImageDraw.Draw(img_)
    for p in pred:
        x1,y1,x2,y2,_, id = p
        id = int(id)
        drawer.rectangle((x1,y1,x2,y2),outline=IDX2COLORs[id],width=3)  
        drawer.text((x1, y1 - 12), str(IDX2TAGs[id]), fill=IDX2COLORs[id])
    return img_