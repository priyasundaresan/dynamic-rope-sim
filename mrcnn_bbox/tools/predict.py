from os import listdir
from xml.etree import ElementTree
import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, mold_image
import cv2
import os

# define the prediction configuration
class PredictionConfig(Config):
    NAME = "knot_cfg"
    # number of classes (background + knot)
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class BBoxFinder:
    def __init__(self, model, cfg, prediction_thresh=0.96):
        self.model = model
        self.cfg = cfg
        self.prediction_thresh = prediction_thresh

    def predict(self, image, thresh=0.96, plot=True):
        scaled_image = mold_image(image, self.cfg)
        sample = expand_dims(scaled_image, 0)
        pred = self.model.detect(sample, verbose=0)
        yhat = pred[0]
        boxes = []
        vis = image.copy()
        for box, score in zip(yhat['rois'], yhat['scores']):
            y1, x1, y2, x2 = box
            boxes.append(((x1,y1,x2,y2), score))
            vis = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
            break # HACK!!
            #if score > thresh:
            #    y1, x1, y2, x2 = box
            #    boxes.append(((x1,y1,x2,y2), score))
            #    vis = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
        if plot:
            cv2.imshow("predicted", vis)
            cv2.waitKey(0)
        boxes.sort(key = lambda x: x[1])
        return boxes
