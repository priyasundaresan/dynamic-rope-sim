import torch
from torch.autograd import Variable
import torch.nn.functional as f
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import numpy as np

# @ PRIYA
class Prediction:
    def __init__(self, model, num_keypoints, img_height, img_width, use_cuda):
        self.model = model
        self.num_keypoints = num_keypoints
        self.img_height  = img_height
        self.img_width   = img_width
        self.use_cuda = use_cuda
        
    def predict(self, imgs):
        # img: torch.Tensor(3, height, width) 
        if len(imgs.shape) == 4:
            imgs = imgs.view(-1, imgs.shape[1], imgs.shape[2], imgs.shape[3])    
        elif len(imgs.shape) == 3:
            imgs = imgs.view(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
            
        heatmap = self.model.forward(Variable(imgs))
        return heatmap

    def best_match(self, distribution):
        distribution = distribution.T
        width, height = distribution.shape
        flattened_dist = distribution.ravel()
        best_match = np.argmax(flattened_dist)
        u = best_match % width
        v = best_match % height
        best_match_px = (u,v)
        return best_match_px
    
    def plot(self, img, heatmap, cls, cls_to_label, image_id=0):
        print("Running inferences on image: %d"%image_id)
        label = cls_to_label[cls]
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heatmap_vis = []
        for h in heatmap[0]:
            #best_match_px = self.best_match(h)
            y,x = np.where((h == np.amax(h)))
            best_match_px = (x,y)
            h = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            r = cv2.addWeighted(grayscale, 0.5, h, 0.5, 0)
            try:
                r = cv2.circle(r, best_match_px, 3, (0,0,0), -1)
            except:
                pass
            heatmap_vis.append(r)
        res1 = cv2.hconcat([heatmap_vis[0],heatmap_vis[-1]]) # endpoints (r1 = right, r4 = left)
        res2 = cv2.hconcat([heatmap_vis[1],heatmap_vis[2]])
        result = cv2.vconcat([res2,res1])
        cv2.putText(result, label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite('preds/out%04d.png'%image_id, result)
