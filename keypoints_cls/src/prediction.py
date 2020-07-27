import torch
from torch.autograd import Variable
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
    
    def plot(self, img, heatmap, image_id=0, cls=None, classes=None):
        print("Running inferences on image: %d"%image_id)
        all_overlays = []
        for i in range(self.num_keypoints):
            h = heatmap[0][i]
            pred_y, pred_x = np.unravel_index(h.argmax(), h.shape)
            vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis = np.repeat(vis[:,:,np.newaxis], 3, axis=2)
            #grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            overlay = cv2.addWeighted(img, 0.3, vis, 0.7, 0)
            overlay = cv2.circle(overlay, (pred_x,pred_y), 3, (255,50,0), -1)
            all_overlays.append(overlay)
        result1 = cv2.vconcat(all_overlays[:self.num_keypoints//2])
        result2 = cv2.vconcat(all_overlays[self.num_keypoints//2:])
        result = cv2.hconcat((result1, result2))
        if cls is not None:
            label = classes[cls]
            cv2.putText(result, label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite('preds/out%04d.png'%image_id, result)
