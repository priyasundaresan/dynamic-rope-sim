import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import numpy as np

# @ PRIYA
class Prediction:
    def __init__(self, model, num_classes, img_height, img_width, img_small_height, img_small_width, use_cuda):
        self.model = model
        self.num_classes = num_classes
        self.img_height  = img_height
        self.img_width   = img_width
        self.use_cuda = use_cuda

    def predict(self, imgs):
        # img: torch.Tensor(3, height, width)
        if len(imgs.shape) == 4:
            imgs = imgs.view(-1, imgs.shape[1], imgs.shape[2], imgs.shape[3])
        elif len(imgs.shape) == 3:
            imgs = imgs.view(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])

        result, (maps_pred, offsets_x_pred, offsets_y_pred) = self.model.forward(Variable(imgs))

        keypoints = []
        maps_array = result[0]
        offsets_x_array = result[1]
        offsets_y_array = result[2]
        if self.use_cuda:
            maps_array = maps_array.cpu().data.numpy()
            offsets_x_array = offsets_x_array.cpu().data.numpy()
            offsets_y_array = offsets_y_array.cpu().data.numpy()
        else:
            maps_array = maps_array.data.numpy()
            offsets_x_array = offsets_x_array.data.numpy()
            offsets_y_array = offsets_y_array.data.numpy()
        for i in range(self.num_classes):
            heatmap = maps_array[0][i]
            heatmap_width, heatmap_height = heatmap.shape
            CLASSIFICATION_THRESH = 0.95
            # offsets
            offsets = np.sqrt(offsets_x_array[0][i] * offsets_x_array[0][i] + offsets_y_array[0][i] * offsets_y_array[0][i])
            offsets[heatmap < CLASSIFICATION_THRESH] = float('inf')

            pred_y, pred_x = np.unravel_index(offsets.argmin(), offsets.shape)
            pred_y = int(((pred_y % heatmap_height)/heatmap_height) * self.img_width)
            pred_x = int(((pred_x % heatmap_width)/heatmap_width) * self.img_height)
            keypoints.append([pred_x, pred_y])

        maps_array = result[0]
        offsets_x_array = result[1]
        offsets_y_array = result[2]
        return (maps_array, offsets_x_array, offsets_y_array), keypoints

    def plot(self, plt_img, result, keypoints, image_id=0):
        print("Running inferences on image: %d"%image_id)

        maps_array = result[0]
        offsets_x_array = result[1]
        offsets_y_array = result[2]
        if self.use_cuda:
            maps_array = maps_array.cpu().data.numpy()
            offsets_x_array = offsets_x_array.cpu().data.numpy()
            offsets_y_array = offsets_y_array.cpu().data.numpy()
        else:
            maps_array = maps_array.data.numpy()
            offsets_x_array = offsets_x_array.data.numpy()
            offsets_y_array = offsets_y_array.data.numpy()

        all_heatmaps = []
        CLASSIFICATION_THRESH = 0.95
        for i in range(self.num_classes):
            heatmap = maps_array[0][i]
            indexes = heatmap > CLASSIFICATION_THRESH

            # offsets
            offsets = np.sqrt(offsets_x_array[0][i] * offsets_x_array[0][i] + offsets_y_array[0][i] * offsets_y_array[0][i])
            offsets_repeated = offsets.repeat(3)

            offsets_array = offsets_repeated.reshape((self.img_height, self.img_width, 3))
            offsets_array = offsets_array / offsets_array.max()

            # offsets disk
            offsets_array = np.zeros((self.img_height, self.img_width, 3))
            offsets_array[indexes] = offsets_repeated.reshape((self.img_height, self.img_width, 3))[indexes]
            offsets_array = offsets_array / offsets_array.max()

            offsets[heatmap < CLASSIFICATION_THRESH] = float('inf') # Disregard all non-classified pixels by setting their offset to inf (need to do this before taking argmin)
            pred_y, pred_x = np.unravel_index(offsets.argmin(), offsets.shape)

            offsets_array = cv2.normalize(offsets_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            result = cv2.addWeighted(offsets_array,0.8,plt_img,0.6,0)
            result = cv2.circle(result, (pred_x,pred_y), 3, (255,50,0), -1)
            all_heatmaps.append(result)

        # Assumes 4 keypoints, modify for diff # of classes
        r1,r2,r3,r4 = all_heatmaps
        res1 = cv2.hconcat([r1,r4]) # endpoints (r1 = right, r4 = left)
        res2 = cv2.hconcat([r2,r3]) # pull, hold (r2 = pull, r3 = hold)
        result = cv2.vconcat([res2,res1])
        cv2.imwrite('preds/out%04d.png'%image_id, result)
