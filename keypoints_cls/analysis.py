import pickle
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
#from src.model_kpts import KeypointsGauss
from src.model import KeypointsGauss
#from src.dataset_kpts import KeypointsDataset, transform
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np

# model
keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
keypoints.load_state_dict(torch.load('checkpoints/undo_reid_term/model_2_1_199.pth'))

# cuda
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    torch.cuda.set_device(0)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

image_dir = 'data/undo_reid_term/test/images'

classes = {0: "Undo", 1:"Reidemeister", 2:"Terminate"}
for i, f in enumerate(sorted(os.listdir(image_dir))):
    #img = Image.open(os.path.join(image_dir, f)).convert('RGB')
    img = Image.open(os.path.join(image_dir, f))
    img = np.array(img)
    img_t = transform(img)
    img_t = img_t.cuda()
    heatmap, cls = prediction.predict(img_t)
    cls = torch.argmax(cls).item()
    #heatmap = prediction.predict(img_t)
    #cls = 0
    heatmap = heatmap.detach().cpu().numpy()
    prediction.plot(img, heatmap, cls, classes, image_id=i)
