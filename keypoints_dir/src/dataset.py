import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pickle
import os
from datetime import datetime

transform = transforms.Compose([transforms.ToTensor()])

class KeypointsDataset(Dataset):
    def __init__(self, img_folder, labels_folder, num_classes, img_height, img_width, radius, transform):
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.radius = radius         
        self.transform = transform

        self.imgs = []
        self.labels = []
        for i in range(len(os.listdir(labels_folder))):
            label = np.load(os.path.join(labels_folder, '%05d.npy'%i))
            label_formatted = []
            for u,v in label:
                label_formatted.append(np.clip(u, 0, self.img_width-1))
                label_formatted.append(np.clip(v, 0, self.img_height-1))
                visible=1
                label_formatted.append(visible)
            self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
            #self.imgs.append(os.path.join(img_folder, '%05d.png'%i))
            self.labels.append(label_formatted)

        self.map_value = np.array([[np.linalg.norm([self.img_width - _x, self.img_height - _y]) 
                          for _x in range(img_width * 2)] for _y in range(img_height * 2)])
        
        self.offsets_x_value = np.array([[self.img_width - _x for _x in range(self.img_width * 2)] 
                                         for _y in range(self.img_height * 2)])
        self.offsets_y_value = np.array([[self.img_height - _y for _x in range(self.img_width * 2)] 
                                         for _y in range(self.img_height * 2)])
        
    def __getitem__(self, index):  
       
        starttime = datetime.now() 
        img = self.transform(Image.open(self.imgs[index]))
        #img = self.transform(Image.open(self.imgs[index]).convert('RGB'))
        labels = self.labels[index]

        visible = np.zeros(self.num_classes)
        keypoints = np.zeros((self.num_classes, 2))      
     
        maps = np.zeros((self.num_classes, self.img_height, self.img_width), dtype='float32')
        offsets_x = np.zeros((self.num_classes, self.img_height, self.img_width), dtype='float32')
        offsets_y = np.zeros((self.num_classes, self.img_height, self.img_width), dtype='float32')
        
        for i in range(0, self.num_classes * 3, 3):
            x = labels[i]
            y = labels[i + 1]
            
            _i = i // 3

            if labels[i + 2] > 0:
                visible[_i] = 1
            else:
                visible[_i] = 0
            
            keypoints[_i][0] = x
            keypoints[_i][1] = y

            if x == 0 and y == 0:
                maps[_i] = np.zeros((self.img_height, self.img_width))
                continue
            if self.img_height - y < 0 or self.img_width - x < 0:
                continue          
            maps[_i] = self.map_value[self.img_height - y : self.img_height * 2 - y, 
                                      self.img_width  - x : self.img_width * 2  - x]       
            maps[_i][maps[_i] <= self.radius] = 1
            maps[_i][maps[_i] >  self.radius] = 0
            offsets_x[_i] = self.offsets_x_value[self.img_height - y : self.img_height * 2 - y, 
                                                 self.img_width  - x : self.img_width * 2  - x]
            offsets_y[_i] = self.offsets_y_value[self.img_height - y : self.img_height * 2 - y, 
                                                 self.img_width  - x : self.img_width * 2  - x]      
        return img, (maps, offsets_x, offsets_y), (visible, keypoints)
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    NUM_CLASSES = 4
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    RADIUS = 5
    test_dataset = KeypointsDataset('/host/data/pull_hold_ends/test/images',
                           '/host/data/pull_hold_ends/test/keypoints', NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS, transform=transform)
    print(test_dataset[0])
 
