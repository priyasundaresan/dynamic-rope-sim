import cv2
import os
import numpy as np
import sys
from PIL import Image
BASE_DIR = '/Users/jennifergrannen/Documents/Berkeley/projects/rope/dynamic-rope-sim'
sys.path.insert(0, os.path.join(BASE_DIR, "keypoints_dir/src"))

from model import Keypoints
from prediction_simple import Prediction
from dataset import transform

import torch

def load_kp(kp_dir, network_name, image_width, image_height, num_classes):
    network_dir = os.path.join(kp_dir, 'checkpoints', network_name)
    keypoints = Keypoints(num_classes, img_height=image_height, img_width=image_width)
    keypoints.load_state_dict(torch.load(os.path.join(network_dir, os.listdir(network_dir)[0]), map_location='cpu'))
    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    if use_cuda:
        torch.cuda.set_device(0)
        keypoints = keypoints.cuda()
    prediction = Prediction(keypoints, num_classes, image_height, image_width, None, None, use_cuda)
    return prediction

def kp_matches(prediction, path_to_curr_img, curr_frame, num_classes, use_cuda=0):
    img = Image.open(path_to_curr_img)
    img = np.array(img)
    img_t = transform(img)
    if use_cuda:
        img_t = img_t.cuda()
    result, keypoints = prediction.predict(img_t)
    # prediction.plot(img, result, keypoints, image_id=curr_frame)

    print("KEYPOINTS", keypoints)
    # TODO: later return end1, end2 pixels if ends_cf OR hold, pull pixels if knot_cf
    # currently returns end1, pull, hold, end2
    return keypoints

# dir = "."
# dir = "datasets/kpt_chord_ph/train"
kp_dir = os.path.join(BASE_DIR, 'keypoints_dir')
# dir = "image_crop"
dir = "datasets/kpt_chord_ends/train"

for filename in sorted(os.listdir("{}/keypoints".format(dir))):
    print(filename)
    num = filename[:5]
    # keypoints = np.load("{}/keypoints/".format(dir)+filename)
    keypoints = kp_matches(load_kp(kp_dir, "kpt_chord_ends_e20", 640, 480, 2), "{}/images/{}.jpg".format(dir, num), 0, 2)
    result = cv2.imread("{}/images/{}.jpg".format(dir, num))
    print("{}/images/{}.jpg".format(dir, num))
    for kp in keypoints:
        result = cv2.circle(result, tuple(kp), 3, (255, 0, 0), -1)

    cv2.imshow("GT", result)
    cv2.waitKey(0)
