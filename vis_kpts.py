import cv2
import numpy as np
import argparse
from xml.etree import ElementTree
import os
import math
import json
import colorsys

def show_kpts(idx):
    image_filename = "{0:05d}.jpg".format(idx)
    img = cv2.imread('images/{}'.format(image_filename))
    vis = img.copy()
    kpts = np.load('keypoints/%05d.npy'%idx)
    for i, (u,v) in enumerate(kpts):    
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/kpts.shape[0], 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(vis,(u,v),4,(R,G,B), -1)
    annotated_filename = "{0:06d}_annotated.png".format(idx)
    print("Annotating: %d"%idx)
    cv2.imwrite('./annotated/{}'.format(annotated_filename), vis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=len(os.listdir('./images')) - 1)
    parser.add_argument('-d', '--dir', type=str, default='images')
    args = parser.parse_args()
    if not os.path.exists("./annotated"):
        os.makedirs('./annotated')
    else:
        os.system("rm -rf ./annotated")
        os.makedirs("./annotated")
    for i in range(args.num):
        show_kpts(i)
