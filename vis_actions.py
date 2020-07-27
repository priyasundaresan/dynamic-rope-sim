import cv2
import numpy as np
import argparse
from xml.etree import ElementTree
import os
import math
import json
import colorsys

def show_kpts(idx, image_dir):
    image_filename = "{0:05d}.jpg".format(idx)
    img = cv2.imread('images/{}'.format(image_filename))
    vis = img.copy()
    kpts = np.load('%s/%05d.npy'%(image_dir, idx))
    kpts = kpts.reshape(3,2)
    pull_loc, drop_loc, hold_loc = kpts.astype(int)
    cv2.circle(vis, tuple(hold_loc), 3, (255,0,0), -1)
    cv2.arrowedLine(vis, tuple(pull_loc), tuple(drop_loc), (0,255,0), 2)
    annotated_filename = "{0:06d}_annotated.png".format(idx)
    print("Annotating: %d"%idx)
    cv2.imwrite('./annotated/{}'.format(annotated_filename), vis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='actions')
    args = parser.parse_args()
    if not os.path.exists("./annotated"):
        os.makedirs('./annotated')
    else:
        os.system("rm -rf ./annotated")
        os.makedirs("./annotated")
    for i in range(len(os.listdir(args.dir))):
        show_kpts(i, args.dir)
