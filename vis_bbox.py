import cv2
import numpy as np
import argparse
from xml.etree import ElementTree
import os
import math
import json
import colorsys

def show_knots(idx, knots_info, save=True):
    image_filename = "{0:05d}.jpg".format(idx)
    img = cv2.imread('images/{}'.format(image_filename))
    pixels = knots_info[str(idx)]
    pixels = [i[0] for i in pixels]
    vis = img.copy()
    print("Annotating %06d"%idx)
    start, end = pixels
    vis = cv2.rectangle(vis, tuple(start), tuple(end), (255, 0, 0), 2) 
    if save:
    	annotated_filename = "{0:06d}_annotated.png".format(idx)
    	cv2.imwrite('./annotated/{}'.format(annotated_filename), vis)

def show_boxes(idx, save=True):
    # load and parse the file
    image_filename = "{0:05d}.jpg".format(idx)
    #image_filename = "{0:06d}_rgb.png".format(idx)
    img = cv2.imread('images/{}'.format(image_filename))
    vis = img.copy()
    filename = 'annots/%05d.xml'%idx
    tree = ElementTree.parse(filename)
    # get the root of the document
    root = tree.getroot()
    # extract each bounding box
    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        vis = cv2.rectangle(vis, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2)
    if save:
    	annotated_filename = "{0:06d}_annotated.png".format(idx)
    	cv2.imwrite('./annotated/{}'.format(annotated_filename), vis)
        #coors = [xmin, ymin, xmax, ymax]
        #boxes.append(coors)

def generate_crops(idx, knots_info, save_resized=True, aspect=(640, 480)):
    image_filename = "{0:05d}.jpg".format(idx)
    img = cv2.imread('images/{}'.format(image_filename))
    pixels = knots_info[str(idx)]
    pixels = [i[0] for i in pixels]
    print("Cropping %06d"%idx)
    start, end = pixels
    xmin, ymin = start
    xmax, ymax = end
    width = xmax - xmin
    height = ymax - ymin
    new_width = int((height*aspect[0])/aspect[1])
    offset = new_width - width
    xmin -= int(offset/2)
    xmax += offset - int(offset/2)
    crop = img[ymin:ymax, xmin:xmax]
    if idx==22:
        cv2.imshow("crop", crop)
        cv2.waitKey(0)
    resized = cv2.resize(crop, aspect)
    result = resized if save_resized else crop
    cropped_filename = "{0:06d}_cropped.png".format(idx)
    cv2.imwrite('./crops/{}'.format(cropped_filename), result)
    #vis = cv2.rectangle(vis, tuple(start), tuple(end), (255, 0, 0), 2) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=len(os.listdir('./images')) - 1)
    args = parser.parse_args()
    if not os.path.exists("./annotated"):
        os.makedirs('./annotated')
    else:
        os.system("rm -rf ./annotated")
        os.makedirs("./annotated")
    if not os.path.exists("./crops"):
        os.makedirs('./crops')
    else:
        os.system("rm -rf ./crops")
        os.makedirs("./crops")
    print("parsed")
    print("loaded knots info")
    for i in range(args.num):
        show_boxes(i)
        #generate_crops(i, knots_info)
