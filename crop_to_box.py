import cv2
import colorsys
import json
import os
import sys
import argparse
from xml.etree import ElementTree
import numpy as np

def process(img_idx, input_img_dir, output_img_dir, crop_size=(200,200), plot=False):
    
    img_filename = '%05d.jpg'%img_idx
    img = cv2.imread(os.path.join(input_img_dir, img_filename))

    crop_width, crop_height = crop_size

    tree  = ElementTree.parse('annots/%05d.xml'%img_idx)
    root = tree.getroot()
    hold_annot = root.findall('.//hold_pixel')[0]
    hold_x = int(hold_annot.find('x').text) + np.random.randint(-15,-10)
    hold_y = int(hold_annot.find('y').text) + np.random.randint(-5,5)
    #hold_x = int(hold_annot.find('x').text) 
    #hold_y = int(hold_annot.find('y').text) 

    box_x = hold_x - crop_width//2
    box_y = hold_y - crop_height//2
    img_crop = img[hold_y-(crop_height//2):hold_y+(crop_height//2),\
                   hold_x-(crop_width//2):hold_x+(crop_width//2)]

    print(img_crop.shape)
    if plot:
        vis = img_crop.copy()
        cv2.imshow("vis", vis)
        cv2.waitKey(0)
    try:
        cv2.imwrite(os.path.join(output_img_dir, img_filename), img_crop)
    except:
        pass

if __name__ == '__main__':
    output_img_dir = "images_cropped"
    input_img_dir = "images"
    if os.path.exists(output_img_dir):
        os.system("rm -rf %s"%output_img_dir)
    os.makedirs(output_img_dir)
    for i in range(len(os.listdir(input_img_dir))-1):
        print("Processing image %04d"%i)
        process(i, input_img_dir, output_img_dir, plot=False)
