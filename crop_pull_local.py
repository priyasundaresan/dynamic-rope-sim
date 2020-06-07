import cv2
import colorsys
import json
import os
import sys
import argparse
from xml.etree import ElementTree
import numpy as np

def process(img_idx, knots_info, crop_size=(50,50), plot=False):
    img_filename = 'images/%06d_rgb.png'%img_idx
    depth_filename = 'images_depth/%06d_rgb.png'%img_idx
    mask_vis_filename = 'image_masks/%06d_visible_mask.png'%img_idx
    mask_filename = 'image_masks/%06d_mask.png'%img_idx
    img = cv2.imread(img_filename)
    depth_img = cv2.imread(depth_filename)
    mask_vis = cv2.imread(mask_vis_filename)
    mask = cv2.imread(mask_filename)

    pixels = knots_info[str(img_idx)]
    crop_pixels = np.array([i[0] for i in pixels])
    tree  = ElementTree.parse('annots/%05d.xml'%img_idx)
    root = tree.getroot()
    hold_annot = root.findall('.//hold_pixel')[0]
    hold_x = int(hold_annot.find('x').text) + np.random.randint(-5,5)
    hold_y = int(hold_annot.find('y').text) + np.random.randint(-5,5)

    crop_width, crop_height = crop_size
    box_x = hold_x - crop_width//2
    box_y = hold_y - crop_height//2
    img_crop = img[hold_y-(crop_height)//2:hold_y+(crop_height)//2, hold_x-(crop_width)//2:hold_x+(crop_width)//2]
    depth_crop = depth_img[hold_y-(crop_height)//2:hold_y+(crop_height)//2, hold_x-(crop_width)//2:hold_x+(crop_width)//2]
    mask_vis_crop = mask_vis[hold_y-(crop_height)//2:hold_y+(crop_height)//2, hold_x-(crop_width)//2:hold_x+(crop_width)//2]
    mask_crop = mask[hold_y-(crop_height)//2:hold_y+(crop_height)//2, hold_x-(crop_width)//2:hold_x+(crop_width)//2]
    cv2.imwrite(os.path.join('image_crop', img_filename), img_crop)
    cv2.imwrite(os.path.join('image_crop', depth_filename), depth_crop)
    cv2.imwrite(os.path.join('image_crop', mask_filename), mask_crop)
    cv2.imwrite(os.path.join('image_crop', mask_filename), mask_vis_crop)
    crop_pixels[:, 0] -= box_x
    crop_pixels[:, 1] -= box_y

    if plot:
        vis = img_crop.copy()
        for i, (u, v) in enumerate(crop_pixels):
            (r, g, b) = colorsys.hsv_to_rgb(float(i)/len(crop_pixels), 1.0, 1.0)
            R, G, B = int(255 * r), int(255 * g), int(255 * b)
            cv2.circle(vis,(int(u), int(v)), 1, (R, G, B), -1)
        cv2.imshow("vis", vis)
        cv2.waitKey(0)

    new_pixels = [[i] for i in crop_pixels.tolist()]
    knots_info[str(img_idx)] = new_pixels
    return knots_info

if __name__ == '__main__':
    if os.path.exists("./image_crop"):
        os.system("rm -rf ./image_crop")
    os.makedirs('./image_crop')
    if os.path.exists("./image_crop/images"):
        os.system("rm -rf ./image_crop/images")
    os.makedirs('./image_crop/images')
    if os.path.exists("./image_crop/image_masks"):
        os.system("rm -rf ./image_crop/image_masks")
    os.makedirs('./image_crop/image_masks')
    if os.path.exists("./image_crop/images_depth"):
        os.system("rm -rf ./image_crop/images_depth")
    os.makedirs('./image_crop/images_depth')
    with open("{}/knots_info.json".format('images', 'r')) as stream:
        knots_info = json.load(stream)
    for i in range(len(os.listdir('images'))-1):
        knots_info = process(i, knots_info)
    with open("./image_crop/images/knots_info.json", 'w') as outfile:
        json.dump(knots_info, outfile, sort_keys=True, indent=2)
    
