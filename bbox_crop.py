import cv2
import os
import sys
import argparse
import numpy as np
from shutil import copyfile
import json
from xml.etree import ElementTree
from mrcnn.model import MaskRCNN

sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/pytorch-segmentation-detection"))
sys.path.insert(0, os.path.join(os.getcwd(), "dense_correspondence/tools"))
sys.path.insert(0, os.path.join(os.getcwd(), "mrcnn_bbox/tools"))

from predict import BBoxFinder, PredictionConfig

def crop_and_resize(box, img, aspect=(80,60)):
    x1, y1, x2, y2 = box
    x_min, x_max = min(x1,x2), max(x1,x2)
    y_min, y_max = min(y1,y2), max(y1,y2)
    box_width = x_max - x_min
    box_height = y_max - y_min

    new_width = int((box_height*aspect[0])/aspect[1])
    offset = new_width - box_width
    x_min -= int(offset/2)
    x_max += offset - int(offset/2)

    crop = img[y_min:y_max, x_min:x_max]
    resized = cv2.resize(crop, aspect)
    
    rescale_factor = new_width/aspect[0]
    offset = (x_min, y_min)
    return resized, rescale_factor, offset

def pixel_full_to_crop(pixels, crop_rescale_factor, x_offset, y_offset, aspect=(80,60)):
    ret = []
    for p in pixels:
        new_p = [int((p[0] - x_offset)/crop_rescale_factor), int((p[1] - y_offset)/crop_rescale_factor)]
        ret.append(new_p)
        #if not (new_p[0] < 0 or new_p[1] < 0 or new_p[0] > 80 or new_p[1] > 60):
        #    ret.append(new_p)
    return ret

def crop(filename, dir, bbox_predictor, knots_info):
    num = int(filename[:6])
    mask_filename = '%06d_visible_mask.png'%num
    img_filename = filename
    img = cv2.imread('./%s/images/%s'%(dir, filename)).copy()
    mask = cv2.imread('./%s/image_masks/%s'%(dir, mask_filename)).copy()
    depth = cv2.imread('./%s/images_depth/%s'%(dir, filename)).copy()
    # get box
    boxes = bbox_predictor.predict(img, plot=False, annotate=False)
    boxes = sorted(boxes, key=lambda box: box[0][2], reverse=True)
    box = boxes[0][0]

    #filename = 'annots/%05d.xml'%num
    #tree = ElementTree.parse(filename)
    #root = tree.getroot()
    #box = root.findall('.//bndbox')[0]
    #xmin = int(box.find('xmin').text)
    #ymin = int(box.find('ymin').text)
    #xmax = int(box.find('xmax').text)
    #ymax = int(box.find('ymax').text)
    #box = [xmin, ymin, xmax, ymax]
    # crop img and mask
    cropped_img, rescale_factor_img, (x_off_img, y_off_img) = crop_and_resize(box, img)
    cropped_mask, _, _ = crop_and_resize(box, mask)
    cropped_depth, _, _ = crop_and_resize(box, depth)

    # rescale annotations
    num_annots = len(knots_info[str(num)])
    pixels = [i[0] for i in knots_info[str(num)]]
    cropped_pixels = pixel_full_to_crop(pixels, rescale_factor_img, x_off_img, y_off_img)
    if not len(cropped_pixels) == num_annots: # if annots go off the crop
        print(len(cropped_pixels), num_annots)
        knots_info.pop(str(num), None)
        return knots_info
    knots_info[str(num)] = [[i] for i in cropped_pixels]
    cv2.imwrite('./image_crop/images/{}'.format(img_filename), cropped_img)
    cv2.imwrite('./image_crop/image_masks/{}'.format(mask_filename), cropped_mask)
    cv2.imwrite('./image_crop/images_depth/{}'.format(img_filename), cropped_depth)
    return knots_info

if __name__ == '__main__':
    # goes from ./{dir}/images and ./{dir}/image_masks to ./image_crop/images and ./image_crop/image_masks
    # runs python mask.py
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='.')
    parser.add_argument('-b', '--bbox_detector', type=str, default="knot_capsule_mult")
    args = parser.parse_args()
    if os.path.exists('./image_crop'):
        os.system('rm -r ./image_crop')
    os.mkdir('./image_crop')
    os.mkdir('./image_crop/images')
    os.mkdir('./image_crop/image_masks')
    os.mkdir('./image_crop/images_depth')

    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model_path = 'mrcnn_bbox/networks/{}/mask_rcnn_knot_cfg_0010.h5'.format(args.bbox_detector)
    model.load_weights(model_path, by_name=True)
    bbox_predictor = BBoxFinder(model, cfg)
    #bbox_predictor = None

    with open("./{}/images/knots_info.json".format(args.dir), "r") as stream:
        knots_info = json.load(stream)
        print("loaded knots info")

    for filename in sorted(os.listdir('./{}/images'.format(args.dir))):
        try:
            print("Cropping %s" % filename)
            knots_info = crop(filename, args.dir, bbox_predictor, knots_info)
        except:
            pass

    # fix knots_info.json for crop
    with open("./image_crop/images/knots_info.json", 'w') as outfile:
        json.dump(knots_info, outfile, sort_keys=True, indent=2)
    
    os.system('python mask.py --dir ./image_crop/image_masks')
    os.system('python reorder.py --dir image_crop')
