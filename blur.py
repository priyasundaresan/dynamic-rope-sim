import cv2
import os
import argparse
import numpy as np
from shutil import copyfile
import json

def blur(filename, dir, end_scale=1):
    num = filename[:6]
    mask_filename = num + '_visible_mask.png'
    img_filename = filename
    img = cv2.imread('./%s/images/%s'%(dir, filename)).copy()
    original_x = 640
    original_y = 480
    # scale = np.random.uniform(7, 10)
    # resized = cv2.resize(img, (int(original_x/scale), int(original_y/scale)))
    resized = img
    blur_img = cv2.resize(resized, (original_x, original_y))
    blur_img = cv2.resize(resized, (int(original_x/end_scale), int(original_y/end_scale)))
    cv2.imwrite('./image_blur/images/{}'.format(filename), blur_img)

if __name__ == '__main__':
    # goes from ./{dir}/images and ./{dir}/image_masks to ./image_blur/images and ./image_blur/image_masks
    # runs python mask.py
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='.')
    parser.add_argument('-s', '--scale', type=int, default=1)
    args = parser.parse_args()
    if os.path.exists('./image_blur'):
        os.system('rm -r ./image_blur')
    os.mkdir('./image_blur')
    os.mkdir('./image_blur/images')
    os.mkdir('./image_blur/image_masks')
    for filename in os.listdir('./{}/images'.format(args.dir)):
    	try:
    		print("Blurring %s" % filename)
    		blur(filename, args.dir, args.scale)
    	except:
            pass
    if args.scale != 1:
        # fix knots_info.json for scale
        with open("./{}/images/knots_info.json".format(args.dir), "r") as stream:
            knots_info = json.load(stream)
            print("loaded knots info")
        scaled_annot= {}
        original_x, original_y = 640, 480
        scaled_x, scaled_y = int(original_x/args.scale), int(original_y/args.scale)
        for img in knots_info.keys():
            pixels = knots_info[img]
            scaled_pixels = []
            for p in pixels:
                scaled_p = [min(int(p[0][0]/args.scale), scaled_x-1), min(int(p[0][1]/args.scale), scaled_y-1)]
                scaled_pixels.append([scaled_p])

            scaled_annot[img] = scaled_pixels
        with open("./image_blur/images/knots_info.json", 'w') as outfile:
            json.dump(scaled_annot, outfile, sort_keys=True, indent=2)

        # fix masks for scale
        for mask_filename in os.listdir('./{}/image_masks'.format(args.dir)):
            print("Resizing "+mask_filename)
            mask = cv2.imread('./%s/image_masks/%s'%(args.dir, mask_filename)).copy()
            mask_resized = cv2.resize(mask, (int(original_x/args.scale), int(original_y/args.scale)))
            cv2.imwrite('./image_blur/image_masks/{}'.format(mask_filename), mask_resized)
        os.system('python mask.py --dir ./image_blur/image_masks')

    else:
        os.system('cp {}/images/knots_info.json ./image_blur/images/knots_info.json'.format(args.dir))
        os.system('python mask.py')
        os.system('cp -r ./image_masks ./image_blur')
