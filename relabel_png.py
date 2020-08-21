import cv2
import os
import argparse
import json
import random
if __name__ == '__main__':
    # goes from ./{dir}/images and ./{dir}/image_masks to ./{dir}/images and ./{dir}/image_masks ordered
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='.')
    args = parser.parse_args()
    if "reordered_png" in os.listdir('/Users/priyasundaresan/Desktop'):
        os.system('rm -r /Users/priyasundaresan/Desktop/reordered_png')
    os.system('mkdir /Users/priyasundaresan/Desktop/reordered_png')
    os.system('mkdir /Users/priyasundaresan/Desktop/reordered_png/images')
    i = 0
    for filename in sorted(os.listdir('.')):
    # for filename in os.listdir('.'):
        print("Relabeling " + filename)
        try:
            num = int(filename[:6])
            save_img_filename = '%06d_rgb.png'%i
            img = cv2.imread('./%s/%s'%(args.dir, filename)).copy()
            cv2.imwrite('/Users/priyasundaresan/Desktop/reordered_png/images/{}'.format(save_img_filename), img)
            i += 1
        except:
            pass
    os.system('rm *')
    os.system('cp -a ~/Desktop/reordered_png/images/. .')
