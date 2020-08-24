import numpy as np
import os
import cv2

if __name__ == '__main__':
    dirname = 'blue_rope_textures'
    for f in os.listdir(dirname):
        if f != '.DS_Store':
            path_to_img = os.path.join(dirname, f)
            img16 = cv2.imread(path_to_img)
            img8 = (img16/256).astype('uint8')
            cv2.imwrite(path_to_img, img8)
