import os
import random
import colorsys
import numpy as np
import cv2

def flip_aug(img, keypoints, output_dir_img, output_dir_kpt, new_idx, show=False):
    flip_axis = random.choice((0,1))
    img_flip = cv2.flip(img, flip_axis)
    height, width, _ = img.shape
    keypoints = np.reshape(keypoints, (2,2))
    keypoints_aug = []
    for i, (u,v) in enumerate(keypoints):    
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/keypoints.shape[0], 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(img,(u,v),4,(R,G,B), -1)
        if flip_axis == 0:
            v = height-v
            keypoints_aug.append([u,v])
        else:
            u = width-u
            keypoints_aug.insert(0,[u,v])
    for i, (u,v) in enumerate(keypoints_aug):
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/keypoints.shape[0], 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(img_flip,(u,v),4,(R,G,B), -1)
    if show:
        cv2.imshow("vis", img)
        cv2.waitKey(0)
        cv2.imshow("vis", img_flip)
        cv2.waitKey(0)
    keypoints_aug = np.array(keypoints_aug)
    img_aug = img_flip
    cv2.imwrite(os.path.join(output_dir_img, "%05d.jpg"%new_idx), img_aug)
    np.save(os.path.join(output_dir_kpt, "%05d.npy"%new_idx), keypoints_aug)

if __name__ == '__main__':
    img_dir = 'images'
    keypoints_dir = 'keypoints'
    output_dir_img = img_dir
    output_dir_kpt = keypoints_dir
    idx = len(os.listdir(img_dir))
    for i in range(len(os.listdir(img_dir))):
        img = cv2.imread(os.path.join(img_dir, '%05d.jpg'%i))
        kpts = np.load(os.path.join(keypoints_dir, '%05d.npy'%i))
        flip_aug(img, kpts, output_dir_img, output_dir_kpt, idx+i, show=False)
