import os
import random
import colorsys
import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def darken_img(img, filename, output_dir_img, show=False):
    img = adjust_gamma(img, gamma=0.53)
    alpha = 1.2 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imwrite(os.path.join(output_dir_img, filename), img)
    if show:
        cv2.imshow('img', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    img_dir = 'images'
    output_dir_img = 'images_dark'
    if not os.path.exists(output_dir_img):
        os.mkdir(output_dir_img)
    for i in range(len(os.listdir(img_dir))):
        print(i)
        filename = '%05d.jpg'%i
        img = cv2.imread(os.path.join(img_dir, filename))
        darken_img(img, filename, output_dir_img, show=False)

