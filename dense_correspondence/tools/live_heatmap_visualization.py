"""
ADAPTED THIS SCRIPT FROM DGX TO TRITON1 SETUP
"""
import sys
import json
import os, random
import cv2
import numpy as np
import copy
from PIL import Image, ImageOps
from dense_correspondence_network import DenseCorrespondenceNetwork
from find_correspondences import CorrespondenceFinder 

COLOR_RED = np.array([0, 0, 255])
COLOR_GREEN = np.array([0,255,0])

class HeatmapVisualization(object):
    """
    Launches a live interactive heatmap visualization.
    """
    def __init__(self, dcn, dataset_mean, dataset_std_dev, image_dir):
        self._norm_diff_threshold = 0.25
        self._heatmap_vis_upper_bound = 0.75
        self._blend_weight_original_image = 0.3
        self._cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev)
        self._reticle_color = COLOR_GREEN
        self._network_reticle_color = COLOR_RED
        self._image_dir = image_dir
        self._image_width = 640
        self._image_height = 480

    def _get_new_images(self):
        """
        Gets a new pair of images
        :return:
        :rtype:
        """
        img1_index = random.choice(range(0, len(os.listdir(self._image_dir))))
        img2_index = random.choice(range(0, len(os.listdir(self._image_dir))))
        print(img1_index, img2_index)
        filename = "%06d_rgb.png"
        #filename = "%06d.png"
        #f1 = os.path.join(self._image_dir, filename % img1_index)
        f1 = os.path.join('../../reference_images/knot_reference.png')
        f2 = os.path.join(self._image_dir, filename % img2_index)
        self.img1_pil = Image.open(f1).convert('RGB').resize((self._image_width, self._image_height))
        self.img2_pil = Image.open(f2).convert('RGB').resize((self._image_width, self._image_height))
	self._compute_descriptors()

    def draw_reticle(self, img, x, y, label_color):
        white = (255, 255, 255)
        cv2.circle(img,(x,y),10,label_color,1)
        cv2.circle(img,(x,y),11,white,1)
        cv2.circle(img,(x,y),12,label_color,1)
        cv2.line(img,(x,y+1),(x,y+3),white,1)
        cv2.line(img,(x+1,y),(x+3,y),white,1)
        cv2.line(img,(x,y-1),(x,y-3),white,1)
        cv2.line(img,(x-1,y),(x-3,y),white,1)

    def _compute_descriptors(self):
        """
        Computes the descriptors for image 1 and image 2 for each network
        :return:
        :rtype:
        """
	print "computing descriptors"
        self.img1 = self._cf.pil_image_to_cv2(self.img1_pil)
        self.img2 = self._cf.pil_image_to_cv2(self.img2_pil)
        self.rgb_1_tensor = self._cf.rgb_image_to_tensor(self.img1_pil)
        self.rgb_2_tensor = self._cf.rgb_image_to_tensor(self.img2_pil)
        self.img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_RGB2GRAY) / 255.0
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY) / 255.0

        cv2.imshow('source', self.img1)
        cv2.imshow('target', self.img2)

        self._res_a = self._cf.dcn.forward_single_image_tensor(self.rgb_1_tensor).data.cpu().numpy()
        self._res_b = self._cf.dcn.forward_single_image_tensor(self.rgb_2_tensor).data.cpu().numpy()
        self.find_best_match(None, 0, 0, None, None)

    def scale_norm_diffs_to_make_heatmap(self, norm_diffs, threshold):
        """
        Scales the norm diffs to make a heatmap. This will be scaled between 0 and 1.
        0 corresponds to a match, 1 to non-match

        :param norm_diffs: The norm diffs
        :type norm_diffs: numpy.array [H,W]
        :return:
        :rtype:
        """
        heatmap = np.copy(norm_diffs)
        greater_than_threshold = np.where(norm_diffs > threshold)
        heatmap = heatmap / threshold * self._heatmap_vis_upper_bound # linearly scale [0, threshold] to [0, 0.5]
        heatmap[greater_than_threshold] = 1 # greater than threshold is set to 1
        heatmap = heatmap.astype(self.img1_gray.dtype)
        return heatmap


    def find_best_match(self, event,u,v,flags,param):

        """
        For each network, find the best match in the target image to point highlighted
        with reticle in the source image. Displays the result
        :return:
        :rtype:
        """

        img_1_with_reticle = np.copy(self.img1)
        self.draw_reticle(img_1_with_reticle, u, v, self._reticle_color)
        cv2.imshow("source", img_1_with_reticle)

        alpha = self._blend_weight_original_image
        beta = 1 - alpha

        img_2_with_reticle = np.copy(self.img2)

        print "\n\n"

        self._res_uv = dict()

        res_a = self._res_a
        res_b = self._res_b
        best_match_uv, best_match_diff, norm_diffs = \
            self._cf.dcn.find_best_match((u, v), res_a, res_b)
        self._res_uv = dict()
        self._res_uv['source'] = res_a[v, u, :].tolist()
        self._res_uv['target'] = res_b[v, u, :].tolist()

        #print "best match diff: %.3f" %(best_match_diff)

        threshold = self._norm_diff_threshold
        #print(norm_diffs.shape)
        heatmap = self.scale_norm_diffs_to_make_heatmap(norm_diffs, threshold)

        reticle_color = self._network_reticle_color
        self.draw_reticle(heatmap, best_match_uv[0], best_match_uv[1], reticle_color)
        self.draw_reticle(img_2_with_reticle, best_match_uv[0], best_match_uv[1], reticle_color)
        blended = cv2.addWeighted(self.img2_gray, alpha, heatmap, beta, 0)
        cv2.imshow("heatmap", blended)

        cv2.imshow("target", img_2_with_reticle)

    def run(self):
        self._get_new_images()
	print "got new images"
        cv2.namedWindow('target')
        cv2.setMouseCallback('source', self.find_best_match)

        while True:
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('n'):
                print "HEY"
                self._get_new_images()
            elif k == ord('s'):
                print "HEY"
                img1_pil = self.img1_pil
                img2_pil = self.img2_pil
                self.img1_pil = img2_pil
                self.img2_pil = img1_pil
                self._compute_descriptors()

if __name__ == "__main__":
    base_dir = '../networks'
    #network_dir = 'rope_cyl_400_dim16'
    #network_dir = 'rope_400_cyl_rot_16'
    network_dir = 'full_length_corr'
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    dcn.eval()
    with open('../cfg/dataset_info.json', 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    image_dir = '../../rope_400_cyl_rot/processed/images'
    #image_dir = '../../images'
    heatmap_vis = HeatmapVisualization(dcn, dataset_mean, dataset_std_dev, image_dir)
    print "starting heatmap vis"
    heatmap_vis.run()
    print "ran heatmap_vis"
    cv2.destroyAllWindows()
