from dense_correspondence_network import DenseCorrespondenceNetwork
import pprint
import json
import sys
import os
import cv2
import numpy as np
import copy
from PIL import Image
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

#from image_utils import geometric_median, sample_nearest_points_on_mask, farthest_pixel_correspondence
from image_utils import *
#from pixel_selector import PixelSelector

class CorrespondenceFinder:
    def __init__(self, dcn, dataset_mean, dataset_std_dev, image_width=640, image_height=480):
        self.dcn = dcn
        self.dataset_mean = dataset_mean
        self.dataset_std_dev = dataset_std_dev
        self.image_width = image_width
        self.image_height = image_height

    def get_rgb_image(self, rgb_filename):
        return Image.open(rgb_filename).convert('RGB').resize((self.image_width, self.image_height))

    def get_grayscale_image(self, grayscale_filename):
        return Image.open(grayscale_filename).resize((self.image_width, self.image_height))

    def pil_image_to_cv2(self, pil_image):
        return np.array(pil_image)[:, :, ::-1].copy() # open and convert between BGR and RGB

    def rgb_image_to_tensor(self, img):
        norm_transform = transforms.Normalize(self.dataset_mean, self.dataset_std_dev)
        return transforms.Compose([transforms.ToTensor(), norm_transform])(img)
        #return transforms.ToTensor()(img)

    def load_image_pair(self, img1_filename, img2_filename):
        self.img1_pil = self.get_rgb_image(img1_filename)
        self.img2_pil = self.get_rgb_image(img2_filename)
        self.img1 = cv2.resize(cv2.imread(img1_filename, 0), (self.image_width, self.image_height))
        self.img2 = cv2.resize(cv2.imread(img2_filename, 0), (self.image_width, self.image_height))
        print("loaded images successfully")

    def compute_descriptors(self):
        self.img1 = self.pil_image_to_cv2(self.img1_pil)
        self.img2 = self.pil_image_to_cv2(self.img2_pil)
        self.rgb_1_tensor = self.rgb_image_to_tensor(self.img1_pil)
        self.rgb_2_tensor = self.rgb_image_to_tensor(self.img2_pil)
        self.img1_descriptor = self.dcn.forward_single_image_tensor(self.rgb_1_tensor).data.cpu().numpy()
        self.img2_descriptor = self.dcn.forward_single_image_tensor(self.rgb_2_tensor).data.cpu().numpy()

    def find_k_best_matches(self, pixels, k, mode="median", annotate=True, hold=None):
        # Finds k best matches in descriptor space (either by median or mean filtering)
        max_range = float(len(pixels))
        pixel_matches = []
        model = None
        # best_matches, norm_diffs, model = self.dcn.find_best_match_for_descriptors_KNN(np.array(pixels), self.img1_descriptor, self.img2_descriptor, k)
        for i, (u, v) in enumerate(pixels):
            if mode == "median":
                best_matches, norm_diffs, norm_diffs_all = self.dcn.find_k_best_matches((u, v), self.img1_descriptor, self.img2_descriptor, k)
                best_match = np.round(np.median(best_matches, axis=0))
            elif mode == "geometric_median_cloud":
                cloud = sample_nearest_points_on_mask((u, v), self.img1, 10)
                best_matches, norm_diffs, model = self.dcn.find_best_match_for_descriptors_KNN(cloud, self.img1_descriptor, self.img2_descriptor, 1, model)
                best_match = geometric_median(best_matches.squeeze())
            else:
                best_matches, norm_diffs, norm_diffs_all = self.dcn.find_k_best_matches((u, v), self.img1_descriptor, self.img2_descriptor, k)
                best_match = np.round(np.mean(best_matches, axis=0))
            match = [int(best_match[0]), int(best_match[1])]
            pixel_matches.append(match)

        if not hold is None:
            pixel_matches = [[h[0]-15, h[1]] for h in hold]

        for i, (u, v) in enumerate(pixels):
            match = pixel_matches[i]
            if annotate:
                self.annotate_correspondence(u, v, match[0], match[1])
        return pixel_matches, pixels


    def find_best_match_pixel(self, u, v):
        best_match_uv, best_match_diff, norm_diffs = \
                self.dcn.find_best_match((u, v), self.img1_descriptor, self.img2_descriptor)
        return (best_match_uv, best_match_diff, norm_diffs)

    def find_best_matches_raw(self, pixels):
        max_range = float(len(pixels))
        for i, (u, v) in enumerate(pixels):
            best_match, norm_diff, norm_diffs = self.find_best_match_pixel(u, v)
            self.annotate_correspondence(u, v, int(best_match[0]), int(best_match[1]))

    def annotate_correspondence(self, u1, v1, u2, v2, line=False, flip=False):
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        src = self.img1
        dest = self.img2
        cv2.circle(src, (u1, v1), 4, color, -1)
        cv2.circle(dest, (u2, v2), 4, color, -1)
        if line:
            cv2.line(src, (u1, v1), (u2, v2), (255, 255, 255), 4)

    def show_side_by_side(self, pixels=None, plot=True):
        if not pixels is None: #pixels = [[original pts], [list of mapped to pixels]]
            for i in range(len(pixels[0])):
                self.annotate_correspondence(pixels[0][i][0], pixels[0][i][1], pixels[1][i][0], pixels[1][i][1])
        vis = np.concatenate((self.img1, self.img2), axis=0)
        if plot:
            cv2.imshow("correspondence", vis)
            k = cv2.waitKey(0)
            if k == 27:         # wait for ESC key to exit
                cv2.destroyAllWindows()
        return vis

if __name__ == '__main__':
    base_dir = '../networks'
    network_dir = 'rope_400_cyl_rot_16'
    dcn = DenseCorrespondenceNetwork.from_model_folder(os.path.join(base_dir, network_dir), model_param_file=os.path.join(base_dir, network_dir, '003501.pth'))
    dcn.eval()
    with open('../cfg/dataset_info.json', 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev)
    f1 = "../../reference_images/crop_ref.png"
    with open('../../reference_images/ref_pixels.json', 'r') as f:
        ref_annots = json.load(f)
        pull = [ref_annots["pull_x"], ref_annots["pull_y"]]
        hold = [ref_annots["hold_x"], ref_annots["hold_y"]]
        pixels = [pull, hold]
    for i in range(10,50,10):
        f2 = "../../rope_400_cyl_rot/processed/images/%06d_rgb.png"%i
        cf.load_image_pair(f1, f2)
        cf.compute_descriptors()
        best_matches = cf.find_k_best_matches(pixels, 50, mode="median")
        vis = cf.show_side_by_side()
