from dense_correspondence_network import DenseCorrespondenceNetwork
import random
import yaml
import json
import time
#from yaml import CLoader
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2
import torch
from PIL import Image
import argparse
from find_correspondences import CorrespondenceFinder

'''
example usage: python vis_descriptors.py --mask --mask_dir ../image_masks --gpu_id 0,1
'''

parser = argparse.ArgumentParser()
parser.add_argument('--network_dir', default='../networks', type=str, help='path to dir with trained networks')
parser.add_argument('--network', default='rope_cyl_knots', type=str, help='path to network')
parser.add_argument('--image_dir', default='../../rope_cyl_knots/processed/images', type=str, help='path to image dir')
parser.add_argument('--save_dir', default='./output', type=str, help='path to output visualizations')
parser.add_argument('--mask', action='store_true', help='make masked visualizations')
parser.add_argument('--mask_dir', default='', type=str, help='path to folder with image masks')
parser.add_argument('--gpu_id', default='', type=str, help='CUDA visible devices')

def normalize_descriptor(res, stats=None):
    """
    Normalizes the descriptor into RGB color space
    :param res: numpy.array [H,W,D]
        Output of the network, per-pixel dense descriptor
    :param stats: dict, with fields ['min', 'max', 'mean'], which are used to normalize descriptor
    :return: numpy.array
        normalized descriptor
    """
    if stats is None:
        res_min = res.min()
        res_max = res.max()
    else:
        res_min = np.array(stats['min'])
        res_max = np.array(stats['max'])

    normed_res = np.clip(res, res_min, res_max)
    eps = 1e-10
    scale = (res_max - res_min) + eps
    normed_res = (normed_res - res_min) / scale
    return normed_res

def make_descriptors_images(cf, image_folder, save_images_dir, descriptor_stats_config, make_masked_video=False, mask_folder=None):

    for img_file in sorted(os.listdir(image_folder)):
        start = time.time()
        idx_str = img_file.split("_rgb")[0]
        img_file_fullpath = os.path.join(image_folder, img_file)
        rgb_a = Image.open(img_file_fullpath).convert('RGB')

        # compute dense descriptors
        # This takes in a PIL image!
        rgb_a_tensor = cf.rgb_image_to_tensor(rgb_a)

        # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
        res_a = cf.dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
        #descriptor_image_stats = yaml.load(file(descriptor_stats_config), Loader=CLoader)
        descriptor_image_stats = yaml.load(file(descriptor_stats_config))
        res_a = normalize_descriptor(res_a, descriptor_image_stats["mask_image"])

        # This chunk of code would produce masked descriptors
        if make_masked_video:
            mask_name = idx_str + "_mask.png"
            mask_filename = os.path.join(mask_folder, mask_name)
            mask = np.asarray(Image.open(mask_filename))
            res_a[np.where((mask == [0]))] = 0

        # save rgb image, descriptor image, masked descriptor image
        save_file_name = os.path.join(save_images_dir, idx_str + "_res.png")
        plt.imsave(save_file_name, res_a)
        print "forward and saving at rate", time.time()-start

if __name__ == '__main__':
    args = parser.parse_args()
    # Use GPU (0, 1 on Triton1)
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
    # Path to network 
    network_path = os.path.join(args.network_dir, args.network)
    dcn = DenseCorrespondenceNetwork.from_model_folder(network_path, model_param_file=os.path.join(network_path, '003501.pth'))
    dcn.eval()
    with open(os.path.join('../cfg', 'dataset_info.json'), 'r') as f:
        dataset_stats = json.load(f)
    dataset_mean, dataset_std_dev = dataset_stats["mean"], dataset_stats["std_dev"]
    cf = CorrespondenceFinder(dcn, dataset_mean, dataset_std_dev)
    descriptor_stats_config = os.path.join(network_path, 'descriptor_statistics.yaml')

    make_descriptors_images(cf, args.image_dir, args.save_dir, descriptor_stats_config, make_masked_video=args.mask, mask_folder=args.mask_dir)
