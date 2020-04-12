"""
Helper functions for processing images.
"""
import random

from PIL import Image
import random
import cv2
import numpy as np
from scipy.signal import correlate2d
#from autolab_core import Point
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize

def IOU(img1, img2):
	"""
	Takes in two binary images IMG1 and IMG2 and computes IOU over the masks.
	"""
	assert img1.shape == img2.shape
	nonzero_1, nonzero_2 = img1 > 0, img2 > 0
	intersection = np.logical_and(nonzero_1, nonzero_2).sum()
	union = np.logical_or(nonzero_1, nonzero_2).sum()
	return intersection / float(union)

def normalize_depth(depth_img):
    normalized = depth_img.copy()
    positive_idxs = np.where((120 < normalized) & (normalized < 160))
    depth_min = np.amin(normalized[positive_idxs])
    depth_max = np.amax(normalized)
    normalized[positive_idxs] = 255*(normalized[positive_idxs] - float(depth_min))/(depth_max - depth_min)
    return normalized

def maxIOU(img1, img2):
	"""
	Takes in two binary images IMG1 and IMG2 and computes max IOU over
	the masks.
	"""
	assert img1.shape == img2.shape
        img1 = cv2.resize(img1, (160, 120))
        img2 = cv2.resize(img2, (160, 120))
        img1 = inpainted_binary(img1)
        img2 = inpainted_binary(img2)
	nonzero_1, nonzero_2 = (img1 > 0).astype(float), (img2 > 0).astype(float)
	num_nonzero_1, num_nonzero_2 = nonzero_1.sum(), nonzero_2.sum()
	correlated = correlate2d(nonzero_1, nonzero_2)
	return np.divide(correlated, -correlated + nonzero_1.sum() + num_nonzero_2.sum()).max()

def geometric_median(pts, T=40, thresh=1e-5):
	"""
	Iteratively computes the geometric median using Weiszfeld's algorithm.
	"""
	y = np.random.randn(pts.shape[1])
	for t in range(T):
		distances = np.linalg.norm(pts - y, axis=1)
		inv_distances = np.reciprocal(distances)
		ynew = pts.T.dot(inv_distances) / inv_distances.sum()
		if np.linalg.norm(ynew - y) < thresh:
			y = ynew
			break
		y = ynew
	return y
def sample_nearest_points_on_mask(px, img, halfwidth):
	"""
	Returns a list of pixels correspnoding to all pixels in a square
	surrounding px.
	"""
	pixels = []
	# TODO: vectorize this
	for i in range(-halfwidth, halfwidth+1):
		for j in range(-halfwidth, halfwidth+1):
			u = px[0] + i
			v = px[1] + j
			if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
				pixels.append((px[0] + i, px[1] + j))
	return np.array(pixels)

def pixel_to_world(camera_intr, depth, T_camera_world, (x, y), rescaled_pixel=None):
        if rescaled_pixel is not None:
            pixel = np.array(rescaled_pixel)
            x, y = rescaled_pixel
        else:
            pixel = np.array([x, y])
        #print "Depth", (depth[y, x])
        depth_val = depth[y, x]
        if depth[y, x] == 0:
            depth_val = 0.805
	point_cam = camera_intr.deproject_pixel(depth_val, Point(pixel, frame="phoxi"))
	point_world = T_camera_world * point_cam
	if list(point_cam.data) == [0, 0, 0]:
		print "INVALID"
	return list(point_world.data)

def sample_nonzero_points(img, k):
	"""
	Samples k points corresponding to nonzero pixels.
	"""
	pts = []
	while len(pts) < k:
		pt = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
		if img[pt[0], pt[1]].sum() > 0:
			pts.append(pt[::-1])
	return np.array(pts)

def flip_pixels_horizontal(pixels):
    return [[640 - i[0], i[1]] for i in pixels]

def locate_circle_center_hough(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=9,minRadius=35,maxRadius=50)
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=9,minRadius=25,maxRadius=33)
        if circles is not None:
           for i in circles[0,:]:
               print i[2]
               i = [int(i[0]), int(i[1]), int(i[2])]
               pixel_val = img[i[1], i[0]] 
               cond = pixel_val[0] == 0 and pixel_val[1] == 0 and pixel_val[2] == 0
               if not cond:
                   return (i[0], i[1])

def locate_circle(img, diameter, downsample_factor=5):
	"""
	Locates ball at end of rope.
	"""
	if len(img.shape) == 3:
		img = img[:,:,0]
	nonzero = (img > 0).astype(float)
	nonzero = nonzero[::downsample_factor]
	nonzero = nonzero[:,::downsample_factor]
	mask = np.ones((diameter//downsample_factor, diameter//downsample_factor))
	correlated = correlate2d(nonzero, mask, mode='same')
	best = np.array(np.unravel_index(correlated.argmax(), nonzero.shape)) * downsample_factor
	return best

def load_rgb_image(rgb_filename):
	img = Image.open(rgb_filename).convert('RGB').resize((640, 480))
	image = np.array(img)[:, :, ::-1].copy()
	return image

def remove_circle(img, diameter=60, plot=False):
	"""
	Locates and removes the tennis ball.
	"""
	if len(img.shape) == 3:
		img = img[:,:,0]
	img = img.copy()
	center = locate_circle(img, diameter)
	radius = int(diameter//2 * 1.2) # added a small correction here
	img[max(center[0]-radius, 0):min(center[0]+radius, img.shape[0]), max(center[1]-radius, 0):min(center[1]+radius, img.shape[1])] = 0
	if plot:
		cv2.imshow("correspondence", img)
		cv2.waitKey(0)
	return img

def remove_rect(img, center, halfwidth):
	"""
	Masks out a rectangle with width 2 * HALFWIDTH centered at CENTER.
	"""
	img[max(center[0]-halfwidth, 0):min(center[0]+halfwidth, img.shape[0]), max(center[1]-halfwidth, 0):min(center[1]+halfwidth, img.shape[1])] = 0

def sample_sparse_points(img, k=100, dist=50, plot=False):
	"""
	Sparsely samples k points at least dist apart and not on the ball.
	This may not be possible, and the greedy algorithm terminates when no
	more points can be sampled satisfying this distance criterion.
	"""
	def annotate_img(img, u1, v1, val=None):
		color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
		cv2.circle(img, (u1, v1), 4, color, -1)

	original_image = img.copy()

	img = remove_circle(img, diameter=85)
	kernel = np.ones((7,7),np.uint8)
	img = cv2.erode(img,kernel,iterations = 1)

	points = []
	for i in range(k):
		nonzero_indices = np.nonzero(img)
		num_nonzero = len(nonzero_indices[0])
		if num_nonzero == 0:
			break
		idx = np.random.randint(num_nonzero)
		idx = [n[idx] for n in nonzero_indices]
		annotate_img(original_image, idx[1], idx[0])
		remove_rect(img, idx, dist)
                points.append(idx[::-1])
	if plot:
		cv2.imshow("correspondence", original_image)
		cv2.waitKey(0)
        return points


def prune_close_pixel_indices(pixels):
    min_thresh = 10
    neigh = NearestNeighbors(1)
    neigh.fit(pixels)
    pruned = []
    for i in range(len(pixels)):
        dists, idxs = neigh.kneighbors(pixels[i], 2, return_distance=True)
        closest_dist = dists.squeeze()[1]
        if closest_dist < min_thresh:
            pruned.append(i)
    #return pruned
    return pruned

def farthest_pixel_correspondence(source_pixels, target_pixels):
    max_dist = 150
    #max_dist = 140
    farthest_idx = 0
    curr_max_dist = -1
    for i in range(len(source_pixels)):
        dist = np.linalg.norm(np.array(source_pixels[i]) - np.array(target_pixels[i]))
        if dist > max_dist:
            dist = 0
        if dist > curr_max_dist:
            curr_max_dist = dist
            farthest_idx = i
    print "DIST:", curr_max_dist
    paired = list(zip(source_pixels, target_pixels))
    return paired[farthest_idx]
    #return max(paired, key=lambda p: np.linalg.norm(np.array(p[0]) - np.array(p[1])))

        
def downsample_image(image, factor):
	"""
	Applies a Gaussian low-pass filter to IMAGE before downsampling by FACTOR.
	"""
	return resize(image, (image.shape[0] // factor, image.shape[1] // factor),
            anti_aliasing=True)

def resample_image(image, factor):
	"""
	Upsamples IMAGE by FACTOR.
	"""
	#return resize(image, (image.shape[0] * factor, image.shape[1] * factor))
	return resize(image, (480, 640))

if __name__ == '__main__':
        for i in range(5):
            depth_img = cv2.imread('../images/phoxi/segdepth_%d.png'%i)
            norm = normalize_depth(depth_img)
            cv2.imshow("img", norm)
            #cv2.imshow("img", depth_img)
            cv2.waitKey(0)

        #img = cv2.resize(img, (640, 480))
        #u1, v1 = locate_circle_center_hough(img)
	#cv2.circle(img, (u1, v1), 4, (255, 255, 255), -1)
        #cv2.imshow("img", img)
        #cv2.waitKey(0)

       # sample_sparse_points(img, plot=1)
        #img0 = cv2.resize(img0, (320, 240))
        #img1 = cv2.imread("../images_test/phoxi/segdepth_1.png")
        #print IOU(img0, img1)
        #sample_sparse_points(img0, k=100, dist=50, plot=True)
        #sample_sparse_points(img0, k=50, dist=25, plot=True)
