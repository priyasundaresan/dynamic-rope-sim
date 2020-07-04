import os

def merge(path_to_datasets, output_dir_name):
    if not os.path.exists(output_dir_name):
        os.mkdir(output_dir_name)
    image_output = os.path.join(output_dir_name, 'images')
    keypoints_output = os.path.join(output_dir_name, 'keypoints')
    if not os.path.exists(image_output):
        os.mkdir(image_output)
    if not os.path.exists(keypoints_output):
        os.mkdir(keypoints_output)
    ctr = 0
    for d in os.listdir(path_to_datasets):
        image_dir = os.path.join(path_to_datasets, d, 'images')
        keypoints_dir = os.path.join(path_to_datasets, d, 'keypoints')
        for img_fn, kpt_fn in zip(sorted(os.listdir(image_dir)), sorted(os.listdir(keypoints_dir))):
            path_to_img = os.path.join(image_dir, img_fn)
            path_to_kpts = os.path.join(keypoints_dir, kpt_fn)
            img_cmd = 'cp %s %s'%(path_to_img, image_output+'/'+'%05d.jpg'%ctr)
            kpts_cmd = 'cp %s %s'%(path_to_kpts, keypoints_output+'/'+'%05d.npy'%ctr)
            ctr += 1
            os.system(img_cmd)
            os.system(kpts_cmd)

if __name__ == '__main__':
    path_to_datasets = 'kpts_datasets'
    output_name = 'test'
    merge(path_to_datasets, output_name)
    
