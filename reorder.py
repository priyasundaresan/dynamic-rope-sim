import cv2
import os
import argparse
import json

if __name__ == '__main__':
    # goes from ./{dir}/images and ./{dir}/image_masks to ./{dir}/images and ./{dir}/image_masks ordered
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='.')
    args = parser.parse_args()
    if os.path.exists('./reordered'):
        os.system('rm -r ./reordered')
    os.mkdir('./reordered')
    os.mkdir('./reordered/images')
    os.mkdir('./reordered/image_masks')
    os.mkdir('./reordered/images_depth')

    with open("./{}/images/knots_info.json".format(args.dir), "r") as stream:
        knots_info = json.load(stream)
        print("loaded knots info")

    new_knots_info = {}
    i = 0
    for filename in sorted(os.listdir('./{}/images'.format(args.dir))):
        print("Relabeling " + filename)
        try:
            num = int(filename[:6])
            mask_filename = '%06d_visible_mask.png'%num
            masked_filename = '%06d_mask.png'%num
            save_img_filename = '%06d_rgb.png'%i
            save_mask_filename = '%06d_visible_mask.png'%i
            save_masked_filename = '%06d_mask.png'%i
            new_knots_info[str(i)] = knots_info[str(num)]
            img = cv2.imread('./%s/images/%s'%(args.dir, filename)).copy()
            mask = cv2.imread('./%s/image_masks/%s'%(args.dir, mask_filename)).copy()
            masked = cv2.imread('./%s/image_masks/%s'%(args.dir, masked_filename)).copy()
            depth = cv2.imread('./%s/images_depth/%s'%(args.dir, filename)).copy()
            cv2.imwrite('./reordered/images/{}'.format(save_img_filename), img)
            cv2.imwrite('./reordered/image_masks/{}'.format(save_mask_filename), mask)
            cv2.imwrite('./reordered/image_masks/{}'.format(save_masked_filename), masked)
            cv2.imwrite('./reordered/images_depth/{}'.format(save_img_filename), depth)
            i += 1
        except:
            pass

    # fix knots_info.json for crop
    with open("./reordered/images/knots_info.json", 'w') as outfile:
        json.dump(new_knots_info, outfile, sort_keys=True, indent=2)

    if not args.dir == ".":
        os.system('rm -r ./{}'.format(args.dir))
        os.system('mv reordered {}'.format(args.dir))
    else:
        os.system('rm -r ./images')
        os.system('rm -r ./images_depth')
        os.system('rm -r ./image_masks')
        os.system('mv reordered/images images')
        os.system('mv reordered/images_depth images_depth')
        os.system('mv reordered/image_masks image_masks')
