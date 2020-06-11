import cv2
import os
import argparse
import json

if __name__ == '__main__':
    # goes from for dir in dirs ./{dir}/images , ./{dir}/image_masks , ./{dir}/images_depth to
    # combined and ordered ./{output_dir}/images , ./{output_dir}/image_masks , ./{output_dir}/images_depth
    # with updated knots_info.json in ./{output_dir}/images
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', type=str, nargs='+', help='an integer for the accumulator')
    parser.add_argument('-o', '--output_dir', type=str, default='combined')
    args = parser.parse_args()
    if os.path.exists('./' + args.output_dir):
        os.system('rm -r ./' + args.output_dir)
    os.mkdir('./' + args.output_dir)
    os.mkdir('./{}/images'.format(args.output_dir))
    os.mkdir('./{}/image_masks'.format(args.output_dir))
    os.mkdir('./{}/images_depth'.format(args.output_dir))

    new_knots_info = {}
    i = 0
    for dir in args.dirs:
        print("Processing", dir)
        os.system('python mask.py --dir ./{}/image_masks'.format(dir))
        with open("./{}/images/knots_info.json".format(dir), "r") as stream:
            knots_info = json.load(stream)
            print("loaded knots info")

        for filename in sorted(os.listdir('./{}/images'.format(dir))):
            print("Relabeling " + filename + " in " + dir)
            try:
                num = int(filename[:6])
                mask_filename = '%06d_visible_mask.png'%num
                masked_filename = '%06d_mask.png'%num
                save_img_filename = '%06d_rgb.png'%i
                save_mask_filename = '%06d_visible_mask.png'%i
                save_masked_filename = '%06d_mask.png'%i
                new_knots_info[str(i)] = knots_info[str(num)]
                img = cv2.imread('./%s/images/%s'%(dir, filename)).copy()
                mask = cv2.imread('./%s/image_masks/%s'%(dir, mask_filename)).copy()
                masked = cv2.imread('./%s/image_masks/%s'%(dir, masked_filename)).copy()
                depth = cv2.imread('./%s/images_depth/%s'%(dir, filename)).copy()
                cv2.imwrite('./{}/images/{}'.format(args.output_dir, save_img_filename), img)
                cv2.imwrite('./{}/image_masks/{}'.format(args.output_dir, save_mask_filename), mask)
                cv2.imwrite('./{}/image_masks/{}'.format(args.output_dir, save_masked_filename), masked)
                cv2.imwrite('./{}/images_depth/{}'.format(args.output_dir, save_img_filename), depth)
                i += 1
            except:
                pass

    # fix knots_info.json for crop
    with open("./{}/images/knots_info.json".format(args.output_dir), 'w') as outfile:
        json.dump(new_knots_info, outfile, sort_keys=True, indent=2)
