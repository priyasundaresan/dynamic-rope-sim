import os
import argparse

# python copy_network --network_dir rope_400_cyl_16

parser = argparse.ArgumentParser()
parser.add_argument('--network_dir', type=str, help='path to dir with trained network')
parser.add_argument('--iter', default='003501', type=str, help='iteration pth to copy over')

if __name__ == '__main__':
    args = parser.parse_args()
    os.system('mkdir %s' % args.network_dir)
    os.system('rsync -av priya@jensen.ist.berkeley.edu:/raid/priya/data/pdc_synthetic_2/trained_models/tutorials/%s/*.yaml ./%s/' % (args.network_dir, args.network_dir))
    os.system('rsync -av priya@jensen.ist.berkeley.edu:/raid/priya/data/pdc_synthetic_2/trained_models/tutorials/%s/%s* ./%s/' % (args.network_dir, args.iter, args.network_dir))
