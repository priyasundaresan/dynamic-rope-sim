#!/usr/bin/env python3

import os

if __name__ == '__main__':
    num_trials = 15

    start_index = 1
    
    BLENDER_CMD = '/Applications/Blender.app/Contents/MacOS/blender'
    for i in range(start_index,start_index+num_trials):
    # for i in range(4, 16):
        os.system('%s -b -P untangle.py' % BLENDER_CMD)
        os.system('cd images && ffmpeg -framerate 200 -i %06d_rgb.png -pix_fmt yuv420p out.mp4 && rm *.png && cd ..')
        os.system('mv ./images/out.mp4 ./preds/')
        os.system('mv ./preds ./exp%d ' % (i))
        os.system('mv ./exp%d ./cap_heur_pretzel/' % (i))

    #for i in range(1,num_trials+1):
    # for i in range(4, 16):
    #   os.system('%s -b -P untangle_descriptors.py --fig8' % BLENDER_CMD)
    #   os.system('cd images && ffmpeg -framerate 200 -i %06d_rgb.png -pix_fmt yuv420p out.mp4 && rm *.png && cd ..')
    #   os.system('mv ./images/out.mp4 ./preds/')
    #   os.system('mv ./preds ./exp%d ' % (i))
    #   os.system('mv ./exp%d ./test_fig8/' % (i))
        # os.system('mv ./exp%d ./test/' % (i))
