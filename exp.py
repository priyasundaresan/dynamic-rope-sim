import os 

if __name__ == '__main__':
    BLENDER_CMD = '/Users/priyasundaresan/Downloads/blender-2.80.0-git20190620.d30f72dfd8ac-x86_64/blender.app/Contents/MacOS/blender'
    for i in range(2,11):
        os.system('%s -b -P untangle.py' % BLENDER_CMD)
        os.system('cd images && ffmpeg -framerate 200 -i %06d_rgb.png -pix_fmt yuv420p out.mp4 && rm *.png && cd ..')
        os.system('mv ./images/out.mp4 ./preds/')
        os.system('mv ./preds ./exp%d ' % (i))
        os.system('mv ./exp%d ./hierarchical_pretzel_capsule_2/' % (i))
