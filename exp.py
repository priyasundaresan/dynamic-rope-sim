import os 

if __name__ == '__main__':
    BLENDER_CMD = '/Users/priyasundaresan/Downloads/blender-2.80.0-git20190620.d30f72dfd8ac-x86_64/blender.app/Contents/MacOS/blender'
    for i in range(25,26):
        os.system('%s -b -P untangle.py' % BLENDER_CMD)
        #os.system('cd images && ffmpeg -framerate 200 -i %06d_rgb.png -pix_fmt yuv420p out.mp4 && rm *.png && cd ..')
        os.system('cd images && python ~/Desktop/blender/dynamic-rope/relabel_png.py && ffmpeg -framerate 70 -i %06d_rgb.png -pix_fmt yuv420p out.mp4')
        print("IN", os.getcwd())
        os.system('cd images && rm *.png && cd ..')
        os.system('mv ./images/out.mp4 ./preds/')
        os.system('mv ./preds ./exp%d ' % (i))
        os.system('mv ./exp%d ./gauss_capsule_2knot/' % (i))
