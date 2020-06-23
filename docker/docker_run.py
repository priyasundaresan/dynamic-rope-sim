#!/usr/bin/env python
import os

if __name__=="__main__":
    cmd = "docker run -it -v /home/jennifer/Desktop/dynamic-rope-sim:/code jenn-blender"
    code = os.system(cmd)
