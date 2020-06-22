#!/usr/bin/env python
import os

if __name__=="__main__":
    #entrypoint = 
    cmd = "docker run -it -v /home/jennifer/dynamic-rope-sim:/code jenn-blender"
    #cmd += "--entrypoint=\"%(entrypoint)s\" " % {"entrypoint": entrypoint}
    code = os.system(cmd)
