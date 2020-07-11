import numpy as np
import cv2

def vis_webcam(pixels=[], actions=[], bboxes=[], name='test.mp4'):
    cap = cv2.VideoCapture(0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(name, fourcc, 20.0, (width,height))
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        vis = frame.copy()
        for (u,v) in pixels:
            cv2.circle(vis, (u,v), 10, (255,0,0), -1)
        for (pull,hold) in actions:
            cv2.arrowedLine(vis, pull, hold, (0,0,255), 2)
        for (minx,miny,maxx,maxy) in bboxes:
            cv2.rectangle(vis, (minx,miny), (maxx,maxy), (0,255,0), 2)
        out.write(vis)
        cv2.imshow('frame',vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pixels = [(100,200)]
    actions = [((100,200),(200,300))]
    bboxes = [(90,190,210,310)]
    vis_webcam(pixels=pixels, actions=actions, bboxes=bboxes)
