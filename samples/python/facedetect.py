#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
#from video import create_capture
from common import clock, draw_str

def brighten_region(image, rect, factor=1.5):
    (x, y, w, h) = (rect[0], rect[1], rect[2], rect[3])
    #(x, y, w, h) = (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
    face_region = image[y:h, x:w]
    #face_region = image[y:y+h, x:x+w]
    hsv_face = cv.cvtColor(face_region, cv.COLOR_BGR2HSV)
    
    # 明るさを調整
    hsv_face[..., 2] = np.clip(hsv_face[..., 2] * factor, 0, 255)
    
    bright_face_region = cv.cvtColor(hsv_face, cv.COLOR_HSV2BGR)
    image[y:h, x:w] = bright_face_region
    #image[y:y+h, x:x+w] = bright_face_region

def detect(gray, cascade):
    rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

#def draw_rects(img, rects, color):
#    for x1, y1, x2, y2 in rects:
#        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def main():
    import sys, getopt
    image_path = '../../../sample.jpg'
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    #try:
    #    video_src = video_src[0]
    #except:
    #    video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    #cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('lena.jpg')))
    img = cv.imread(image_path)
    if img is None:
        print("画像が読み込まれませんでした。")
        return

    #while True:
        #_ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

        #t = clock()
    rects = detect(gray, cascade)
    vis = img.copy()
    #draw_rects(vis, rects, (0, 255, 0))
    if not nested.empty():
        for x1, y1, x2, y2 in rects:
            #roi = gray[y1:y2, x1:x2]
            #vis_roi = vis[y1:y2, x1:x2]
            #subrects = detect(roi.copy(), nested)
            #draw_rects(vis_roi, subrects, (255, 0, 0))
            brighten_region(vis, (x1,y1, x2, y2))
            #dt = clock() - t

            #draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
    cv.imshow('Brightened Profile Faces', vis)
    cv.imwrite('../../../sample.jpg', vis)
    cv.waitKey(0)

    #if cv.waitKey(5) == 27:
    #    break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
