import cv2 as cv
import numpy as np
import os

def write_tuples_to_file(array_of_tuples, subdirectory, file_name):
    # See if there is a better way to do this
    cwd = os.getcwd()
    file_path = os.path.join(subdirectory, file_name)
    with open(cwd + file_path, 'w') as file:
        for tpl in array_of_tuples:
            line = ' '.join(map(str, tpl)) + '\n'
            file.write(line)

def estimate_motion(video_path):
    cap = cv.VideoCapture(video_path)

    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, prev_frame = cap.read()
    old_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_frame)
    while(1):
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv.destroyAllWindows()


video_path = 'labeled/0.hevc'
estimate_motion(video_path=video_path)
# write_tuples_to_file([(1,2),(3,4),(5,6)], "/labeled-predictions", "0.txt")


