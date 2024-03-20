import cv2
import numpy as np

cap = cv2.VideoCapture('gleb_move.mp4')

# Util functions

def dist(a, b):
    return np.sqrt(np.power(b[0] - a[0], 2) + np.power(b[1] - a[1], 2))

def get_angle(v1, v2):
    dx = v2[0] - v1[0]
    dy = v2[1] - v2[1]
    return np.arctan2(dy, dx) * 180 / np.pi

def norm_dist(v1, v2, sig=15):
    theta = get_angle(v1, v2)

    x = v1[0] + sig * np.cos(theta)
    y = v1[1] + sig * np.sin(theta)

    #print 'check', dist((x, y), v1)
    return v1, (int(x), int(y))


# Lucas Kanade optical flow
def optical_flow(old_gray, frame_gray, p0):
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if st is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        return good_new, good_old
    else:
        return None, None


# Displays output for each frame
def draw_status(frame, mean_angle, kps):
    c = (161,242,108)
    h = 230

    cv2.putText(frame, 'mean angle: ' + str(int(mean_angle)), (20, 20 + h), cv2.FONT_HERSHEY_TRIPLEX, 0.5, c)
    cv2.putText(frame, 'keypoints:   ' + str(kps), (20, 40 + h), cv2.FONT_HERSHEY_TRIPLEX, 0.5, c)

    if 90 > mean_angle > 0:
        motion = 'forward'
        dirc = 'right'
    elif 180 > mean_angle > 90:
        motion = 'forward'
        dirc = 'left'
    elif 270 > mean_angle > 180:
        motion = 'backward'
        dirc = 'left'
    elif 360 > mean_angle > 270:
        motion = 'backward'
        dirc = 'right'
    else:
        motion = 'stopped'
        dirc = 'n/a'

    cv2.putText(frame, 'motion:      ' + motion, (20, 60 + h), cv2.FONT_HERSHEY_TRIPLEX, 0.5, c)
    cv2.putText(frame, 'direction:    ' + dirc, (20, 80 + h), cv2.FONT_HERSHEY_TRIPLEX, 0.5, c)

    if 60 > mean_angle > 0:
        theta = 45
    elif 120 > mean_angle > 60:
        theta = 90
    elif 180 > mean_angle > 120:
        theta = 135
    else:
        theta = mean_angle

    bkp = (20, 0 + h)
    ax = int(bkp[0] + 15 * np.cos(-theta))
    ay = int(bkp[1] + 15 * np.sin(-theta))
    cv2.arrowedLine(frame, bkp, (int(ax), int(ay)), c, 2, tipLength=0.5)
    cv2.circle(frame,bkp, 4, c,-1)


    return frame


# Initialize the First Frame

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
feature_detector = cv2.ORB_create(400)


def get_features(gray):
    kp, descs = feature_detector.detectAndCompute(gray, None)
    return np.array([np.array([k.pt]).astype(np.float32,) for k in kp])

p0 = get_features(old_gray)

mask = np.zeros_like(old_frame)
color = np.random.randint(0,255,(500,3))
frame_count = 0
current_angle = 0

while(1):
    frame_count += 1
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    angles = [0]
    kps = 0
    good_new, good_old = optical_flow(old_gray, frame_gray, p0)

    if good_new is not None:
        min_distance = 0.8

        # Collect angle between keypoint points and draw vectors to frame
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            if dist((a,b), (c, d)) > min_distance:
                kps += 1
                angles.append(get_angle((a,b), (c,d)))
                xx, yy = norm_dist((a,b), (c,d))
                xx, yy = norm_dist((a,b), (c,d))
                cv2.arrowedLine(frame, (int(xx[0]), int(xx[1])), (int(yy[0]), int(yy[1])), color[i].tolist(), 2, tipLength=0.5)
                cv2.circle(frame,(int(a),int(b)), 4, color[i].tolist(),-1)

    # Calculate the mean angle every n frames
    if frame_count % 1 == 0:
        current_angle = np.mean(angles)

    frame = draw_status(frame, current_angle, kps)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Update Frame and Keypoints
    old_gray = frame_gray.copy()
    if good_new is not None:
        p0 = good_new.reshape(-1,1,2)
    else:
        p0 = get_features(old_gray)

cv2.destroyAllWindows()
cap.release()