import cv2

def HoughTransform(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 #   edges = cv2.Canny(gray,160,200,apertureSize = 3)  #160,200
    edges = cv2.Canny(gray,200,250,apertureSize = 3)  #160,200   for input6

    lines = cv2.HoughLines(edges,1,np.pi/180,300)
    return lines


def getCoodinatesFound(image, boxes, scores):
    det_coords = []
    im_height, im_width = image_np.shape[:2]
    max_boxes_to_draw = boxes[0].shape[0]
    min_score_thresh = .5
    for i in range(min(max_boxes_to_draw, boxes[0].shape[0])):
        if scores is None or scores[0][i] > min_score_thresh:
            class_name = category_index[classes[0][i]]['name']
            position = boxes[0][i]
            (xmin, xmax, ymin, ymax) = (
            position[1] * im_width, position[3] * im_width, position[0] * im_height, position[2] * im_height)
            det_coords.append([int(xmin), int(xmax), int(ymin), int(ymax)])

    s = sorted(det_coords, key=operator.itemgetter(0))
    return s

def get_crosspt(x11,y11, x12,y12, x21,y21, x22,y22):
    if x12==x11 or x22==x21:
        print('delta x=0')
        if x12==x11:
            cx = x12
            m2 = (y22 - y21) / (x22 - x21)
            cy = m2 * (cx - x21) + y21
            return cx, cy
        if x22==x21:
            cx = x22
            m1 = (y12 - y11) / (x12 - x11)
            cy = m1 * (cx - x11) + y11
            return cx, cy

    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    if m1==m2:
        print('parallel')
        return None
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    return cx, cy

def EuclideanD(f1,f2):
    x1 = f1[0]
    y1 = f1[1]
    x2 = f2[0]
    y2 = f2[1]
    dx = x2-x1
    dy = y2-y1
    d_len = math.sqrt((dx*dx)+(dy*dy))
    return d_len