import warnings

from function import HoughTransform, getCoodinatesFound, get_crosspt, EuclideanD

warnings.filterwarnings('ignore')

import numpy as np
import os
import sys
import tensorflow as tf
import math

import cv2
import operator

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'indoor_training_dir'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'door_label_map.pbtxt')
NUM_CLASSES = 90

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'door_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


s = input("Input the input image number: ")

fname = "indoor_input/experiment/" + s + ".jpeg"
print("Input image: " + fname + '\n')
image_np = cv2.imread(fname, cv2.IMREAD_COLOR)

lines = HoughTransform(image_np)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 7
color = (255, 0, 0)
thickness = 2

# Object Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        # coordinates objects found: [xmin, xmax, ymin, ymax]
        coords_found = []
        coords_found = getCoodinatesFound(image_np, boxes, scores)
        print("coordinates found:  " + str(coords_found) + '\n')

        lines_coords = []  # d,x1,x2,y1,y2,degree

        for i in range(0, len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                if y1 < 0:
                    x1 = int(-1 * y1 * (x2 - x1) / (y2 - y1) + x1)
                    y1 = 0

                if x1 < 0:
                    y1 = int(y1 - x1 * (y2 - y1) / (x2 - x1))
                    x1 = 0

                dx = x2 - x1
                dy = y2 - y1
                d_len = math.sqrt((dx * dx) + (dy * dy))
                if y1 > y2:
                    degree = abs((np.arctan2(y1 - y2, x2 - x1) * 180) / np.pi)
                elif y1 < y2:
                    degree = 180 - abs((np.arctan2(y2 - y1, x2 - x1) * 180) / np.pi)
                if 10 < degree < 80 or 100 < degree < 170:
                    lines_coords.append([d_len, x1, x2, y1, y2, degree])

        sorted_line = sorted(lines_coords, key=operator.itemgetter(0))
        print("sorted line: " + str(sorted_line) + '\n')

        longest_1 = sorted_line[len(sorted_line) - 1]
        longest_degree = longest_1[5]
        print('\n' + "longest degree: " + str(longest_degree) + '\n')

        c_x = coords_found[0][0]
        c_y = (longest_1[4] - longest_1[3]) / (longest_1[2] - longest_1[1]) * (c_x - longest_1[1]) + longest_1[3]
        c_avg = (coords_found[0][2] + coords_found[0][3]) / 2
        if c_y > c_avg:
            longest_1.append("below")
            if longest_1[5] > 90:
                longest_1.append("right")
            elif longest_1[5] < 90:
                longest_1.append("left")

        elif c_y < c_avg:
            longest_1.append("above")
            if longest_1[5] < 90:
                longest_1.append("right")
            elif longest_1[5] > 90:
                longest_1.append("left")

        flag = 0
        len_line = len(sorted_line)
        while 1:
            if flag == len_line:
                break

            if abs(longest_degree - sorted_line[flag][5]) < 10:
                del sorted_line[flag]
                flag = flag - 1
                len_line = len_line - 1

            if longest_1[7] == "left":
                if longest_1[6] == "above":
                    if sorted_line[flag][5] > 90:
                        del sorted_line[flag]
                        flag = flag - 1
                        len_line = len_line - 1
                elif longest_1[6] == "below":
                    if sorted_line[flag][5] < 90:
                        del sorted_line[flag]
                        flag = flag - 1
                        len_line = len_line - 1

            elif longest_1[7] == "right":
                if longest_1[6] == "above":
                    if sorted_line[flag][5] < 90:
                        del sorted_line[flag]
                        flag = flag - 1
                        len_line = len_line - 1
                elif longest_1[6] == "below":
                    if sorted_line[flag][5] > 90:
                        del sorted_line[flag]
                        flag = flag - 1
                        len_line = len_line - 1

            flag = flag + 1

        longest_2 = sorted_line[len(sorted_line) - 1]
        print("longest line1: " + str(longest_1))
        print("longest line2: " + str(longest_2) + '\n')

        vanishing_pt = get_crosspt(longest_1[1], longest_1[3], longest_1[2], longest_1[4],
                                   longest_2[1], longest_2[3], longest_2[2], longest_2[4])
        print("Vanishing Point: " + str(vanishing_pt) + '\n')

        y_1 = ((longest_1[4] - longest_1[3]) / (longest_1[2] - longest_1[1])) * (-1) * longest_1[1] + longest_1[3]
        y_2 = ((longest_2[4] - longest_2[3]) / (longest_2[2] - longest_2[1])) * (-1) * longest_2[1] + longest_2[3]

        below_line = []
        above_line = []

        if y_1 > y_2:
            above_line = longest_2
            below_line = longest_1
        elif y_1 < y_2:
            above_line = longest_1
            below_line = longest_2

        print("above line: " + str(above_line))
        print("below line: " + str(below_line) + '\n')

        factor_2D = []
        for i in range(0, len(coords_found)):
            x1 = coords_found[i][0]
            x2 = coords_found[i][1]
            y1 = (below_line[4] - below_line[3]) / (below_line[2] - below_line[1]) * (x1 - below_line[1]) + below_line[
                3]
            y2 = (below_line[4] - below_line[3]) / (below_line[2] - below_line[1]) * (x2 - below_line[1]) + below_line[
                3]
            factor_2D.append([x1, y1])
            factor_2D.append([x2, y2])

        print("2d factor: " + str(factor_2D) + '\n\n')

        print("-----Real Distance Ratio--------\n")

        if longest_1[7] == "left":
            real_ratio = [1]
            for i in range(0, int(len(factor_2D)) - 2):
                next_ratio = 1 / (
                            ((EuclideanD(factor_2D[i], factor_2D[i + 2]) * EuclideanD(factor_2D[i + 1], vanishing_pt)) /
                             (EuclideanD(factor_2D[i + 1], factor_2D[i + 2]) * EuclideanD(factor_2D[i],
                                                                                          vanishing_pt))) - 1) * \
                             real_ratio[i]
                real_ratio.append(abs(next_ratio))
            real_ratio.append('left')

        elif longest_1[7] == "right":
            factor_2D.reverse()
            real_ratio = [1]
            for i in range(0, int(len(factor_2D)) - 2):
                next_ratio = 1 / (
                            ((EuclideanD(factor_2D[i], factor_2D[i + 2]) * EuclideanD(factor_2D[i + 1], vanishing_pt)) /
                             (EuclideanD(factor_2D[i + 1], factor_2D[i + 2]) * EuclideanD(factor_2D[i],
                                                                                          vanishing_pt))) - 1) * \
                             real_ratio[i]
                real_ratio.append(abs(next_ratio))
            real_ratio.append('right')

        print("ratio: " + str(real_ratio) + '\n')

        print("--------------------------------\n")

        for i in range(0, len(factor_2D)):
            cv2.putText(image_np, '2Dfactor' + str(i), (int(factor_2D[i][0]), int(factor_2D[i][1])), font,
                        4, color, thickness, cv2.LINE_AA)
            cv2.line(image_np, (int(factor_2D[i][0]), int(factor_2D[i][1])),
                     (int(factor_2D[i][0]), int(factor_2D[i][1])), color, 30)

        cv2.putText(image_np, 'Point', (int(vanishing_pt[0]), int(vanishing_pt[1])), font,
                    fontScale, color, thickness, cv2.LINE_AA)
        cv2.line(image_np, (int(vanishing_pt[0]), int(vanishing_pt[1])),
                 (int(vanishing_pt[0]), int(vanishing_pt[1])), color, 30)

        cv2.line(image_np, (longest_1[1], longest_1[3]), (longest_1[2], longest_1[4]), (255, 0, 0),
                 8)  ##detection box color: (144,238,144)
        cv2.line(image_np, (longest_2[1], longest_2[3]), (longest_2[2], longest_2[4]), (255, 0, 0), 8)

##sketch
sketch = cv2.imread("indoor_input/sketch.png", cv2.IMREAD_COLOR)
sk_height, sk_width = sketch.shape[:2]

side = real_ratio[len(real_ratio) - 1]

ratio_sum = 0
for i in range(0, len(real_ratio) - 1):
    ratio_sum = ratio_sum + real_ratio[i]

print("\n---------- blueprint ------------\n")

if real_ratio[len(real_ratio) - 1] == "left":

    sketch_ratio = []
    for i in range(0, len(real_ratio) - 1):
        sketch_ratio.append(sk_height * (real_ratio[i] / ratio_sum))

    h_flag = 0
    sketch_ratio.reverse()
    a = 0
    b = 0
    for i in range(0, len(sketch_ratio)):
        if i % 2 == 0:
            b = b + int(sketch_ratio[i])
            print("door(left): " + str(a))
            print("door(right): " + str(b) + '\n')
            cv2.line(sketch, (0, a), (0, b), (255, 0, 0), 5)
            a = b

        else:
            a = a + int(sketch_ratio[i])
            b = b + int(sketch_ratio[i])


elif real_ratio[len(real_ratio) - 1] == "right":
    sketch_ratio = []
    for i in range(0, len(real_ratio) - 1):
        sketch_ratio.append(sk_height * (real_ratio[i] / ratio_sum))

    h_flag = 0
    sketch_ratio.reverse()
    a = 0
    b = 0
    for i in range(0, len(sketch_ratio)):
        if i % 2 == 0:
            b = b + int(sketch_ratio[i])
            print("door(left): " + str(a))
            print("door(right): " + str(b) + '\n')
            cv2.line(sketch, (sk_width, a), (sk_width, b), (255, 0, 0), 5)
            a = b

        else:
            a = a + int(sketch_ratio[i])
            b = b + int(sketch_ratio[i])

print("--------------------------------\n")

blueprint_write = 'indoor_output/experiment/blp' + s + '.jpg'
result_write = 'indoor_output/experiment/rst' + s + '.jpg'

print("Done")
cv2.imwrite(blueprint_write, sketch)
cv2.imwrite(result_write, image_np)