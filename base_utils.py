import cv2
import math
import numpy as np


def landmarksDetection(img, landmarks):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in landmarks]
    return mesh_coord

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def isOpen(landmarks, *args):

    p_right = landmarks[args[0]]
    p_left = landmarks[args[2]]
    p_top = landmarks[args[3]]
    p_bottom = landmarks[args[1]]

    hDistance = euclaideanDistance(p_right, p_left)
    vDistance = euclaideanDistance(p_top, p_bottom)
    if vDistance == 0:
        return 0
    Ratio = hDistance/vDistance
    return Ratio



def rotateImage(img, angle, scale=1):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(w, h))
    return img

def blackfilter_overlay(image, filter_img, filter_img_mask,  location, blur = False):
    annotated_image = image.copy()
    filter_img_height, filter_img_width, _ = filter_img.shape
    ROI = image[location[1]: location[1] + filter_img_height,
          location[0]: location[0] + filter_img_width]
    resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)
    resultant_image = cv2.add(resultant_image, filter_img)
    # if blur:
    #     resultant_image = cv2.GaussianBlur(resultant_image, (5,5),0)
    annotated_image[location[1]: location[1] + filter_img_height,
    location[0]: location[0] + filter_img_width] = resultant_image
    if blur:
        annotated_image[location[1]: location[1] + int(1.5 * filter_img_height),
        location[0]: location[0] + int(1.5*filter_img_width)] = cv2.GaussianBlur(annotated_image[location[1]: location[1] + int(1.5*filter_img_height),
                                                                                  location[0]: location[0] + int(1.5*filter_img_width)], (7,7), 0)
    return annotated_image

def greenfilter_overlay(image, filter_img, location, u_green, l_green):
    annotated_image = image.copy()
    hsv = cv2.cvtColor(filter_img, cv2.COLOR_BGR2HSV)
    filter_img_mask = cv2.inRange(hsv, u_green, l_green)
    filter_img_height, filter_img_width = filter_img_mask.shape
    ROI = image[location[1]: location[1] + filter_img_height,
          location[0]: location[0] + filter_img_width]
    resultant_image = cv2.bitwise_and(filter_img, filter_img, mask=filter_img_mask)
    f = filter_img - resultant_image
    resultant_image = np.where(f == 0, ROI, f)
    annotated_image[location[1]: location[1] + filter_img_height,
    location[0]: location[0] + filter_img_width] = resultant_image
    # Catch and handle the error(s).
    return annotated_image



