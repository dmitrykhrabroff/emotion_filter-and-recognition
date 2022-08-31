import cv2
import math
import numpy as np
import base_utils as utils

def head_overlay(image, filter_img, landmarks):
    filter_img_height, filter_img_width, _ = filter_img.shape
    part_top = landmarks[10]
    part_bottom = landmarks[8]
    u_green = np.array([45, 136, 188])
    l_green = np.array([179, 255, 255])
    required_height = part_bottom[1] - part_top[1]
    resized_filter_img = cv2.resize(filter_img, (int(filter_img_width *
                                                     (required_height / filter_img_height)),
                                                 required_height))
    filter_img_height, filter_img_width,_ = resized_filter_img.shape
    center = (part_top[0], int(part_top[1] / 2 + part_bottom[1] / 2))
    location = (int(center[0] - 2*filter_img_width / 5), part_top[1] - filter_img_height//2)
    image = utils.greenfilter_overlay(image, resized_filter_img,  location, u_green, l_green)
    return image

def noses_overlay(image, filter_img, landmarks, green = False):
    filter_img_height, filter_img_width, _ = filter_img.shape
    p_right_nose_top = landmarks[309]
    p_right_nose_bottom = landmarks[290]
    p_left_nose_top = landmarks[79]
    p_left_nose_bottom = landmarks[60]
    p_bottom = landmarks[152]

    required_height = p_bottom[1] - p_left_nose_bottom[1]
    resized_filter_img = cv2.resize(filter_img, (int(filter_img_width *
                                                     (required_height / filter_img_height)),
                                                 required_height))
    required_height = min(p_bottom[1] - p_left_nose_bottom[1],int(image.shape[0] - max(p_left_nose_bottom[1], p_right_nose_bottom[1])))
    resized_filter_img = resized_filter_img[-required_height:,:]

    right_center = (p_right_nose_top[0], int(p_right_nose_top[1] / 2 + p_right_nose_bottom[1] / 2))
    left_center = (p_left_nose_top[0], int(p_left_nose_top[1] / 2 + p_left_nose_bottom[1] / 2))
    filter_img_height, filter_img_width, _ = resized_filter_img.shape
    _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                       25, 255, cv2.THRESH_BINARY_INV)

    right_location = (int(right_center[0] - filter_img_width / 2), int(right_center[1]))
    left_location = (int(left_center[0] - filter_img_width / 2), int(left_center[1]))
    if not green:
        image = utils.blackfilter_overlay(image, resized_filter_img, filter_img_mask,  right_location)
        image = utils.blackfilter_overlay(image, resized_filter_img, filter_img_mask,  left_location)
    else:
        image = utils.greenfilter_overlay(image, resized_filter_img, right_location)
        image = utils.greenfilter_overlay(image, resized_filter_img, left_location)
    return image

def leftIrisOverlay(frame, filter_image, mesh_coords):
    left_center = mesh_coords[468]
    right_center = mesh_coords[473]

    required_height = int((mesh_coords[472][1] - mesh_coords[470][1])*0.8)
    filter_img_height, filter_img_width, _ = filter_image.shape
    resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                     (required_height / filter_img_height)),
                                                 required_height))
    filter_img_height, filter_img_width, _ = resized_filter_img.shape
    left_location = (int(left_center[0] - filter_img_width / 2), int(left_center[1] - filter_img_height / 2))

    _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                        25, 255, cv2.THRESH_BINARY_INV)
    frame = utils.blackfilter_overlay(frame, resized_filter_img, filter_img_mask,  left_location)

    return frame

def rightIrisOverlay(frame, filter_image, mesh_coords):
    right_center = mesh_coords[473]

    required_height = int((mesh_coords[477][1] - mesh_coords[475][1])*0.8)
    filter_img_height, filter_img_width, _ = filter_image.shape
    resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                     (required_height / filter_img_height)),
                                                 required_height))
    filter_img_height, filter_img_width, _ = resized_filter_img.shape
    right_location = (int(right_center[0] - filter_img_width / 2), int(right_center[1] - filter_img_height / 2))
    _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                        25, 255, cv2.THRESH_BINARY_INV)
    frame = utils.blackfilter_overlay(frame, resized_filter_img, filter_img_mask,  right_location)
    return frame

def hatOverlay(frame, filter_image, mesh_coords):
    center = int((mesh_coords[109][0] + mesh_coords[338][0])/2), int((mesh_coords[109][1] + mesh_coords[338][1])/2)
    required_width = mesh_coords[284][0] - mesh_coords[54][0]
    # angle = utils.euclaideanDistance(mesh_coords[9], mesh_coords[109]) / utils.euclaideanDistance(mesh_coords[9], mesh_coords[338])
    # filter_image = utils.rotateImage(filter_image, -angle)
    filter_img_height, filter_img_width, _ = filter_image.shape
    required_height = min(int(filter_img_height * (required_width / filter_img_width)), center[1])
    resized_filter_img = cv2.resize(filter_image, (required_width, int(filter_img_height * (required_width / filter_img_width))))
    resized_filter_img = resized_filter_img[-required_height:, :]

    filter_img_height, filter_img_width, _ = resized_filter_img.shape
    location = (int(center[0] - filter_img_width / 2), int(center[1] - filter_img_height))
    _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                        25, 255, cv2.THRESH_BINARY_INV)
    frame = utils.blackfilter_overlay(frame, resized_filter_img, filter_img_mask,  location)

    return frame

def mouth_overlay(image, filter_img, landmarks, green = False):
    filter_img_height, filter_img_width, _ = filter_img.shape
    p_right = landmarks[308]
    p_left = landmarks[78]
    p_top = landmarks[13]
    p_bottom = landmarks[14]
    hDistance = utils.euclaideanDistance(p_right, p_left)
    vDistance = utils.euclaideanDistance(p_top, p_bottom)+1
    required_height = int(vDistance * 5)
    resized_filter_img = cv2.resize(filter_img, (int(filter_img_width *
                                                     (required_height / filter_img_height)),
                                                 required_height))

    center = (p_bottom[0], int(p_bottom[1] / 2 + p_top[1] / 2))
    required_height = min(int(vDistance * 5), int(image.shape[0] - center[1]))
    resized_filter_img = resized_filter_img[:required_height, :]
    filter_img_height, filter_img_width, _ = resized_filter_img.shape
    _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                       25, 255, cv2.THRESH_BINARY_INV)
    location = (int(center[0] - filter_img_width / 3), int(center[1]))

    if not green:
        image = utils.blackfilter_overlay(image, resized_filter_img, filter_img_mask,  location)
    else:
        image = utils.greenfilter_overlay(image, resized_filter_img, location)

    return image

def facepartExtractor(image, coords,  scale_x =1.2, scale_y=2):
    annotated_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(coords, dtype=np.int32)], 255)
    part_face = cv2.bitwise_and(annotated_image, annotated_image, mask=mask)
    max_x = (max(coords, key=lambda item: item[0]))[0]
    min_x = (min(coords, key=lambda item: item[0]))[0]
    max_y = (max(coords, key=lambda item: item[1]))[1]
    min_y = (min(coords, key=lambda item: item[1]))[1]
    cropped_img = part_face[min_y: max_y, min_x: max_x]
    cropped_img = cv2.resize(cropped_img, None, fx = scale_x, fy = scale_y)
    filter_img_height, filter_img_width, _ = cropped_img.shape
    location = (min_x - int((filter_img_width - filter_img_width/scale_x)/2), min_y - int((filter_img_height - filter_img_height/scale_y)/2))
    # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    _, filter_img_mask = cv2.threshold(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY),
                                       25, 255, cv2.THRESH_BINARY_INV)

    image = utils.blackfilter_overlay(image, cropped_img, filter_img_mask,  location, blur=True)
    # image = cv2.GaussianBlur(image, (7,7), 0)
    return image