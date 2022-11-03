import cv2
import numpy as np


# trackbar callback fucntion to update HSV value
def callback(x):
    global H_low, H_high, S_low, S_high, V_low, V_high
    # assign trackbar position value to H,S,V High and low variable
    H_low = cv2.getTrackbarPos('low H', 'controls')
    H_high = cv2.getTrackbarPos('high H', 'controls')
    S_low = cv2.getTrackbarPos('low S', 'controls')
    S_high = cv2.getTrackbarPos('high S', 'controls')
    V_low = cv2.getTrackbarPos('low V', 'controls')
    V_high = cv2.getTrackbarPos('high V', 'controls')


# create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls', 2)
cv2.resizeWindow("controls", 550, 10);

# global variable
H_low = 0
H_high = 179
S_low = 0
S_high = 255
V_low = 0
V_high = 255

# create trackbars for high,low H,S,V
cv2.createTrackbar('low H', 'controls', 0, 255, callback)
cv2.createTrackbar('high H', 'controls', 255, 255, callback)

cv2.createTrackbar('low S', 'controls', 0, 255, callback)
cv2.createTrackbar('high S', 'controls', 255, 255, callback)

cv2.createTrackbar('low V', 'controls', 0, 255, callback)
cv2.createTrackbar('high V', 'controls', 255, 255, callback)
mouth_animation = cv2.VideoCapture('media/fire.mp4')
head_animation = cv2.VideoCapture('media/monkey.mp4')
evil_horn = cv2.imread('media/evil_horn2.png')  # , cv2.IMREAD_UNCHANGED)
cap = cv2.VideoCapture('media/demo.mp4')
nose_animation = cv2.VideoCapture('media/smoke4.mp4')
mouth_animation2 = cv2.VideoCapture('media\cigarette.mp4')
head_animation2 = cv2.VideoCapture('media\SlimDiligentCoral-mobile.mp4')
filter_counter = 0
nose_color = [np.array([0, 0, 200], np.uint8), np.array([255, 255, 255], dtype=np.uint8)]
mouth_color = [np.array([43, 42, 148], np.uint8), np.array([80, 255, 255], dtype=np.uint8)]
bomb_color = [np.array([43, 42, 103], np.uint8), np.array([84, 255, 255], dtype=np.uint8)]
spider_color = [np.array([0, 0, 103], np.uint8), np.array([255, 255, 255], dtype=np.uint8)]
smoke = [np.array([50, 0, 47], np.uint8), np.array([255, 255, 255], dtype=np.uint8)]
fire_iris = cv2.VideoCapture('media/fire_iris.mp4')
smoke = cv2.VideoCapture('media/RelievedConsciousIcterinewarbler-mobile.mp4')
bomb = cv2.VideoCapture('media/5555.mp4')
spider = cv2.VideoCapture('media/spider.mp4')
eyes = cv2.VideoCapture('media\crying green screen effect.mp4'), [np.array([32, 93, 158], np.uint8), np.array([87, 204, 229], dtype=np.uint8)]
u_color = {'black': np.array([0, 5, 50], np.uint8), 'white': np.array([0, 0, 240], dtype=np.uint8), 'green' : np.array([0, 0, 174], np.uint8)}
l_color = {'black': np.array([179, 50, 255], np.uint8), 'white': np.array([255, 15, 255], dtype=np.uint8), 'green': np.array([255, 255, 255], dtype=np.uint8)}
while (1):
    # read source image
    ret_filter, filter_frame = head_animation2.read()
    # ret_filter2, filter_frame2 = nose_animation.read()
    # filter_frame = cv2.cvtColor(filter_frame, cv2.COLOR_RGB2BGR)
    filter_counter += 1
    if filter_counter == head_animation2.get(cv2.CAP_PROP_FRAME_COUNT)//2:
      head_animation2.set(cv2.CAP_PROP_POS_FRAMES, 0)
      filter_counter = 0
    # convert sourece image to HSC color mode
    hsv = cv2.cvtColor(filter_frame, cv2.COLOR_BGR2HSV)
    #

    hsv_low = np.array([H_low, S_low, V_low], np.uint8)
    hsv_high = np.array([H_high, S_high, V_high], np.uint8)
    #
    # hsv_low = np.array([58,34,0], np.uint8)
    # hsv_high = np.array([154,255,255], np.uint8)
    # making mask for hsv range
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    # mask = cv2.bitwise_not(mask)
    # masking HSV value selected color becomes black
    mask = cv2.bitwise_not(mask)

    res = cv2.bitwise_and(filter_frame, filter_frame, mask=mask)

    print((H_low, S_low, V_low), H_high, S_high, V_high)
    # show image
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    # waitfor the user to press escape and break the while loop
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break

# destroys all window
cv2.destroyAllWindows()