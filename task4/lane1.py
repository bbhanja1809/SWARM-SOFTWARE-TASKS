import cv2
from cv2 import cvtColor
from cv2 import imshow
from cv2 import waitKey
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
import math

def ROI(img, vertices):
    mask = np.zeros_like(img)
    #channel_num = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask,vertices,match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img,lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:        
            cv2.line(blank_image, (x1,y1), (x2,y2), (0,0,255), thickness = 5)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img 

#image = cv2.imread('lane1.jpg')
#image = cvtColor(image,cv2.COLOR_BGR2RGB)
#print(image.shape)
def process(image):
    height = image.shape[0]
    width = image.shape[1]

    vertices_ROI = [
        (0,height),
        (width/2,height/2),
        (height,width)
    ]

    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)

    cropped_image = ROI(canny_image, np.array([vertices_ROI], np.int32))

    lines = cv2.HoughLinesP(cropped_image,rho=6,theta = np.pi/60,threshold=160,lines=np.array([]),minLineLength=20,maxLineGap=100)#changeable

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient = (y2-y1)/(x2-x1)
            if math.fabs(gradient) < 0.5:
                continue
            if gradient <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = int(image.shape[0] * (3 / 5))
    max_y = int(image.shape[0])
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
 
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
 
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
       deg=1
    ))
 
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))



    image_with_lines = draw_the_lines(image,[[[left_x_start,max_y,left_x_end,min_y],[right_x_start,max_y,right_x_end,min_y]]])
    return image_with_lines


cap = cv2.VideoCapture('Level 1.mp4')
frameTime = 18
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(frameTime) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#image_with_lines = process(image)
#plt.imshow(image_with_lines)
#plt.show()