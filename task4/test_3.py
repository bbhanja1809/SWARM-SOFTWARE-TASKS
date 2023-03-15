import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
'''
For Level 3 some adjustments need to be made
like adjust the region of interest
adjust hough line parameters rho theta threshold etc

'''

def ROI(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = (255)*3
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines, color=[0, 0, 255], thickness=3):
    img = np.copy(img)
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    
    pts = np.array([])
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
            pts = np.append(pts,x1)
            pts = np.append(pts,y1)
            pts = np.append(pts,x2)
            pts = np.append(pts,y2)
            
            
    arr = np.array([[pts[2],pts[3]],[pts[0],pts[1]],[pts[4],pts[5]],[pts[6],pts[7]]],np.int32)
    
    arr = arr.reshape((-1,1,2))
    
    cv2.fillPoly(line_img,np.int32([arr]),(0,255,0))
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img
def process(image):
    
    height = image.shape[0]
    width = image.shape[1]
    ROI_vertices = [
        (0, height),
        (width/2 , (0.5*height)),
        
        (width, height),
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
 
    cropped_image = ROI(
        canny_image,
        np.array(
            [ROI_vertices],
            np.int32
        ),
    )
 
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=30,
        maxLineGap=25
    )
 
    x_left = []
    y_left = []
    x_right = []
    y_right = []
 
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5:
                continue
            if slope <= 0:
                x_left.extend([x1, x2])
                y_left.extend([y1, y2])
            else:
                x_right.extend([x1, x2])
                y_right.extend([y1, y2])

    y_min = int(image.shape[0] * (3 / 5))
    y_max = int(image.shape[0])
    poly_left = np.poly1d(np.polyfit(
        y_left,
        x_left,
        deg=1
    ))
 
    left_x_start = int(poly_left(y_max))
    left_x_end = int(poly_left(y_min))
 
    poly_right = np.poly1d(np.polyfit(
        y_right,
        x_right,
       deg=1
    ))
 
    right_x_start = int(poly_right(y_max))
    right_x_end = int(poly_right(y_min))
    image_with_lines = draw_the_lines(
        image,
        [[
            [left_x_start, y_max, left_x_end, y_min],
            [right_x_start, y_max, right_x_end, y_min],
        ]],
        thickness=5,
    )
    return image_with_lines




image = cv2.imread('lane3_pic3.jpg')
height = image.shape[0]
width  = image.shape[1]
ROI_vertices = [
        (0, height),
        (width/2 , (0.5*height)),        
        (width, height)
    ]
image_with_lines = ROI(image,ROI_vertices)
plt.imshow(image_with_lines)
plt.show()
'''
cap = cv2.VideoCapture('Level 3.mp4')
frameTime = 12
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
size = (frame_width,frame_height)
#result = cv2.VideoWriter('Level3.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    #result.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(frameTime) & 0xFF == ord('q'):
        break

cap.release()
#result.release()
cv2.destroyAllWindows()'''