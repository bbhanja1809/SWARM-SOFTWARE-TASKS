import cv2 as cv
import numpy as np
from numpy.fft import fft2,ifft2
from scipy.signal import convolve2d,wiener



def wiener_filter(image,kernel,K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel,s = image.shape)
    kernel = np.conj(kernel)/(np.abs(kernel)**2 + K)
    dummy = dummy*kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

def average_kernel(kernel_size = 3):
    h = np.ones(kernel_size).reshape(kernel_size,1)
    h = np.dot(h,h.transpose())
    h /= np.sum(h)
    return h





img = cv.imread(r"C:\Users\BIBHUDATTA BHANJA\Desktop\USB\car.jpeg")
img = cv.resize(img,(img.shape[0],img.shape[0]))
kernel_size = 30
cv.imshow("ORIGINAL",img)

vertical_blurr = np.zeros((kernel_size,kernel_size))
horizontal_blurr = np.copy(vertical_blurr)

vertical_blurr[:,int((kernel_size-1)/2)] = np.ones(kernel_size)
horizontal_blurr[int((kernel_size-1)/2),:] = np.ones(kernel_size)

vertical_blurr /= kernel_size
horizontal_blurr /= kernel_size

vertical_mb = cv.filter2D(img,-1,vertical_blurr)
horizontal_mb = cv.filter2D(vertical_mb, -1, horizontal_blurr)
#kernel = average_kernel(5)
filtered_img = wiener(horizontal_mb,(5,5))


cv.imshow("BLURR",horizontal_mb)
cv.imshow("FILTERED",filtered_img)

cv.waitKey(0)


    


