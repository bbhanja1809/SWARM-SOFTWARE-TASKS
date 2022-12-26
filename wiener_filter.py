import numpy as np
import cv2 as cv
from scipy.signal import convolve2d,gaussian
from numpy.fft import fft2,ifft2

def kernel_matrix(kernel_size):
    m = gaussian(kernel_size,kernel_size/3).reshape(kernel_size,1)
    m = np.dot(m,m.transpose())
    m /= np.sum(m)
    return m

def wiener_filter(img,kernel,K):    
	dummy = np.copy(img)
    kernel = np.pad(kernel, [(0, dummy.shape[0] - kernel.shape[0]), (0, dummy.shape[1] - kernel.shape[1])], 'constant')
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    filtered = wiener_filter(frame,kernel_matrix(5),K=10)
    cv.imshow('frame', filtered)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
