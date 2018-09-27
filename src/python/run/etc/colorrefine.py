from math import atan2

import cv2
import numpy as np

from utils.workdir import cd_work
import matplotlib.pyplot as plt

xshift = 140
yshift = 66
wsize = 30
color_thresh = (71,80,140)
cd_work()
filename = 'resource/Jungle_Gym_gate.png'
filename = 'resource/ext/samples/basement_course1/00180.jpg'
org = cv2.imread(filename)
# org = cv2.resize(org,(int(org.shape[1]/2),int(org.shape[0]/2)))
# cv2.imshow("orig", org)
hsv = cv2.cvtColor(org,cv2.COLOR_BGR2YUV)
cv2.imshow("hsv", hsv)
color = hsv - color_thresh
color = 255-color / np.ptp(color) * 255
print(color.shape)
cv2.imshow("color", color.astype(np.uint8))
# img = cv2.resize(img,(400,400))

crop = color[yshift-wsize:yshift+wsize, xshift-wsize:xshift+wsize]
background = np.mean(hsv[yshift-wsize*2:yshift,xshift-wsize*2:xshift],0)
crop = crop# - background
gray = cv2.cvtColor(crop.astype(np.uint8),cv2.COLOR_BGR2GRAY)

cv2.imshow("gray crop",gray)
# img = cv2.pyrDown(img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, 5)
dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, 5)

dx = cv2.convertScaleAbs(dx)
dy = cv2.convertScaleAbs(dy)
# dx[dx < 0.3*dx.max()] = 0
# dy[dy < 0.3*dy.max()] = 0
dxy = 0.5 * dx + 0.5 * dy
dxy[dxy < 0.3*dxy.max()] = 0

cv2.imshow('dy', dy.astype(np.uint8))
cv2.imshow('dx', dx.astype(np.uint8))
cv2.imshow('dxy', dxy.astype(np.uint8))

hist_x = np.zeros((dxy.shape[1],))

for x in range(0, dxy.shape[1], 2):
    for y in range(dxy.shape[0]):
        hist_x[x] += gray[y, x] + dxy[y, x+1]

hist_y = np.zeros((dxy.shape[0]))
for y in range(0,dxy.shape[0],2):
    for x in range(dxy.shape[1]):
        hist_y[y] += gray[y, x] + dxy[y+1, x]

plt.figure()
plt.title('Histogram')
plt.subplot(1, 2, 1)
plt.hist(hist_x)
plt.subplot(1, 2, 2)
plt.hist(hist_y)
# plt.show(True)
y_max = np.argmax(hist_y)
x_max = np.argmax(hist_x)

org = cv2.circle(org, (x_max+xshift-wsize , y_max+yshift-wsize), 2, (0, 0, 255))
cv2.imshow('final', org.astype(np.uint8))

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
