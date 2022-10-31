# -*- coding: utf-8 -*-
"""
Created on Sat May 15 11:17:55 2021

@author: Shiv
"""

from skimage.io import imread, imshow
from skimage.measure import compare_ssim
import numpy as np
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

org_image1= imread("1.jpg" , as_gray = False)
org_image2= imread("2.jpg" , as_gray = False)
image1= imread("1.jpg" , as_gray = True)
image2= imread("2.jpg" , as_gray = True)
resize1= cv2.resize(image1,(500,400))
resize2= cv2.resize(image2,(500,400))

org_resize1= cv2.resize(org_image1,(500,400))
org_resize2= cv2.resize(org_image2,(500,400))


(score,diff)= compare_ssim(resize1,resize2,full=True)
diff=(diff*225).astype("uint8")

moded= cv2.threshold(diff,0,25,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(moded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(org_resize1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(org_resize2, (x, y), (x + w, y + h), (0, 0, 255), 2)
 

cv2.imshow("Original", org_resize1)
cv2.imshow("Modified", org_resize2)
cv2.waitKey(0)



