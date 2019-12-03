#!/usr/bin/env python
# coding: utf-8

# # Lab3 数字图像分割与边缘检测实验
# **Author: Gao Pengbing (nbgao)**  
# **Email: nbgao@126.com**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# # Work2 Hough线检测
# **Work2 P1: 对作业一中边缘检测的结果，进行Hough线检测**  
# **Work2 P2: 调节参数，提取较长的边界**

# In[4]:


def HoughTransform(I, t):
    [h, w] = I.shape[:2]
    rho_max = int(np.sqrt(h*h+w*w))
    A = np.zeros((2*rho_max, 180))
    for i in range(h):
        for j in range(w):
            if(I[i,j]>0):
                for theta in range(180):
                    r = theta/180*np.pi
                    rho = int(i*np.cos(r) + j*np.sin(r))
                    # Hough
                    rho = rho + rho_max + 1
                    A[rho, theta] = A[rho, theta] + 1
                    
                    
    [rhos, thetas] = np.where(A>t)
    lines_num = len(rhos)
    print('lines_num:', lines_num)
    
    plt.figure(figsize=(8,6))
    plt.imshow(I, 'gray')
    for i in range(lines_num):
        yy = np.arange(h)
        
        r = thetas[i]/180*np.pi
        xx = (rhos[i]-rho_max-yy*np.cos(r))/(np.sin(r))
        if(i==0):
            print(rhos[i]-rho_max)
            break
#         plt.plot(xx, yy, 'b-')
    plt.axis('off')
    plt.show()
    return rhos, thetas


# In[3]:


for i in range(1,6):
    file_path = '../Image/image'+str(i)+'.jpg'
    img = plt.imread(file_path)
    img_gray = np.uint8(skimage.color.rgb2gray(img)*255)
    img_gray = cv2.resize(img_gray, (256, int(256*img_gray.shape[0]/img_gray.shape[1])))
    
    img_edge = cv2.Laplacian(img_gray, -1)
#     img_edge = cv2.Canny(img_gray, 50, 150, apertureSize = 3)
#     img_edge = LaplacianFilter(img_gray, '8')
    minLineLength = 100
    maxLineGap = 8
    lines = cv2.HoughLinesP(img_edge, 1, np.pi/180, 120, minLineLength, maxLineGap)

    print('Lines:', len(lines))
    for x1,y1,x2,y2 in lines[:,0,:]:
        img_hough = cv2.line(img_gray, (x1, y1), (x2, y2), 255, 1)
        
    plt.figure(figsize=(12, 6*img_edge.shape[0]/img_gray.shape[1]))
    plt.subplot(121)
    plt.imshow(img_edge, 'gray')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img_hough, 'gray')
    plt.axis('off')
    plt.show()

