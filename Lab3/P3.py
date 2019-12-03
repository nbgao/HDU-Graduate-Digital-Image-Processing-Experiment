#!/usr/bin/env python
# coding: utf-8

# # Lab3 数字图像分割与边缘检测实验
# **Author: Gao Pengbing (nbgao)**  
# **Email: nbgao@126.com**

# # Work3 采用阈值处理方法进行图像分割
# ## Work3 P1: 参考相关文献，编写程序实现Otsu自动阈值法

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[2]:


def OTSU(I):
    N = np.size(I)
    p = np.zeros(255)
    max_s = 0
    u = 0
    for i in range(255):
        p[i] = np.sum(I==i)/N
        u += p[i]*i
    # 遍历寻找最优二值化阈值
    for k in range(1,254):
        u1 = 0
        for i in range(k):
            u1 += p[i]*i 
        u2 = u - u1
        w1 = np.sum(p[:k])
        w2 = 1 - w1
#         w2 = np.sum(p[t:])
        s = w1*(u-u1)**2 + w2*(u-u2)**2
        if(s>max_s):
            max_s = s
            t = k
    
    print('Binary threshold:', t)
    # 二值化
    I_bw = I.copy()
    I_bw[I_bw<t] = 0
    I_bw[I_bw>=t] = 255
    
    return I_bw, t


# In[3]:


for i in range(1,6):
    file_path = '../Image/image'+str(i)+'.jpg'
    img = plt.imread(file_path)
    img_gray = np.uint8(skimage.color.rgb2gray(img)*255)
    
    img_bw, t = OTSU(img_gray)
    
    plt.figure(figsize=(12,6*img.shape[0]/img.shape[1]))
    plt.subplot(121)
    plt.imshow(img_gray, 'gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img_bw, 'gray')
    plt.title('Binary')
    plt.axis('off')
    plt.show()   


# ## Work3 P2: 实现直方图阈值法，具体方法为采用灰度直方图求双峰或多峰，选择两峰之间的谷底作为阈值，将图像转换为2值图像

# In[4]:


# 直方图双峰二值化
def HistogramThreshold(I):
    I_arr = I.ravel()
    N = np.size(I_arr)
    
    p = np.zeros(256)
    for i in range(256):
        p[i] = np.sum(I_arr==i)

    max_1, max_2, min_t = 0, 0, N
    p1, p2, t = 0, 255, 0
    min_0 = np.size(N)
    l, r = 0, 255
    while(l<r):
        if(p[l]>max_1):
            max_1 = p[l]
            p1 = l
        l += 1
        if(p[r]>max_2):
            max_2 = p[r]
            p2 = r
        r -= 1
            
    for i in range(p1+1,p2):
        if(p[i]<min_t):
            min_t = (p[i])
            t = (i)
    
    # 二值化
    I_bw = I.copy()
    I_bw[I_bw<t] = 0
    I_bw[I_bw>=t] = 255   
    
    print('Binary threshold: {}\tp1:{}\tp2:{}'.format(t, min(p1,p2), max(p1,p2)))
    # 直方图
    plt.figure(figsize=(8,4))
    plt.hist(I_arr, 256, [0, 255])
    plt.vlines(t, min_t, 0, 'r') # 阈值
#     plt.vlines(p1, max_1, 0, 'b') # 波峰１
#     plt.vlines(p2, max_2, 0, 'b') # 波峰２
    plt.show()
    
    return I_bw, t


# In[5]:


for i in range(1,6):
    file_path = '../Image/image'+str(i)+'.jpg'
    img = plt.imread(file_path)
    img_gray = np.uint8(skimage.color.rgb2gray(img)*255)
    
    img_bw, t = HistogramThreshold(img_gray)
    
    plt.figure(figsize=(12,6*img.shape[0]/img.shape[1]))
    plt.subplot(121)
    plt.imshow(img_gray, 'gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(img_bw, 'gray')
    plt.title('Binary')
    plt.axis('off')
    plt.show()   


# In[ ]:




