import os
import numpy as np
import cv2 
import string
#E:\Documents\GitHub\Express-U\data
from skimage import util
globalfile = 'E:/Documents/Github/Express-U/data/'
MainDirectory = 'E:/Documents/Github/Express-U/data/Num'
a = list(string.ascii_uppercase)

# def preprocessing(path):
#         img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
#         kernel = np.ones((2,2),np.uint8)
#         resized_img = cv2.resize(img,(200,200))
        
#         dilated_img = cv2.dilate( resized_img,kernel,iterations=1)
#         erode_img = cv2.erode(dilated_img,kernel,iterations=1)
#         blur_img = cv2.GaussianBlur(erode_img,(3,3),0)
#         return blur_img


def canny(img):
  kernel = np.ones((2,2),np.uint8)
  canny_img = cv2.Canny(img,100,200)
  dilated_img = cv2.dilate(canny_img,kernel,iterations=1)
  erode_img = cv2.erode(dilated_img,kernel,iterations=1)
  return erode_img
#   cv2.imshow('canny',erode_img)

def laplacian(img):
  laplacian = cv2.Laplacian(img,cv2.CV_64F).astype(np.uint8)

  inv = util.invert(laplacian)
  return inv

def sobel(img):
  sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,5)
  return sobelxy

def BrightnessContrast(img, brightness=319,contrast=164):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
        al_pha = (max - shadow) / 255
        ga_mma = shadow
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    return cal

for i in range(1,10):
    dirpath = f'{globalfile}Num/{i}'
    No_of_files = len(os.listdir(dirpath))
    # os.mkdir(f'{globalfile}preprocessed/canny/{i}')
    # sortedNumCanny = f'{globalfile}preprocessed/canny/{i}'
    os.mkdir(f'{globalfile}preprocessed/laplacian/{i}')
    sortedNumLaplacian = f'{globalfile}preprocessed/laplacian/{i}'
    for j in range(No_of_files):
        filepath = f'{globalfile}Num/{i}/{j}.jpg'      
        img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        brightened_img = cv2.resize(BrightnessContrast(img),(256,256))
        # cv2.imshow('bright',brightened_img)
        # resized_img = cv2.resize(brightened_img,(200,200))
        blur_img = cv2.GaussianBlur(brightened_img,(3,3),0)

        # cv2.imshow('img',blur_img)

        # img=canny(blur_img)
        img1=laplacian(blur_img)
        # path1 = sortedNumCanny
        path2 = sortedNumLaplacian
        # cv2.imwrite(os.path.join(path1 , f'sign({i}){j}.jpg'), img)
        cv2.imwrite(os.path.join(path2 , f'sign({i}){j}.jpg'), img1)
        

        cv2.waitKey(0)
    

       
for i in range(0,26):
    dirpath = f'{globalfile}Alpha/{a[i]}'
    No_of_files = len(os.listdir(dirpath))
    # os.mkdir(f'{globalfile}preprocessed/canny/{a[i]}')
    # sortedcanny = f'{globalfile}preprocessed/canny/{a[i]}'
    os.mkdir(f'{globalfile}preprocessed/laplacian/{a[i]}')
    sortedlaplacian = f'{globalfile}preprocessed/laplacian/{a[i]}'
    for j in range(No_of_files):
        filepath = f'{globalfile}Alpha/{a[i]}/{j}.jpg'      
        img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        brightened_img = cv2.resize(BrightnessContrast(img),(200,200))
        # cv2.imshow('bright',brightened_img)
        #resized_img = cv2.resize(brightened_img,(200,200))
        blur_img = cv2.GaussianBlur(brightened_img,(3,3),0)

        # cv2.imshow('img',blur_img)

        # img=canny(blur_img)
        img1=laplacian(blur_img)
        # path1 = sortedcanny
        path2= sortedlaplacian
        # cv2.imwrite(os.path.join(path1 , f'sign({a[i]}){j}.jpg'), img)
        
        cv2.imwrite(os.path.join(path2 , f'sign({a[i]}){j}.jpg'), img1)

        cv2.waitKey(0)

       
        
        
