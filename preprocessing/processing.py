import cv2
import numpy as np
from skimage import util

def canny(img):
  kernel = np.ones((2,2),np.uint8)
  canny_img = cv2.Canny(img,100,200)
  dilated_img = cv2.dilate(canny_img,kernel,iterations=1)
  erode_img = cv2.erode(dilated_img,kernel,iterations=1)
  cv2.imshow('canny',erode_img)

def laplacian(img):
  laplacian = cv2.Laplacian(img,cv2.CV_64F)
  inv = util.invert(laplacian)
  cv2.imshow('inverse',inv)


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


if __name__ == '__main__':
  dir_path = 'E://Documents//GitHub//Express-U//data//alpha//K//4.jpg'

  img = cv2.imread(dir_path,cv2.IMREAD_GRAYSCALE)
  brightened_img = cv2.resize(BrightnessContrast(img),(200,200))
  cv2.imshow('bright',brightened_img)
  resized_img = cv2.resize(brightened_img,(200,200))
  blur_img = cv2.GaussianBlur(resized_img,(3,3),0)

  cv2.imshow('img',blur_img)

  canny(blur_img)
  laplacian(blur_img)

  cv2.waitKey(0)
