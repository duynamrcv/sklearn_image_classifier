import cv2
import numpy as np
import glob
import image_dehazer
# img = cv2.imread('training_dataset/blue/draw_453.jpg')  # mandrill reference image from USC SIPI


def apply_brightness_contrast(input_img, brightness = 0, contrast = 64):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def enhance(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(1, 1))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

paths = glob.glob("C:/Users/Iwanna/Documents/tire_counting/test/*.jpg")
for path in paths:
    img = cv2.imread(path)
    new = apply_brightness_contrast(img,brightness=0, contrast=45)
    # new = image_dehazer.remove_haze(img)
    cv2.imshow("new", new)
    cv2.imshow("raw", img)
    cv2.waitKey(0)