#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:18:27 2019

@author: qurunlu and all the OH5 team
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import sys
import cv2
import numpy as np
import pytesseract
import os

def rotate_bound(image, angle):
    #获取宽高
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image=cv2.bitwise_not(image)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # 提取旋转矩阵 sin cos 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    #     nH = int((h * cos) + (w * sin))
    nH = h
 
    # 调整旋转矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def get_minAreaRect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    return cv2.minAreaRect(coords)

def rot_img(image): 
    #image_path = nom_img
    #image = cv2.imread(image_path)
    angle = get_minAreaRect(image)[-1]+90
    rotated = rotate_bound(image, angle)
def reco_im(nom_img):
    #image=retouche_img(nom_img)
    image=cv2.imread(nom_img)
    #angle = get_minAreaRect(image)[-1]+90
    #rotated = rotate_bound(image, angle)
    #img_cv=np.array(image)


    image[image <180]=0
    #image=rot_img(image)
    #path=''
    #cv2.imwrite(path,rotated)

    text= pytesseract.image_to_string(image,config='outputbase digits')
    t=os.path.splitext(nom_img)
    cv2.imwrite(t[0]+'_processing_'+text+t[1], image)
    return text

def coord_centre(liste):
    centre=[]
    centre_abs=liste[0]+liste[2]/2
    centre_ord=liste[1]-liste[3]/2
    centre.append(centre_abs)
    centre.append(centre_ord)
    return centre

fichier=open("data.txt","w")

# USAGE
# python text_detection_morpho.py --image images/lebron_james.jpg --height 21

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# files and directory manipulation
import sys
import os
import getopt


def round_up_to_odd(f):
    """
    Round a number to the nearest larger odd integer
    :param f: float to round
    :type f: float
    :return: the smallest odd integer larger than g
    :rtype: int
    """
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f


def localize_numbers(image_path, expected_digit_height):
    """
    Localize all numbers in an image.

    :param image_path: path to image
    :param expected_digit_height: expected size of the digits (height or width) in pixels
    :type image_path: string
    :type expected_digit_height: int
    :return: coordinates of all bounding boxes (xmin, ymin, width, height)
    :rtype: list of list
    """
    # Check image exists
    assert os.path.exists(image_path), "Could not find file at, " + str(image_path)

    # load the image and grab the image dimensions
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    blurring_kernel_size = round_up_to_odd(expected_digit_height / 4)
    opening_kernel_size = round_up_to_odd(expected_digit_height / 4)
    print("opening_kernel_size = ", opening_kernel_size)

    # Smooth the image using a Gaussian
    smooth = cv2.GaussianBlur(image, (blurring_kernel_size, blurring_kernel_size), 0)
    smooth_small = cv2.resize(smooth, (960, 960))
    #cv2.imshow("Step 1: blur", smooth_small)
    #cv2.waitKey(0)

    # Otsu binary
    grey = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    ret2, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_small = cv2.resize(binary, (960, 960))
    #cv2.imshow("Step 2: Otsu", binary_small)
    #cv2.waitKey(0)

    # Erode then dilate (i.e. open)
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel_size, opening_kernel_size))
    binary_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, sqKernel)
    binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, sqKernel)
    binary_small = cv2.resize(binary_open, (960, 960))
    #cv2.imshow("Step 3: Opening", binary_small)
    #cv2.waitKey(0)

    # Detect connected components
    npaContours, npaHierarchy = cv2.findContours(binary_open.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 0.25 * expected_digit_height * expected_digit_height
    max_contour_area = 4 * expected_digit_height * expected_digit_height

    image_all_boxes = image.copy()
    image_true_boxes = image.copy()

    bounding_box_list = []
    for npaContour in npaContours:
        [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
        cv2.rectangle(image_all_boxes, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)
        if min_contour_area < cv2.contourArea(npaContour) < max_contour_area:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
            bounding_box_list.append([intX, intY, intW, intH])
            cv2.rectangle(image_true_boxes, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)

    # Write image to file
    #cv2.imwrite(image_path + "result_allBoxes_open.png", image_all_boxes)
    #cv2.imwrite(image_path + 'result_trueBoxes_open.png', image_true_boxes)

    image_all_boxes_small = cv2.resize(image_all_boxes, (960, 960))
    #cv2.imshow("Step 4: detection all boxes", image_all_boxes_small)
    #cv2.waitKey(0)

    image_true_boxes_small = cv2.resize(image_true_boxes, (960, 960))
    #cv2.imshow("Step 5: detection boxes of correct size", image_true_boxes_small)
    #cv2.waitKey(0)

    return bounding_box_list


def crop_image_using_bounding_boxes(image_path, bounding_box):
    """
   Crop an image into snippets using bounding boxes

    :param image_path: path to image
    :param bounding_box: coordinates of the bounding box (xmin, ymin, width, height)
    :type image_path: basestring
    :type bounding_box list
    :return image
    :rtype: cv2 image
  """
    # Check image exists
    assert os.path.exists(image_path), "Could not find file at, " + str(image_path)

    # load the image and grab the image dimensions
    image = cv2.imread(image_path)

    # Crop
    [int_x, int_y, int_w, int_h] = bounding_box
    crop_img = image[int_y:int_y + int_h, int_x:int_x + int_w]
    return crop_img


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
                    help="path to input image")
    ap.add_argument("-height", "--height", type=str,
                    help="expected digit height in pixel")
    args = vars(ap.parse_args())

    image_path = args["image"]
    expected_digit_height = int(args["height"])

    bounding_box_list = localize_numbers(image_path, expected_digit_height)
    print("Processing ", len(bounding_box_list), " bounding boxes")
    i=0
    for bounding_box in bounding_box_list:
        [t_x, int_y, int_w, int_h] = bounding_box
        crop_img = crop_image_using_bounding_boxes(image_path, bounding_box)
        i=i+1
        nom='img_'+str(i)+'.png'
        cv2.imwrite(nom,crop_img)
    	
        #cv2.imwrite(image_path + "_crop" + str(int_x) + "_" + str(int_y) + "_" + str(int_w) + "_" + str(int_h) + ".png", crop_img)


        coord=str(coord_centre(bounding_box))
        data_num=(reco_im(nom))

        fichier.write(coord+"   "+data_num+"\n")
    fichier.close()




if __name__ == "__main__":
    main(sys.argv[1:])
