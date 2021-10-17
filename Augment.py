import cv2
import numpy as np


def save_img(img ,label_img ,count):
    print('fun',count)
    cv2.imwrite(  f'/home/Change the path /image{count}.png',img) #path for augmented iamge
    cv2.imwrite( f'/home/Change the path/label{count}.png' ,label_img)# path for augmneted maske
    return 


def transpose_img(img ,label_img ,count):
    save_img(img.transpose(1,0,2) ,label_img.transpose(1,0,2) ,count)
    return

def row_flip( img ,label_img ,count):
    save_img(img[::-1] ,label_img[::-1] ,count)  
    return 

def col_flip(img ,label_img ,count):
    save_img(img[:,::-1] ,label_img[:,::-1] ,count)    
    return

def rev_transimg(img ,label_img ,count):
    img_aug = img[::-1]
    label_aug = label_img[::-1]
    save_img(img_aug.transpose(1,0,2) ,label_aug.transpose(1,0,2) ,count)
    return

def semi_row_and_col(img ,label_img ,count)
    empty_img = img.copy()
    empty_label = label_img.copy()
    for i in range(int(img.shape[1] /2)):
        empty_img[:,[-temp]] =img[:,[i]] 
        temp+=1
    temp = 1
    for i in range(int(img.shape[0] /2)):
        empty_img[-temp] =img[i] 
        temp+=1
     # label   
    for i in range(int(img.shape[1] /2)):
        empty_label[:,[-temp]] =img[:,[i]] 
        temp+=1
    temp = 1
    for i in range(int(img.shape[0] /2)):
        empty_label[-temp] =img[i] 
        temp+=1
    save_img(empty_img ,empty_label ,count)
    return

count = 180 # total Number of image you want to augment
fun = [transpose_img ,rotate_180 ,row_flip ,col_flip]
for i in range(1,count):
    for fun_names in fun:
        count+=1
        img = cv2.imread(f'image{i}.tif')
        label_img = cv2.imread(f'image{i}.tif')
        fun_names(img ,label_img ,count) 