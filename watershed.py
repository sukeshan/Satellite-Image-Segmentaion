from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io
from typing import *
import pandas as pd

def water_shed(predicted,img):
    r'''Arg
    predicted : unet model output
    img : User given Image
    return:
    Overlayed image (boundary detected) ,insta_seg ,dataframe(each object properties)
    '''
    ## transform the unet result to binary image
    #Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
    ret1, thresh = cv2.threshold(predicted, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    # Morphological operations to remove small noise - opening
    #To remove holes we can use closing
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    #from skimage.segmentation import clear_border
    #opening = clear_border(opening) #Remove edge touching grains. 
    #Check the total regions found before and after applying this. 

    #Now we know that the regions at the center of cells is for sure cells
    #The region far away is background.
    #We need to extract sure regions. For that we can use erode. 
    #But we have cells touching, so erode alone will not work. 
    #To separate touching objects, the best approach would be distance transform and then thresholding.

    # let us start by identifying sure background area
    # dilating pixes a few times increases cell boundary to background. 
    # This way whatever is remaining for sure will be background. 
    #The area in between sure background and foreground is our ambiguous area. 
    #Watershed should find this area for us. 
    sure_bg = cv2.dilate(opening,kernel,iterations=10)

    # Finding sure foreground area using distance transform and thresholding
    #intensities of the points inside the foreground regions are changed to 
    #distance their respective distances from the closest 0 value (boundary).
    #https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)


    #Let us threshold the dist transform by starting at 1/2 its max value.
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(),255,0)

    #Later you may realize that 0.2*max value may be better. Also try other values. 
    #High value like 0.7 will drop some small mitochondria. 

    # Unknown ambiguous region is nothing but bkground - foreground
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    #Now we create a marker and label the regions inside. 
    # For sure regions, both foreground and background will be labeled with positive numbers.
    # Unknown regions will be labeled 0. 
    #For markers let us use ConnectedComponents. 
    ret3, markers = cv2.connectedComponents(sure_fg)

    #One problem rightnow is that the entire background pixels is given value 0.
    #This means watershed considers this region as unknown.
    #So let us add 10 to all labels so that sure background is not 0, but 10
    markers = markers+10

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    #plt.imshow(markers, cmap='gray')   #Look at the 3 distinct regions.

    #Now we are ready for watershed filling. 
    markers = cv2.watershed(img, markers)
    #plt.imshow(markers, cmap='gray')
    #The boundary region will be marked -1
    #https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1

    #Let us color boundaries in yellow. 
    img[markers == -1] = [0,255,255]  

    img2 = color.label2rgb(markers, bg_label=0)

   

    #Now, time to extract properties of detected cells
    # regionprops function in skimage measure module calculates useful parameters for each object.


    props = measure.regionprops_table(markers, intensity_image= predicted, 
                                  properties=['label',
                                              'area', 'equivalent_diameter',
                                              'mean_intensity', 'solidity'])
    df = pd.DataFrame(props)
    return img ,img2 ,df
