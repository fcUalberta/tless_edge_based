"""
Created on Tue Nov  5 23:31:05 2019

@author: frincy

Preprocessing module finds the bounding box of class objects from input images
And to crop the images at bounding box to align the ground truth data at its center

"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import rgb2gray

def bounding_box(image,object_name,BLACK_BACKGROUND):
    """
    Function to find the bounding box of the object of interest
    
    Reference: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
    
    Args:
        image: Image on which the bounding box need to be identified
        object_name: The name of object of interest
        BLACK_BACKGROUND: list of objects with black background
        
    Returns:
        List of bounding box coordinates: Xmin, Ymin, width and height
    
    """
    
    # Converting image to grayscale
    gray = rgb2gray(image)
    
    # Applying Otsu's threshold
    thresh = threshold_otsu(gray)
    bw = closing(gray > thresh, square(3))
    
    # Complementing images with objects in white background
    if (object_name not in BLACK_BACKGROUND):
        bw = 255-bw
    
    # Remove artifacts connected to image border
    cleared = clear_border(bw)

    # Label image regions
    label_image = label(cleared)
    #image_label_overlay = label2rgb(label_image, image=image)


    # Display image with bounding box
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    minc,minr,maxc,maxr=0,0,0,0
    for region in regionprops(label_image):
        
        # Take regions with large enough areas
        if region.area >= 5000:
            #print(region.area)
            
            # Draw rectangle on the selected bounding box
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    plt.tight_layout()
    plt.show()
    
    return minc,minr,maxc-minc,maxr-minr


def crop_at_bb(cropped_images_path):
    """
    Function to crop images after finding the bounding box for objects
    
    Args: 
        
        cropped_images_path: Destination of cropped images     
            
    """
    BLACK_BACKGROUND = ['17_rgb','18_rgb','23_rgb','26_rgb','27_rgb','28_rgb']

    # Coordinates to focus for finding bounding box for each orientation
    POS_DIM_LIST = { 'pos1':[525, 275, 335, 315],
                   'pos2':[0,0,625,600],
                   'pos3':[600, 0, 680, 700],
                   'pos4':[600,0,680,700]}
    
    # Creating base folder for cropped images
    if not os.path.exists(cropped_images_path):
        os.makedirs(cropped_images_path)

    path = r'D:\data' # path to original data
    pos_list = os.listdir(path) # Listing all files/folders in the main directory
    
    for pos in pos_list:

        object_list_path = os.path.join(path,pos)
        #print(object_list_path)
        object_list = os.listdir(object_list_path)

        dim_bb_list = POS_DIM_LIST[pos] # For each position, getting BB focus area
        x,y,w,h = dim_bb_list[0],dim_bb_list[1],dim_bb_list[2],dim_bb_list[3]

        for object_name in object_list:
            
            image_list_path = os.path.join(object_list_path,object_name)
  
            image_list = os.listdir(image_list_path)
 
            #count = 0
            for imagename in image_list:
                
                # Bypassing unwanted folder
                if imagename == 'mm':
                    continue
                
                # Loading color images using opencv
                image = cv2.imread(os.path.join(image_list_path,imagename),1)
                print(os.path.join(image_list_path,imagename))

                # Function call to get the bounding box
                bb = bounding_box(image[y:y+h,x:x+w],object_name,BLACK_BACKGROUND)
                
                # Adjusting bounding box values for one class object
                if(pos=='pos2' and object_name=='25_rgb'):
                    x_c,y_c,w_c,h_c = bb[0]+x-5,bb[1]-33+y,bb[2]+10,bb[3]+47
                    x_new,y_new,w_new,h_new = 5,33,bb[2],bb[3]+20 # BB for YoloV3
                    
                else:
                    x_c,y_c,w_c,h_c = bb[0]+x-5,bb[1]-7+y,bb[2]+10,bb[3]+14
                    x_new,y_new,w_new,h_new = 5,7,bb[2],bb[3] # BB for YoloV3
                    
                # Initializing new cropped image
                cropped_new_image = image[y_c:y_c+h_c,x_c:x_c+w_c]
                new_shape  = cropped_new_image.shape
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(cropped_new_image)
                rect = mpatches.Rectangle((x_new, y_new), w_new, h_new,fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                plt.tight_layout()
                plt.show()

                # Creating normalized labels for training on YOLOv3
                
                x_center = (np.round((x_new+w_new/2),0)/new_shape[1])
                y_center = (np.round((y_new+h_new/2),0)/new_shape[0])
                width = w_new/new_shape[1]
                height = h_new/new_shape[0]
                
                # Class details
                
                class_name = int(object_name[:-4]) - 1
                filename_image = str(class_name) + '_' + pos + imagename
                filename_text = filename_image[:-4]+'.txt'
                
                # Copying the cropped images and corresponding annotations for BB
                
                new_image_path = os.path.join(cropped_images_path,filename_image)
                new_file_path = os.path.join(cropped_images_path,filename_text)
                #print(crop_name)
                
                # Writing the image and text file
                cv2.imwrite(new_image_path,cropped_new_image)
                f= open(new_file_path,"w+")
                f.write(str(class_name) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height))
                f.close()
                





