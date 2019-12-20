"""
Created on Tue Nov  5 23:31:05 2019

@author: frincy


Manual Augmentation module implements augmentation of the ground truth
data by applying 4 image processing techniques: 
    1) Contrast Enhancement
    2) Adding white noise
    3) Brightness transformation
    4) Adding blur
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

from skimage import exposure
    
def image_processing(image, image_name, destination,i,source_file):
    """
    Function to apply augmentation via image processing techniques
    
    Args: 
        image: Image file on which augmentation needs to be applied
        image_name: Name of the image on which augmentation is applied
        destination: Directory for the destination annotation file for the augmented images
        i: The specific number of times the augmentation aplied for the same image so as to balance the data
        source_file: path to copy the ground truth annotation file for image 'image'
        
    """
    
    # file name without extension
    name = image_name[:-4]
    
    # 1. Contrast Enhancement by gamma correction
    contrast_enhanced = exposure.adjust_gamma(image, np.random.choice([1.65,1.75,1.85]))

    # Display contrast enhanced image
    plt.subplot(121),plt.imshow(image),plt.title('Original')
    plt.subplot(122),plt.imshow(contrast_enhanced),plt.title('Contrast Enhanced')
    plt.show()
    
    # Write contrast enhanced image
    cv2.imwrite(os.path.join(destination,name+"_m_contrast_"+str(i)+".png"),contrast_enhanced)
    dest_file_contrast = os.path.join(destination,name+"_m_contrast_"+str(i)+".txt")
    shutil.copyfile(source_file,dest_file_contrast)
    
    
    # 2. Adding White noise
    mean = np.random.choice([1,0.1,0.01])   # some constant
    var =  np.random.choice([0.0002,0.00002])
    std = var**0.5    
    noisy_img = image + np.random.normal(mean, std, image.shape)
    #noisy_img_clipped = np.clip(noisy_img, 0, 255) 
    
    # Display noise added image
    plt.subplot(121),plt.imshow(image),plt.title('Original')
    plt.subplot(122),plt.imshow(noisy_img),plt.title('White Noise')
    plt.show()
   
    # Write noise added image
    cv2.imwrite(os.path.join(destination,name+"_m_White_noise_"+str(i)+".png"),noisy_img)
    dest_file_noise = os.path.join(destination,name+"_m_White_noise_"+str(i)+".txt")
    shutil.copyfile(source_file,dest_file_noise)
    
    
    # 3. Brightness Transformation
    alpha = 1.25 # Simple contrast control
    beta = np.random.choice([2,2.5,3])    # Simple brightness control

    brightness_transformed = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Display brightness tranformed image
    plt.subplot(121),plt.imshow(image),plt.title('Original')
    plt.subplot(122),plt.imshow(brightness_transformed),plt.title('Brightness transformed')
    plt.show()
    
    # Write brightness transformed image
    cv2.imwrite(os.path.join(destination,name+"_m_bright_"+str(i)+".png"),brightness_transformed)
    dest_file_bright = os.path.join(destination,name+"_m_bright_"+str(i)+".txt")
    shutil.copyfile(source_file,dest_file_bright)

    # Adding Blur
    filter_b = np.random.choice([3,5,7])
    blur = cv2.GaussianBlur(image,(filter_b,filter_b),0)
    
    # Display blurred image
    plt.subplot(121),plt.imshow(image),plt.title('Original')
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.show()
    
    # Write blurred image
    cv2.imwrite(os.path.join(destination,name+"_m_blur_"+str(i)+".png"),blur)
    dest_file_blur = os.path.join(destination,name+"_m_blur_"+str(i)+".txt")
    shutil.copyfile(source_file,dest_file_blur)


def manual_augmentor(source,destination,balancing_factor):
    """
    Function to apply augmentation to replicate industry lighting and camera settings
    
    Args:
        source: directory for the source ground truth annotation file
        destination: directory for the destination annotation file for the augmented images
    
    """
    
    # Creating destination path
    if not os.path.exists(destination):
        os.mkdir(destination)
    # Source path
    image_list = os.listdir(source)
    for image_name in image_list:
        
        # Finding classname from filename of image
        class_name = int(image_name.split('_')[0])
        
             # Skipping annotation files from ground truth
        if image_name[-4:] == '.txt':
            continue
                
        image = cv2.imread(os.path.join(source,image_name),1)
        
        # Finding the corresponding annotation file
        text_filename = image_name[:-4]+'.txt'
        source_file = os.path.join(source,text_filename)
         
        # Finding classname from filename of image
        class_name = int(image_name.split('_')[0])
        #print("Balancing",balancing_factor)
        dist_range = balancing_factor[class_name]
        
        # Manual augmentation with balancing the data distriution
        for i in range(dist_range):
            image_processing(image,image_name,destination,i,source_file)
        
