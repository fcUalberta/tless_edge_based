"""
Main function for Part 1: Initial Augmentation of Ground Truth Data
It set up all paths and invokes necessary modules like
initial preprocessing, manual_augmentation, automatic augmentation

"""

import os

from preprocessing.preprocessing import crop_at_bb
from preprocessing.manual_augmentation import manual_augmentor 
from preprocessing.automatic_augmentation import automatic_augmentor,create_annotations
from utils.utils import data_balancing, copy_files, copy_separate_files

# main
if __name__ == '__main__':
    
    # Setting up base path, change this to your data
    base = r'D:\data_yolov3'
    
    N_SAMPLES1 = int(180000/2) #Sample size for Automatic Augmentation
    N_SAMPLES2 = int(180000/2)

  
    # Step 1: Finding bounding box and crop the image to get ground truth    
    cropped_images_path = os.path.join(base,'groundtruth')
    crop_at_bb(cropped_images_path)
    
    # Step 2: Finding the data distribution and balancing factor
    balancing_factor = data_balancing(r'D:\data')
    print(balancing_factor)
    
    # Step 3: Manual Augmentation using balancing_factor to remove imbalance of data
    
    m_aug_path = os.path.join(base,'m_augmentor') 
    manual_augmentor(cropped_images_path,m_aug_path,balancing_factor)
    
    # Copying all the ground truth and manually augmented to
    #  one folder for automatic augmentation
    balanced_data_path = os.path.join(base,'balanced_data')
    copy_files(cropped_images_path,balanced_data_path)

    # Copying the balanced images and labels separately to another folder
    balanced_separate_path = os.path.join(base,'balanced_separate')
    balanced_images = os.path.join(balanced_separate_path,'images')
    balanced_labels = os.path.join(balanced_separate_path,'labels')
    copy_separate_files(balanced_data_path,balanced_images,balanced_labels)
    
    # Keeping a copy of ground truth images and labels separately for future    
    groundtruth_separate_path = os.path.join(base,'groundtruth_separate')
    groundtruth_images = os.path.join(groundtruth_separate_path,'images')
    groundtruth_labels = os.path.join(groundtruth_separate_path,'labels')
    copy_separate_files(cropped_images_path,groundtruth_images,groundtruth_labels)
    
    
    # Step 4: Automatic Augmentation after manual augmentation & balancing
    a_aug_path = os.path.join(base,'a_augmentor') 
    a_aug_groundtruth  = os.path.join(a_aug_path,"groundtruth")
    a_aug_balanced  = os.path.join(a_aug_path,"balanced")
    automatic_augmentor(groundtruth_images,a_aug_groundtruth,N_SAMPLES1)
    automatic_augmentor(balanced_images,a_aug_balanced,N_SAMPLES2)
    
    # Creating annotations for automatic augmented files
    create_annotations(a_aug_balanced,os.path.join(a_aug_groundtruth,'labels'))
    
