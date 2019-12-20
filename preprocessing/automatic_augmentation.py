"""
Created on Thu Nov  7 15:02:57 2019

@author: frincy

Automatic Augmentation module implements augmentation of balanced
dataset to apply augmentation using Augmentor API
The Augmentor pipeline consists of the following:
    1) Rotation
    2) Random Erasing
    3) Random Cropping
    4) Flip
    5) Skew

"""

import Augmentor
import cv2
import os

def automatic_augmentor(source, destination,n_samples):

    if not os.path.exists(destination):
        os.mkdir(destination)

    p = Augmentor.Pipeline(source_directory=source, output_directory=destination, save_format='PNG')

    # 1. Rotation
    p.rotate90(probability=0.1)
    p.rotate270(probability=0.1)
    
    # 2. Random Erasing
    p.random_erasing(probability= 0.2, rectangle_area = 0.6)
    
    # 3. Random Cropping
    p.crop_random(probability=0.1, percentage_area=0.8)
    
    # 4. Flip
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.1)
    
    # 5. Skew
    p.skew(probability=0.05, magnitude=0.3)
    
    # Sampling from the above pipeline
    p.sample(n_samples, multi_threaded = True)
    

    
def create_annotations(source,destination):
    
    imagelist = os.listdir(source)
    if not os.path.exists(destination):
        os.mkdir(destination)
    for image_name in imagelist:
        
        filename_text = image_name[:-4] + '.txt'

        print(image_name)

        image = cv2.imread(os.path.join(source,image_name),1)
        shape = image.shape
        
        """ Since input to automatic augmentation is after cropping 
        from bounding box, the labels would be normalized to image 
        center, width and height
        """
        width,height = shape[1]/shape[1],shape[0]/shape[0]
        x_center, y_center = width/2, height/2
        
        # Finding classname to be written in annotation
        class_name = int(image_name.split('_')[2])
        

        new_file_path = os.path.join(destination,filename_text)
                
        # Writing the annotations to a text file        
        f= open(new_file_path,"w+")
        f.write(str(class_name) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height))
        f.close()

        