"""
Created on Thu Nov  7 01:54:32 2019

@author: frincy

Utils module implements utility functions like
1) Finding data distribution
2) Finding data balncing factor
3) Copying files from one folder to another
4) Separating images and labels from source to diffent destinations

"""

import os
import numpy as np
import shutil
from PIL import Image

def data_distribution(path):
    """
    Function to find the data distribution in the ground truth data
    
    Args:
        path: Path where ground truth data is found
        
    Returns:
        dist_list:  List of  data distriution for all classes
        The position in the list denotes the class and the value denotes the number of data for that class
    """
    
    pos_list = os.listdir(path)
    dist_list = []
    sum = 0
            
    for i in range(1,31):
        obj = str(i)+'_rgb'
        count = 0
        for pos in pos_list:
            object_list_path = os.path.join(path,pos)
    
            object_list = os.listdir(object_list_path)
    
            for object_name in object_list:
                image_list_path = os.path.join(object_list_path,object_name)
    
                image_list = os.listdir(image_list_path)
                for imagepath in image_list:
                    if imagepath == 'mm':
                        continue
                    if object_name == obj:
                        count += 1
                        #print(object_name,obj)
                        
         #   print("Count of ",obj," is",count)
        print("Count of ",obj," is",count)
        sum +=count
        print(i)
        dist_list.append(count)
            
    print("Total Sum: ",sum)
    return dist_list

def data_balancing(path):
    """
    Function to find the multiplication factor for balancing data in all classes
        
    Args: 
        path: Path where ground truth data is found
        
    Returns:
        balancing_factor: List of multipliers needed to balance each dataset.
        The position in the list denotes the class and the value denotes the multiplier
    """
    
    distribution_list = data_distribution(path)
    
    balancing_factor = []
    for i in range(len(distribution_list)):
        #print(i,distribution_list[i])
        #multiplier = max(distribution_list) / distribution_list[i] - 1
        multiplier = (np.round(5000 / distribution_list[i],0))
        multiplier = int(np.round(multiplier/4,0))
        balancing_factor.append(multiplier)
        #print("sddada",max(distribution_list) / distribution_list[i])
    return balancing_factor

def copy_files(source,destination):
    """
    Function to copy all files from source to destination
    
    Args:
        source: Path to source files
        destination: Path to destination folder
    """
    filelist = os.listdir(source)
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    for filename in filelist:
        source_file = os.path.join(source,filename)
        shutil.copy(source_file,destination)
        
def copy_separate_files(source,dest1,dest2):
    """
    Function to copy all files from source to destination
    
    Args:
        source: Path to source files
        destination: Path to destination folder
    """
    filelist = os.listdir(source)
    
    if not os.path.exists(dest1):
        os.mkdir(dest1)
    
    if not os.path.exists(dest2):
        os.mkdir(dest2)
    
    for filename in filelist:
        source_file = os.path.join(source,filename)
        
        if filename[-4:] == '.png':
            shutil.copy(source_file,dest1)
        else:
            shutil.copy(source_file,dest2)
            