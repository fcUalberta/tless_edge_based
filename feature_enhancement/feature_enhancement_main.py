"""
Created on Fri Nov 15 20:15:07 2019

@author: frincy

Implements Part 2 of our project: Creating features/feature enhanced dataset

"""

import numpy as np
import cv2
import shutil
import os
import sys
sys.path.append(r"C:\PythonCodes\MM803\code\feature_enhancement\hed") # appending hed_path to PATH
import hed_features
import matplotlib.pyplot as plt
import PIL.Image as pil

def canny(image,gray_img,alpha = 1, beta=0.6, sigma=0.33):
    """
    Implements Canny edge operator
    
    Args:
        image: RGB version of input image on which overlay with edge is applied
        gray_img: Gray version of input image on which edge operator is applied
        alpha: blending ratio for RGB image
        beta: blending ratio for edge feature
        
    Returns:
        newgradientImage: Canny edge feature of input image
        pil_overlay: Canny enhanced RGB version converted to numpy float
    """
	
	# compute the median of the single channel pixel intensities
    v = np.median(gray_img)
	
	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny_edge = cv2.Canny(gray_img, lower, upper)

    # Creating canny enhanced RGB image
    image1 = pil.fromarray(image)
    newgradientImage1 = pil.fromarray(canny_edge)
    pil_overlay=pil.blend(image1.convert('RGBA'),newgradientImage1.convert('RGBA'),0.4)
    
    return canny_edge, np.float32(pil_overlay)



def sobel(image,gray_img,alpha = 1, beta=0.6):
    
    """
    Implements Sobel edge operator
    
    Args:
        image: RGB version of input image on which overlay with edge is applied
        gray_img: Gray version of input image on which edge operator is applied
        alpha: blending ratio for RGB image
        beta: blending ratio for edge feature
        
    Returns:
        newgradientImage: Sobel edge feature of input image
        pil_overlay: sobel enhanced RGB version converted to numpy float
    """
    h, w = gray_img.shape
    # define filters
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1
    
    # define images with 0s
    newhorizontalImage = np.zeros((h, w))
    newverticalImage = np.zeros((h, w))
    newgradientImage = np.zeros((h, w))
    
    # offset by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                             (horizontal[0, 1] * gray_img[i - 1, j]) + \
                             (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                             (horizontal[1, 0] * gray_img[i, j - 1]) + \
                             (horizontal[1, 1] * gray_img[i, j]) + \
                             (horizontal[1, 2] * gray_img[i, j + 1]) + \
                             (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                             (horizontal[2, 1] * gray_img[i + 1, j]) + \
                             (horizontal[2, 2] * gray_img[i + 1, j + 1])
    
            newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)
    
            verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                           (vertical[0, 1] * gray_img[i - 1, j]) + \
                           (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                           (vertical[1, 0] * gray_img[i, j - 1]) + \
                           (vertical[1, 1] * gray_img[i, j]) + \
                           (vertical[1, 2] * gray_img[i, j + 1]) + \
                           (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                           (vertical[2, 1] * gray_img[i + 1, j]) + \
                           (vertical[2, 2] * gray_img[i + 1, j + 1])
    
            newverticalImage[i - 1, j - 1] = abs(verticalGrad)
    
            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag
    
    # Creating Sobel enhanced RGB image
    image1 = pil.fromarray(image)
    newgradientImage1 = pil.fromarray(newgradientImage)
    pil_overlay=pil.blend(image1.convert('RGBA'),newgradientImage1.convert('RGBA'),0.4)

    return newgradientImage, np.float32(pil_overlay)
   

def prewitt(image,gray_img,alpha = 1, beta=0.6):
    
    """
    Implements Prewitt edge operator
    
    Args:
        image: RGB version of input image on which overlay with edge is applied
        gray_img: Gray version of input image on which edge operator is applied
        alpha: blending ratio for RGB image
        beta: blending ratio for edge feature
        
    Returns:
        newgradientImage: Prewitt edge feature of input image
        pil_overlay: Prewitt enhanced RGB version converted to numpy float
    """
    h, w = gray_img.shape
		
	# define filters
    horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # s1
		
	# define images with 0s
    newgradientImage = np.zeros((h, w))
		
	# offset by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
														 (horizontal[0, 1] * gray_img[i - 1, j]) + \
														 (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
														 (horizontal[1, 0] * gray_img[i, j - 1]) + \
														 (horizontal[1, 1] * gray_img[i, j]) + \
														 (horizontal[1, 2] * gray_img[i, j + 1]) + \
														 (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
														 (horizontal[2, 1] * gray_img[i + 1, j]) + \
														 (horizontal[2, 2] * gray_img[i + 1, j + 1])
            verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
													 (vertical[0, 1] * gray_img[i - 1, j]) + \
													 (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
													 (vertical[1, 0] * gray_img[i, j - 1]) + \
													 (vertical[1, 1] * gray_img[i, j]) + \
													 (vertical[1, 2] * gray_img[i, j + 1]) + \
													 (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
													 (vertical[2, 1] * gray_img[i + 1, j]) + \
													 (vertical[2, 2] * gray_img[i + 1, j + 1])
		
            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag													

    # Creating Prewitt enhanced RGB image
    image1 = pil.fromarray(image)
    newgradientImage1 = pil.fromarray(newgradientImage)
    pil_overlay=pil.blend(image1.convert('RGBA'),newgradientImage1.convert('RGBA'),0.5)
    return newgradientImage,np.float32(pil_overlay)

   
    
def create_path(destination):
    """
    Creates the folder if doesnt exists already
    
    Args: 
        destination: path of folder structure to be created
    """
    
    if not os.path.exists(destination):
        os.makedirs(destination)
        
def write_files(image_dest,file_dest,str_to_add,filename,image,source_file):
    """
    Saves the feature/feature enhanced RGB dataset with annotations 
    
    Args:
        image_dest: destination folder for image file
        file_dest: destination folder for annotation text file
        str_to_add: string to be added to the original filename
        filename: filename of image and text file
        image: image to be copied to image_dest
        source_file: annotation file to be copied to file_dest
    
    """
    cv2.imwrite(os.path.join(image_dest,str_to_add+ filename +".png"),image)
    dest_file = os.path.join(file_dest,str_to_add+ filename +".txt")
    shutil.copyfile(source_file,dest_file)
    
    
    
#def feature_enhancement():
if __name__ == '__main__':    
    
    # Loading HED model
    net = hed_features.load_hed_model()
    
    # Setting source file paths    
    imagepath = r"C:\PythonCodes\MM803\training_data\images"
    filepath = r"C:\PythonCodes\MM803\training_data\labels"
    base_dest = r"C:\PythonCodes\MM803\Feature_Only_data_Third"
    
    # Setting the destination files paths for saving 14 datasets
    dest_image11 = os.path.join(base_dest,r"features_canny\images")
    dest_file11 =  os.path.join(base_dest,r"features_canny\labels")
    create_path(dest_image11)
    create_path(dest_file11)
    
    dest_image13 = os.path.join(base_dest,r"overlay_canny2\images")
    dest_file13 =  os.path.join(base_dest,r"overlay_canny2\labels")
    create_path(dest_image13)
    create_path(dest_file13)
        
    dest_image21 = os.path.join(base_dest,r"features_sobel\images")
    dest_file21 =  os.path.join(base_dest,r"features_sobel\labels")
    create_path(dest_image21)
    create_path(dest_file21)
    
    dest_image23 = os.path.join(base_dest,r"overlay_sobel2\images")
    dest_file23 =  os.path.join(base_dest,r"overlay_sobel2\labels")
    create_path(dest_image23)
    create_path(dest_file23)
    
    dest_image31 = os.path.join(base_dest,r"features_prewitt\images")
    dest_file31 =  os.path.join(base_dest,r"features_prewitt\labels")
    create_path(dest_image31)
    create_path(dest_file31)
    
    dest_image33 = os.path.join(base_dest,r"overlay_prewitt2\images")
    dest_file33 =  os.path.join(base_dest,r"overlay_prewitt2\labels")
    create_path(dest_image33)
    create_path(dest_file33)
    
    dest_image41 = os.path.join(base_dest,r"features_hed\images")
    dest_file41 =  os.path.join(base_dest,r"features_hed\labels")
    create_path(dest_image41)
    create_path(dest_file41)
    
    dest_image43 = os.path.join(base_dest,r"overlay_hed2\images")
    dest_file43 =  os.path.join(base_dest,r"overlay_hed2\labels")
    create_path(dest_image43)
    create_path(dest_file43)
     
    
    imagelist = os.listdir(imagepath)
    filelist = os.listdir(filepath)
    
    for images in imagelist:
        #image = cv2.imread(images,1)
        imagename = os.path.join(imagepath,images)
        
        filename = images[:-4]
        # Getting the annotations of input file to be saved for new dataset
        text_filename = filename +'.txt' 
        source_file = os.path.join(filepath,text_filename)
        
        
        # Reading and processing image
        image = cv2.imread(imagename,cv2.COLOR_BGR2GRAY)
        gray_img = cv2.imread(imagename,0)
        blurred = cv2.GaussianBlur(gray_img,(3,3),0)
        
        # Detecting Canny Edge
        canny_edge, canny_pil = canny(image,blurred,alpha = 1, beta=0.6)
        
        # Displaying Results of canny operator   
        plt.subplot(131),plt.imshow(image),plt.title('Original')
        plt.subplot(132),plt.imshow(canny_edge,cmap= "gray"),plt.title('Canny')
        plt.subplot(133),plt.imshow(canny_pil),plt.title('Canny_Overlay')
        plt.show()
                        

        # Writing files
        write_files(dest_image11,dest_file11,"f_canny_",filename,canny_edge,source_file)
        write_files(dest_image13,dest_file13,"p_canny_",filename,canny_pil,source_file)
        
        # Detecting sobel edge        
        sobel_edge,sobel_pil = sobel(image,gray_img,alpha = 1, beta=0.5)

        # Displaying Results of sobel operator         
        plt.subplot(131),plt.imshow(image),plt.title('Original')
        plt.subplot(132),plt.imshow(sobel_edge,cmap= "gray"),plt.title('Sobel')
        plt.subplot(133),plt.imshow(sobel_pil),plt.title('Sobel_Overlay')
        plt.show()
        
        # Writing files
        write_files(dest_image21,dest_file21,"f_sobel_",filename,sobel_edge,source_file)
        write_files(dest_image23,dest_file23,"p_sobel_",filename,sobel_pil,source_file)
       
        # Detecting Prewitt edge
        prewitt_edge,prewitt_pil = prewitt(image,gray_img,alpha = 1, beta=0.5)

        # Displaying Results of prewitt operator
        plt.subplot(131),plt.imshow(image),plt.title('Original')
        plt.subplot(132),plt.imshow(prewitt_edge,cmap= "gray"),plt.title('Prewitt')
        plt.subplot(133),plt.imshow(prewitt_pil),plt.title('Prewitt_Overlay')
        plt.show()
       
        # Writing files
        write_files(dest_image31,dest_file31,"f_prewitt_",filename,prewitt_edge,source_file)
        write_files(dest_image33,dest_file33,"p_prewitt_",filename,prewitt_pil,source_file)
      
        # Detecting Holistically Nested Edge 
        hed_edge,hed_pil = hed_features.holistically_nested(net,image,alpha = 1, beta=0.5)
        
        # Displaying Results of HED edges
        plt.subplot(131),plt.imshow(image),plt.title('Original')
        plt.subplot(132),plt.imshow(hed_edge,cmap= "gray"),plt.title('HED')
        plt.subplot(133),plt.imshow(hed_pil),plt.title('HED_Overlay')
        plt.show()
        
        # Writing files
        write_files(dest_image41,dest_file41,"f_hed_",filename,hed_edge,source_file)
        write_files(dest_image43,dest_file43,"p_hed_",filename,hed_pil,source_file)        
        
        
        