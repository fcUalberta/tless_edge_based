# Textureless Object Recognition - An Edge-based approach
This project is a part of Ualberta Masters program Course project: Textureless Object Recognition using an edge-based Approach


## About
Textureless object recognition has become a significant task in Computer Vision with the advent of Robotics and its applications in manufacturing sector. It has been challenging to obtain good accuracy in real time because of its lack of discriminative features and reflectance properties which makes the techniques for textured object recognition insufficient for textureless objects.

In this project, by applying image processing techniques we created a robust augmented dataset from initial imbalanced smaller dataset. We extracted edge features, feature combinations and all its combination on the RGB dataset to create 15 datasets, each with a size of ~340,000. We then trained four classifiers on these 15 datasets to arrive at a conclusion as to which dataset performs the best overall and whether edge features are important for textureless objects. Based on our experiments and analysis, RGB images enhanced with combination of 3 edge features performed the best compared to all other. Model performance on dataset with HED edges performed comparatively better than other edge detectors like Canny or Prewitt. 

[Link to full Research Report] (https://github.com/fcUalberta/tless_edge_based/blob/master/Industrial%20Textureless%20Object%20Recognition.pdf)

## Implementation structure
Part 1: Creation of Initial Augmented Data
  - Image Data Acquisition
  - Obtaining ground truth data
  - Data Balancing and Augmentation
    – Manual - Image Processing Techniques
    – Automatic – Augmentor API
    
Part 2: Creation of 14 new datasets for feature comparison
  - Creation of Feature Only Dataset
  - Creation of Feature Combinations
  - Creation of Feature Enhanced RGB Dataset
  - Creation of Feature Combination Enhanced RGB Dataset
  
Part 3: Training and Testing on Multiple Classifiers

## Project Structure

![GitHub Logo](/images_for_readme/projectStructure.png)

## Hardware Requirements
1. 16 GB RAM
1. GPU is not mandatory

## Software Requirements
- Refer requirements.txt

## How to run this application?

1. Create environment with requirements.txt
1. Make sure you have hardware requirements 
1. Download the folder
1. Make sure to changes the paths in any files you would like to run
1. For creating initial augmented data: run initial_augmentation_main.py
1. For creating features/featured enhanced versions: run feature_enhancement_main.py
1. For training/testing on any dataset: run train_test_main.py
  - P.S: This train_test_main.py is specific for canny edge dataset. For training other datasets, make sure you do the following:
    - Change the path of all input/destination folders
    - Preprocessing of test inputs in test() to make it similar to training set 

## Sample Results of Project

### Part 1 : Obtaining Ground truth data

![GitHub Logo](/images_for_readme/DataPreprocessing.png)

### Part 1 : Manual Augmentation

![GitHub Logo](/images_for_readme/manual.jpg)

### Part 1 : Automatic Augmentation

![GitHub Logo](/images_for_readme/automatic.jpg)

### Part 2 : Single feature/Feature Combination dataset 

![GitHub Logo](/images_for_readme/features.jpg)

### Part 2 : Single feature/Feature Combination on RGB images dataset

![GitHub Logo](/images_for_readme/overlay.jpg)

### Reference Papers:
- HED: https://arxiv.org/abs/1504.06375

### Reference scripts:
-	Image segmentation and bounding box : https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
-	Edge Operators: https://medium.com/@nikatsanka/comparing-edge-detection-methods-638a2919476e
- HED: https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/ 
-	Augmentor API: https://github.com/mdbloice/Augmentor 
-	Training Testing: Scikit- learn

