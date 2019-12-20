"""
Reference: https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/
"""

# import the necessary packages
#import argparse
import cv2
import matplotlib.pyplot as plt
import PIL.Image as pil
import numpy as np
import sys
#import os
#import glob

## construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--edge-detector", type=str, required=True,
#	help="path to OpenCV's deep learning edge detector")
#ap.add_argument("-i", "--Pics", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())

class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]


def load_hed_model():
    
    # load our serialized edge detector from disk
    print("[INFO] loading edge detector...")
    protoPath = r"C:\PythonCodes\MM803\code\feature_enhancement\hed\hed_model\deploy.prototxt"
    modelPath = r"C:\PythonCodes\MM803\code\feature_enhancement\hed\hed_model\hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # register our new layer with the model
    cv2.dnn_registerLayer("Crop", CropLayer)
    
    #print(net.shape)
    print(net)
    return net

def holistically_nested(net,image,alpha = 1, beta=0.6):
    
#    image = cv2.UMat.get(image)
    # load the all input images and grab their dimensions
    #image = cv2.imread(imagePath,cv2.COLOR_BGR2GRAY)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                                 mean=(104.00698793, 116.66876762, 122.67891434),swapRB=False, crop=False)
#    print("[INFO] performing holistically-nested edge detection...")
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")
#    cv2.imshow("HED", hed)
#    cv2.imshow("Input", gray_img)
#    cv2.waitKey(0)
    
    
    # Create Overlay
#    hed_bgr = cv2.cvtColor(hed,cv2.COLOR_GRAY2BGR)
#    hed_overlay = cv2.addWeighted(image,alpha, hed, beta,0)
    
    # Create Overlay
#    
#    hed_overlay = cv2.addWeighted(image,alpha, 255-hed_bgr, beta,0)
#    plt.imshow(hed_overlay)
#    plt.title("HED overlay")
#    plt.show()
#    
    image1 = pil.fromarray(image)
    hed1 = pil.fromarray(hed)
    pil_overlay=pil.blend(image1.convert('RGBA'),hed1.convert('RGBA'),beta)
#    plt.imshow(pil_overlay)
#    plt.title("test")
#    plt.show()
    
    
    
#    
#    return hed, hed_overlay,np.float32(pil_overlay)
    return np.float32(pil_overlay)

