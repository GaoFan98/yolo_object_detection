import numpy as np
import cv2

# for laptops => 0, cause it has only one camera
# else put 1 or any other value for camera
capture = cv2.VideoCapture(0)
# image parameters
width = 320
height = 320

# fetch coco dataset class names to list
class_file = 'coco.names'
class_names = []

# open file and push it to class names list
# rt=>read file
with open(class_file, 'rt') as function:
    class_names = function.read().rstrip('\n').split('\n')
# print(class_names)

# YOLO NETWORK
# import config and weights
modelConfig = 'yolov3.cfg'
modelWeight = 'yolov3.weights'

# import DarkNet
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeight)
# define openCV as backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# run on CPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    # RUN THE CAMERA
    success, image = capture.read()
    # convert image to blob format
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (width, height), [0, 0, 0], 1, crop=False)
    # set blob fomat image as input in net
    net.setInput(blob)
    # get layers names list
    layerNames = net.getLayerNames()
    # get output layers names on index
    # i[0] -1 because of 0 start
    outputLayerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # get output layers indices
    # net.getUnconnectedOutLayers()

    # send images to net
    # [85] => 80 classes + width + height + X center + Y center + confidence (probability of object presents in frame)
    output = net.forward(outputLayerNames)

    # show image we want to display
    cv2.imshow('Image', image)
    # delay for 1 millisecond
    cv2.waitKey(1)
