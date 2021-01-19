import numpy as np
import cv2

# for laptops => 0, cause it has only one camera
# else put 1 or any other value for camera
capture = cv2.VideoCapture(0)
# image parameters
width = 320
height = 320
treshhold = 0.6
nms_treshhold = 0.3

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


# DETECT OBJECTS
def objDetection(output, image):
    img_height, img_width, img_channels = image.shape
    bounding_box = []
    class_ids = []
    confidence = []

    for out in output:
        for detection in out:
            scores = detection[5:]
            # find index of max value and its' confidence score/probability
            cl_id = np.argmax(scores)
            conf = scores[cl_id]
            # get values greater than treshhold
            if conf > treshhold:
                # convert from percentage to pixels of image
                w, h = int(detection[0] * img_width), int(detection[2] * img_height),
                # formula to get center point x,y => actual_width - width/2
                #                                    actual_height - height/2
                cent_x, cent_y = int((detection[0]) * img_width - w / 2), int((detection[0]) * img_height - h / 2)
                bounding_box.append([cent_x, cent_y, w, h])
                class_ids.append(cl_id)
                confidence.append(float(conf))

    # remove overlaping boxes
    keep_box_index = cv2.dnn.NMSBoxes(bounding_box, confidence, treshhold, nms_treshhold)

    for index in keep_box_index:
        index = index[0]
        box = bounding_box[index]
        # extract box parameterd => x,y, width,height
        cent_x, cent_y, width, height = box[0], box[1], box[2], box[3]
        # draw box
        # x,y centroids + corners of image + box color + thickness level
        cv2.rectangle(image, (cent_x, cent_y), (cent_x + width, cent_y + height), (34, 139, 34), 2)
        # return class name by index, return confidence score of object, detect cetroid of x and y, font, scale, color, thickness
        cv2.putText(image, f'{class_names[class_ids[index]]} {confidence[index]}', (cent_x, cent_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 139, 34), 2)


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

    objDetection(output, image)

    # show image we want to display
    cv2.imshow('Image', image)
    # delay for 1 millisecond
    cv2.waitKey(1)
