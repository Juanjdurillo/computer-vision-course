# This script implements an object detection pipeline in Keras-cv using retinanet
# the script should download on a pre-trained version of the  model from keras-cv
# it should ask the user to provide the path or url of an image to be processed, and it
# should display a textual summary of the found objects, and a graphical representation
# consisting of the image and the object found


# import the necessary packages
import cv2
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import urllib.request
import urllib.error
import urllib.parse
from urllib.parse import urlparse
from urllib.request import urlopen
import keras_cv
from keras_cv import visualization
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


def parse_arguments():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="path to input image")
    ap.add_argument("-u", "--url", required=False, help="url to input image")

    return vars(ap.parse_args())

def read_image(args):
    if args["image"] is not None:
        # Check if the image path exists
        if not os.path.exists(args["image"]):
            print("Error: Image path does not exist.")
            sys.exit()

        # Read the image and print its shape
        image = cv2.imread(args["image"])
    elif args["url"] is not None:
        # Read the image from the URL and print its shape
        try:
            with urllib.request.urlopen(args["url"]) as url:
                s = url.read()
                image = cv2.imdecode(np.asarray(bytearray(s), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except:
            print("Error: Could not read image from URL.")
            sys.exit()

    else:
        print("Error: No image path or URL provided.")
        sys.exit()

    return image

args  = parse_arguments()
image = read_image(args)
# The names in this list are commonly used in object detection datasets such as the 
# PASCAL VOC dataset and the COCO dataset.
class_ids =[ 'Aeroplane',
            'Bicycle',
            'Bird',
            'Boat',
            'Bottle',
            'Bus',
            'Car',
            'Cat',
            'Chair',
            'Cow',
            'Diningtable',
            'Dog',
            'Horse',
            'Motorbike',
            'Person',
            'Pottedplant',
            'Sheep',
            'Sofa',
            'Train',
            'Tvmonitor']

class_mapping = dict(zip(range(len(class_ids)), class_ids))


# obtain the pretrain model from keras-cv
pretrained_model = keras_cv.models.RetinaNet.from_preset("retinanet_resnet50_pascalvoc",bounding_box_format="xywh")

inference_resizing = keras_cv.layers.Resizing(640,640,
                                              pad_to_aspect_ratio=True,
                                              bounding_box_format="xywh")
image_batch = inference_resizing([image])
y_pred = pretrained_model.predict(image_batch)
# The exact format of y_pred depends on the architecture of the object detection model and the 
# specific implementation of the predict() function. However, in general, y_pred is a 
# multi-dimensional array or tensor that contains the predicted output of the model.

# For EfficientNet, y_pred is a tensor that contains the predicted class probabilities 
# and bounding box coordinates for the objects detected in the input image.
# in general, y_pred is a multi-dimensional tensor with shape: 
# (batch_size, num_boxes, num_classes + 4), 
# where batch_size is the number of images in the input batch, 
# num_boxes is the maximum number of bounding boxes predicted per image 
# num_classes is the number of object classes that the model is trained to detect
#  and 4 corresponds to the four coordinates of each bounding box (i.e., x, y, width, and height).

# The first num_classes elements of the last dimension of y_pred 
# correspond to the predicted class probabilities for each bounding box. 
# The remaining four elements correspond to the predicted bounding box coordinates.
print(type(y_pred))
print(y_pred.keys())

visualization.plot_bounding_box_gallery(image_batch, 
                                        y_pred=y_pred, 
                                        value_range=(0,255),
                                        rows=1,
                                        cols=1,
                                        class_mapping=class_mapping, 
                                        font_scale=0.5,
                                        bounding_box_format="xywh")
plt.show()


prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    iou_threshold=0.5,# minimum threshold for two classes to be considered the same
    confidence_threshold=0.05,# minimum confidence for a prediction to be considered
)
pretrained_model.prediction_decoder = prediction_decoder
y_pred = pretrained_model.predict(image_batch)
# The exact format of y_pred depends on the architecture of the object detection model and the 
# specific implementation of the predict() function. However, in general, y_pred is a 
# multi-dimensional array or tensor that contains the predicted output of the model.

# For EfficientNet, y_pred is a tensor that contains the predicted class probabilities 
# and bounding box coordinates for the objects detected in the input image.
# in general, y_pred is a multi-dimensional tensor with shape: 
# (batch_size, num_boxes, num_classes + 4), 
# where batch_size is the number of images in the input batch, 
# num_boxes is the maximum number of bounding boxes predicted per image 
# num_classes is the number of object classes that the model is trained to detect
#  and 4 corresponds to the four coordinates of each bounding box (i.e., x, y, width, and height).

# The first num_classes elements of the last dimension of y_pred 
# correspond to the predicted class probabilities for each bounding box. 
# The remaining four elements correspond to the predicted bounding box coordinates.
print(type(y_pred))
print(y_pred.keys())

visualization.plot_bounding_box_gallery(image_batch, 
                                        y_pred=y_pred, 
                                        value_range=(0,255),
                                        rows=1,
                                        cols=1,
                                        class_mapping=class_mapping, 
                                        font_scale=0.5,
                                        bounding_box_format="xywh")
plt.show()







# Convert the image from BGR to RGB
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the image
plt.imshow(image)
plt.show()


