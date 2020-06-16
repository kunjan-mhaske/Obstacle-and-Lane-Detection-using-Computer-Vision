"""
Author: Mayur Sunil Jawalkar (mj8628)
        Kunjan Suresh Mhaske (km1556)

This program detects the objects in the image using the YOLO model.
"""

# USAGE
# python ObjectDetection_YOLO.py --image images/0.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# Declare Global Variables
global ARG_DICT
global COLORS
global LABELS

# Initialize the global variables
ARG_DICT = dict()
COLORS = list()
LABELS = list()


def set_input_arguments(in_img, yolo_dir, confidence, threshold):
    """
    Extract the argument information in the dictionary
    """
    globals()  # Access global variables
    ARG_DICT.clear()  # Clear the pre-existing content
    ARG_DICT['input_img'] = in_img
    ARG_DICT['yolo'] = yolo_dir
    ARG_DICT['confidence'] = confidence
    ARG_DICT['threshold'] = threshold


def get_labels_from_yolo_model():
    """
    Open the label file and save labels for the prediction classes into a list
    """
    globals()  # Access global variables
    LABELS.clear()  # Clear the pre-existing content
    # save the path for the labels file
    labelsPath = os.path.sep.join([ARG_DICT["yolo"], "coco.names"])
    # Read the file and save the content in the list
    LABELS.extend(open(labelsPath).read().strip().split("\n"))


def set_colors_for_labels():
    """
    Define random color for each of the labelled class.
    """
    globals()  # Access global variables
    COLORS.clear()  # Clear the pre-existing content
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS.extend(np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8"))


def get_loaded_yolo_model():
    globals()  # Access global variables
    # Extract the complete path of the weights and configuration file of a trained yolo model.
    weightsPath = os.path.sep.join([ARG_DICT["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([ARG_DICT["yolo"], "yolov3.cfg"])

    # load the yolo network using the weights and configuration file of a trained model.
    neural_network = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return neural_network


def get_bb_and_confidence_for_detections(layerOutputs, Wt, Ht):
    """
    Get the bounding boxes, confidences and classIDs for the detected objects from the layerOutputs.
    """
    # Initialize the lists to store all bounding boxes, confidence levels, and class IDs.
    boxes = []
    confidences = []
    classIDs = []

    # Iterate over each output from the layer output generated after forward pass.
    for output in layerOutputs:
        # Iterate over each detection from each output.
        for detection in output:

            scores = detection[5:]  # Extract scores
            classID = np.argmax(scores)  # Extract class ID
            confidence = scores[classID]  # Extract Confidence or probability

            # check if the confidence level is above confidence threshold.
            if confidence > ARG_DICT["confidence"]:
                # Create the bounding box for the detected objects.
                bounding_box = detection[0:4] * np.array([Wt, Ht, Wt, Ht])

                # Extract the center co-ordinates, height and width of the bounding box on the frame.
                (centerX, centerY, width, height) = bounding_box.astype("int")

                # Extract the top and bottom co-ordinates of bounding box from the above information.
                x_box = int(centerX - (width / 2))
                y_box = int(centerY - (height / 2))

                # Append the bounding_box, confidence, and classID in the respective lists.
                boxes.append([x_box, y_box, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    return boxes, confidences, classIDs


def object_detection():
    """
    This function performs the object detection using YOLO.
    """
    # load the yolo network using the weights and configuration file of a trained model.
    neural_net = get_loaded_yolo_model()

    # Extract the input image
    image = ARG_DICT['input_img']

    # Extract the shape or dimensions of the     image.
    (Ht, Wt) = image.shape[:2]

    # Determine the layers in the network
    layer_names = neural_net.getLayerNames()
    # Determine the names of only output layers.
    layer_names = [layer_names[i[0] - 1] for i in neural_net.getUnconnectedOutLayers()]

    # Construct a blob from a input frame.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # use this blob as an input to the neural network
    neural_net.setInput(blob)
    # Perform a forward pass of the YOLO object detector.
    layerOutputs = neural_net.forward(layer_names)

    # Get the information(bounidng box, confidence and ID) about the considerable detections to mark on  the image
    boxes, confidences, classIDs = get_bb_and_confidence_for_detections(layerOutputs, Wt, Ht)

    # Perform the non-maximum suppression to suppress the weak, overlapping bounding boxes.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, ARG_DICT["confidence"], ARG_DICT["threshold"])
    return idxs, boxes, confidences, classIDs


def mark_prediction_results_on_image(op_image, idxs, boxes, confidences, classIDs):
    """
    This function adds the prediction results on the output image.
    """
    globals()  # Access global variables
    # check if there is at-least one bounding box.
    if len(idxs) > 0:
        # iterate over each index
        for i in idxs.flatten():
            # extract the coordinates for bounding box.
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box with the label on the output image.
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(op_image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(op_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return op_image


def handle_object_detection_flow(in_img, op_img, yolo_dir, confidence=0.5, threshold=0.3):
    """
    This function handles the flow of the object detection using YOLO.
    """
    # Set input arguments
    set_input_arguments(in_img, yolo_dir, confidence, threshold)
    # Extract the labels from the trained model
    get_labels_from_yolo_model()
    # Associate the color to each label.
    set_colors_for_labels()
    # Perform object detection
    indexes, bounding_boxes, confidences, classIDs = object_detection()
    # Add the results of the detected objects on the output image.
    op_img = mark_prediction_results_on_image(op_img, indexes, bounding_boxes, confidences, classIDs)
    return op_img


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())
    in_image = cv2.imread(args['image'])

    op_image = handle_object_detection_flow(in_image, in_image, args['yolo'], args['confidence'], args['threshold'])
    cv2.imshow("Output", op_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
