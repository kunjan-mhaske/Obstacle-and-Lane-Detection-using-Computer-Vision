"""
This file combines the results of both lanenet model and yolo model
to detect the lanes and objects in the input video.
"""

# Import statements
import cv2
import time
import imutils
import os
import argparse
from ObstacleDetectionYOLO import ObjectDetection_YOLO
from LaneDetectionLaneNet.tools import test_lanenet as tst_ln_net
import tensorflow as tf

# Parse all input arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--input", required=True, help="input video file path")
arg_parser.add_argument("-o", "--output", default="OUT/output_video.avi", help="output video file path")
# extract the argument information in the dictionary
arg_dict = vars(arg_parser.parse_args())

# Create a video stream object for a input video file.
vid_stream = cv2.VideoCapture(arg_dict["input"])

# Initialize the video writer to write the output video
vid_writer = None

# try to determine number of frames in a video file.
try:
    if imutils.is_cv2():
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
    else:
        prop = cv2.CAP_PROP_FRAME_COUNT
    # Save the total number of frames.
    frame_count = int(vid_stream.get(prop))
    print("Video Contains {} frames.".format(frame_count))
# Handle if there occurs any exception / Error while computing the total frames.
except:
    print("Error occurred while determining total frames")
    frame_count = -1

# Initialize the current_frame_count and MAX_FRAME_LIMIT to limit the output video to a certain number of frames.
current_count = 0
MAX_FRAME_LIMIT = 300

# Create session configurations to use CPU only and create a session
sess_config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=sess_config)

# Explore each frame in the video until maximum number of frames are processed.
while current_count < MAX_FRAME_LIMIT:
    # read the frame in the video.
    (read_flag, frame) = vid_stream.read()

    # if the read operation is not successful break out of the loop. (Potentially end of the video).
    if not read_flag:
        break

    # Increment the current frame count
    current_count += 1

    # print(current_count)

    # Note the beginning count
    begin_time = time.time()
    lane_output_frame = tst_ln_net.test_lanenet(
                            weights_path="./LaneDetectionLaneNet/model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt",
                            in_image=frame, session=sess)
    object_ouput_frame = ObjectDetection_YOLO.handle_object_detection_flow(frame, lane_output_frame,
                                                                           "./ObstacleDetectionYOLO/yolo-coco")
    # Note the end time
    finish_time = time.time()

    # check if the video writer is created.
    if vid_writer is None:
        # initialize the video writer.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        vid_writer = cv2.VideoWriter(arg_dict["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # Gather and display the information for processing a single frame.
        if frame_count > 0:
            # Compute the execution time.
            elapsed_time = (finish_time - begin_time)
            # Compute the estimated time based on the total video size of MAX_FRAME_LIMIT
            estimated_time = (elapsed_time * frame_count) if frame_count <= MAX_FRAME_LIMIT \
                else (elapsed_time * MAX_FRAME_LIMIT)
            print("Time Reqired for 1 frame : {:.4f} seconds".format(elapsed_time))
            print("Total estimated time : {:.4f} seconds".format(estimated_time))

    # Write the frame on the output video using video writer.
    vid_writer.write(object_ouput_frame)

# Close all the resources.
sess.close()
vid_writer.release()
vid_stream.release()
