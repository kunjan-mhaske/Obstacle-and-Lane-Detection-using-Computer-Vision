#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Author: Mayur Sunil Jawalkar (mj8628)
        Kunjan Suresh Mhaske (km1556)

        This program tests the LaneNet model on single image
"""

import argparse
import os.path as ops
import time

import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import silence_tensorflow.auto
from LaneDetectionLaneNet.config import global_config
from LaneDetectionLaneNet.lanenet_model import lanenet
from LaneDetectionLaneNet.lanenet_model import lanenet_postprocess


CFG = global_config.cfg


def init_args():
    """
    Initialize the arguments passed while executing the program
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def args_str2bool(arg_value):
    """
    Convert the arguments to the boolean value
    :param arg_value: value of input argument
    :return: boolean value associated with the input value
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """
    Performs the min max scaling operation on the input array.
    :param input_arr: array
    :return: array after min max operations
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(weights_path, in_image=None, image_path=None, session=None):
    """
    Tests the lanenet model on the image passes as an argument.
    :param image_path: path to the input image
    :param weights_path: path to the weights of the lanenet model
    :param in_image: input image
    :return: output image with detected lanes
    """

    # t_start = time.time()
    if in_image is None:
        # make sure that the path is valid
        assert ops.exists(image_path), '{:s} not exist'.format(image_path)

        # log.info('Start reading image and preprocessing')
        # read image from that path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        image = in_image.copy()

    image_vis = image
    # Resize the image to the standard dimensions
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    # log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    # create an empty place holder of a specified size and of type float
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    # Initialize the Lanenet model
    net = lanenet.LaneNet(phase='test', net_flag='vgg')

    # Make predictions using lanenet
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    # Instantiate the postprocessor
    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    # Save the current instance
    saver = tf.train.Saver()

    # Set sess configuration
    # sess_config = tf.ConfigProto(device_count={'GPU': 0})
    # sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    # sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    # sess_config.gpu_options.allocator_type = 'BFC'

    # sess = tf.Session(config=sess_config)

    sess = session

    output_img = None

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        # t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )
        # t_cost = time.time() - t_start
        # log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )
        # mask_image = postprocess_result['mask_image']

        for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        # embedding_image = np.array(instance_seg_image[0], np.uint8)

        output_img = postprocess_result['source_image']

        # plt.figure('mask_image')
        # plt.imshow(mask_image[:, :, (2, 1, 0)])
        # plt.figure('src_image')
        # plt.imshow(image_vis[:, :, (2, 1, 0)])
        # plt.figure('instance_image')
        # plt.imshow(embedding_image[:, :, (2, 1, 0)])
        # plt.figure('binary_image')
        # plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        # plt.show()
        #
        # cv2.imwrite('instance_mask_image.png', mask_image)
        # cv2.imwrite('source_image.png', postprocess_result['source_image'])
        # cv2.imwrite('binary_mask_image.png', binary_seg_image[0] * 255)

    # sess.close()

    return output_img


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(image_path=args.image_path, weights_path=args.weights_path)
