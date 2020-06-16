#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mayur Sunil Jawalkar (mj8628)
        Kunjan Suresh Mhaske (km1556)

Evaluate lanenet model on tusimple lane dataset
"""
import argparse
import glob
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf
import tqdm

from LaneDetectionLaneNet.config import global_config
from LaneDetectionLaneNet.lanenet_model import lanenet
from LaneDetectionLaneNet.lanenet_model import lanenet_postprocess

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='The source tusimple lane test data dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--save_dir', type=str, help='The test output save root dir')

    return parser.parse_args()


def test_lanenet_batch(src_dir, weights_path, save_dir):
    """

    :param src_dir:
    :param weights_path:
    :param save_dir:
    :return:
    """
    assert ops.exists(src_dir), '{:s} not exist'.format(src_dir)

    os.makedirs(save_dir, exist_ok=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        image_list = glob.glob('{:s}/**/*.jpg'.format(src_dir), recursive=True)
        avg_time_cost = []

        vid_writer = None
        # frame_count = 0
        # ignore_count_1 = 0
        for index, image_path in tqdm.tqdm(enumerate(image_list), total=len(image_list)):

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0


            t_start = time.time()
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]}
            )
            avg_time_cost.append(time.time() - t_start)

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )

            if index % 100 == 0:
                log.info('Mean inference time every single image: {:.5f}s'.format(np.mean(avg_time_cost)))
                avg_time_cost.clear()

            input_image_dir = ops.split(image_path.split('clips')[1])[0][1:]
            input_image_name = ops.split(image_path)[1]
            output_image_dir = ops.join(save_dir, input_image_dir)
            os.makedirs(output_image_dir, exist_ok=True)
            output_image_path = ops.join(output_image_dir, input_image_name)
            if ops.exists(output_image_path):
                continue
            # print(output_image_path)

            try:
                cv2.imwrite(output_image_path, postprocess_result['source_image'])
            except:
                continue
    vid_writer.release()
    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet_batch(
        src_dir=args.image_dir,
        weights_path=args.weights_path,
        save_dir=args.save_dir
    )
