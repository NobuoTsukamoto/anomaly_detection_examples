#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorFlow Lite Object detection with RealSense.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import colorsys
import os
import random
import time
import datetime


import cv2
import numpy as np
import pyrealsense2 as rs


WINDOW_NAME = "photograph"


def main():
    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 200)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    print("[INFO] Starting streaming...")
    pipeline.start(config)
    print("[INFO] Camera ready.")


    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays.
        color_image = np.asanyarray(color_frame.get_data())

        display_image = color_image.copy()
        cv2.rectangle(display_image, (512, 232), (767, 487), (255, 0, 0), thickness=3)

        # Display
        cv2.imshow(WINDOW_NAME, display_image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            dt_now = datetime.datetime.now()
            file_name = dt_now.strftime("%Y%m%d_%H%M%S%f") + ".png"
            save_image = color_image[232:488, 512:768]
            print(save_image.shape)
            cv2.imwrite(os.path.join("out", file_name), save_image)
            

    # Stop streaming
    pipeline.stop()

if __name__ == "__main__":
    main()
