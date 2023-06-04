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


import cv2
import numpy as np
import pyrealsense2 as rs

from pathlib import Path

import torch

from anomalib.data.utils import (
    generate_output_image_filename,
    get_image_filenames,
    read_image,
)
from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer, anomaly_map_to_color_map


CAMERA_WINDOW_NAME = "Camera View"
PREDICT_WINDOW_NAME = "Predict View"
HEATMAP_WINDOW_NAME = "Heatmap View"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to torch model weights"
    )
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        CAMERA_WINDOW_NAME,
        cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO,
    )
    cv2.moveWindow(CAMERA_WINDOW_NAME, 10, 10)
    cv2.namedWindow(
        PREDICT_WINDOW_NAME,
        cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO,
    )
    cv2.moveWindow(PREDICT_WINDOW_NAME, 10, 720)
    cv2.namedWindow(
        HEATMAP_WINDOW_NAME,
        cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO,
    )
    cv2.moveWindow(HEATMAP_WINDOW_NAME, 1280, 10)

    # Load models.
    torch.set_grad_enabled(False)

    # Create the inferencer and visualizer.
    inferencer = TorchInferencer(path=args.weights, device="auto")
    visualizer = Visualizer(mode="full", task="classification")

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
        cv2.imshow(CAMERA_WINDOW_NAME, display_image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

        elif key == ord(" "):
            cut_image = color_image[232:488, 512:768]
            predict_image = cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB)

            print("[INFO] Starting Inference ...")
            start = time.perf_counter()

            predictions = inferencer.predict(image=predict_image)

            inference_time = (time.perf_counter() - start) * 1000
            print("[INFO] End Inference ... : {0:.2f} ms".format(inference_time))

            # Display result image
            result_image = visualizer.visualize_image(predictions)
            result_image = cv2.resize(result_image, None, fx=0.5, fy=0.5)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(PREDICT_WINDOW_NAME, result_image)

            # Display anomaly map
            anomaly_map = np.where(
                predictions.anomaly_map < 0.3, 0, predictions.anomaly_map
            )
            anomaly_map = anomaly_map_to_color_map(anomaly_map)
            anomaly_map[np.where((anomaly_map == [0, 0, 128]).all(axis=2))] = [0, 0, 0]
            anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_RGB2BGR)
            alpha = 0.7
            blended = cv2.addWeighted(cut_image, alpha, anomaly_map, 1 - alpha, 0)
            cv2.imshow(HEATMAP_WINDOW_NAME, np.vstack([anomaly_map, blended]))

    # Stop streaming
    pipeline.stop()


if __name__ == "__main__":
    main()
