#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Anomaly detection with anomalib (TensorFlow Lite Model).

    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time

import cv2

from anomalib.deploy import TFLiteInferencer
from anomalib.post_processing import Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", help="Path to torch model weights", type=str, required=True
    )
    parser.add_argument(
        "--input", help="File path of input image file.", required=True, type=str
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="256,256",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--metadata", help="File path of metadata file.", required=True, type=str
    )
    parser.add_argument(
        "--output", help="File path of output image.", required=True, type=str
    )
    args = parser.parse_args()

    # Create the inferencer and visualizer.
    input_shape = tuple(map(int, args.input_shape.split(",")))
    inferencer = TFLiteInferencer(
        path=args.weights,
        input_shape=input_shape,
        metadata=args.metadata
    )
    visualizer = Visualizer(mode="full", task="classification")

    # Read image.
    img = cv2.imread(args.input)
    predict_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # inference
    start = time.perf_counter()
    predictions = inferencer.predict(image=predict_image)
    inference_time = (time.perf_counter() - start) * 1000
    print("[INFO] End Inference ... : {0:.2f} ms".format(inference_time))

    # save output image
    result_image = visualizer.visualize_image(predictions)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_image)


if __name__ == "__main__":
    main()
