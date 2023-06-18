#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Anomaly detection with anomalib (Torch Model).

    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time
from pathlib import Path

import cv2
import torch

from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to torch model weights"
    )
    parser.add_argument(
        "--input", help="File path of input image file.", required=True, type=str
    )
    parser.add_argument(
        "--output", help="File path of output image.", required=True, type=str
    )
    args = parser.parse_args()

    # Load models.
    torch.set_grad_enabled(False)

    # Create the inferencer and visualizer.
    inferencer = TorchInferencer(path=args.weights, device="auto")
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
