#!/usr/bin/python3

import sys
import os
import argparse
import logging

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("files", type=str, nargs="*", help="Input files (images)")
    parser.add_argument(
        "--labels", type=str, help="Label file with corresponding prompts"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for dataset & annotated files"
    )
    parser.add_argument("--model_config", type=str, help="Model configuration")
    parser.add_argument("--weights_file", type=str, help="Model weights")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Set logging level to INFO"
    )

    args = parser.parse_args()

    with open(args.labels) as labels_file:
        labels = json.load(labels_file)

    images = []
    for fname in args.files:
        if os.path.isdir(fname):
            dir_images = []
            for child in os.listdir(fname):
                child_path = os.path.join(fname, child)
                if child.endswith(".jpg") or child.endswith(".png"):
                    dir_images.append(child_path)
        else:
            images.append(fname)

    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(args.model_config)
    cfg.merge_from_list(["MODEL.WEIGHT", args.weights_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])  # TODO GPU id

    glip_demo = GLIPDemo(
        cfg, min_image_size=800, confidence_threshold=0.7, show_mask_heatmaps=False
    )

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    img_folder = os.path.join(args.output_dir, "img")
    bbox_folder = os.path.join(args.output_dir, "bbox")
    train_fname = os.path.join(args.output_dir, "train.txt")
    train_content = []

    for image_fname in images:
        image = load_image(image_fname)
        img_name = os.path.splitext(os.path.basename(image_fname))[0]
        bbox_fname = os.path.join(bbox_folder, img_name + ".txt")

        for lbl in labels:
            result, top_predictions = glip_demo.run_on_web_image(
                image, labels[lbl], thresh=0.5, alpha=0.5, color=250
            )
            labels = top_predictions.get_field("labels")
            boxes = (top_predictions.bbox).to(torch.int64)

        with open(bbox_fname, "w") as bbox_file:
            for i in range(len(labels)):
                bbox_file.write(
                    "%d %d %d %d %d\n"
                    % (int(labels[i].item()), *[int(u.item()) for u in boxes[i]])
                )

        train_content.append("%s %s" % (fname, bbox_fname))

    train_content.append("")
    with open(train_fname, "w") as train_file:
        train_file.writelines(train_content)


# ====


def load_image(filename):
    input_image = np.asarray(Image.open(filename).convert("RGB"))
    return input_image[:, :, [2, 1, 0]]


def load_image_tensor(path, pix_range=(0, 1)):
    a, b = pix_range
    factor = 255 / (b - a)
    input_image = np.asarray(Image.open(path))
    tensor_image = torch.from_numpy(input_image.copy()).to(torch.float) / factor + a
    if len(tensor_image.shape) == 2:  # greyscale
        tensor_image = tensor_image.unsqueeze(2)
    if tensor_image.shape[2] > 3:  # rgba
        tensor_image = tensor_image[:, :, :3]
    tensor_image = tensor_image.permute((2, 0, 1)).unsqueeze(0)
    return tensor_image


if __name__ == "__main__":
    main()
