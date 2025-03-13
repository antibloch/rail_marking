#!/usr/bin/env python
import os
import sys
import cv2
import datetime
import shutil
import matplotlib.pyplot as plt   
import numpy as np

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    curr_pth = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("-snapshot", type=str, default=f"{curr_pth}/pretrained_model.pth")
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    if os.path.exists("masks"):
        shutil.rmtree("masks")
    os.mkdir("masks")

    segmentation_handler = RailtrackSegmentationHandler(args.snapshot, BiSeNetV2Config())
    dirr = args.dir
    img_files = os.listdir(dirr)
    for img_name in img_files:
        image = cv2.imread(os.path.join(dirr, img_name))
    
        image = cv2.resize(image, (5000, 5000))
        # plt.imshow(image)
        # plt.show()
        # sdj

        mask, overlay = segmentation_handler.run(image, only_mask=False)
        plt.imshow(overlay)
        plt.show()

        mask = np.moveaxis(mask, 0, 1).astype(np.uint8)
        

        # cv2.imwrite(f"masks/ima")
        cv2.imwrite(f"masks/{img_name}", mask)


if __name__ == "__main__":
    main()
