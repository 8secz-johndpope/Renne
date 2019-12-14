import torch
import cv2
import matplotlib.pyplot as plt

from openpose_cv import cvpose
from openpose_pytorch import body
from pose2seg import segout

# POSE2SEG: Pose2Seg model
# IMG: Input Image      [H, W, 3]
# POSE: Pose Points     [N, 17, 3]      Point [X, Y, V{0,1,2}]
# MASKS: Mask Array     [N, H, W]       Array Range (0-1)

# 测试模型用

SHOW_MASKS = True

def img2seg(img):
    # Detect Pose(Opencv)
    # poses = cvpose.detect_pose(img)

    # Detect Pose(Pytorch)
    poses = body.detect_pose(img)
    # Detect Segmentations using Poses
    masks = segout.pose_seg(img, poses)
    # Draw Masks
    if SHOW_MASKS:
        for mask in masks:
            print(segout.mask2poly(mask))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.imshow(mask * 128, alpha=0.5)
            plt.show()

    return masks


if __name__ == "__main__":
    IMG = cv2.imread('upload/000158.jpg')
    img2seg(IMG)
