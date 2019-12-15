import numpy as np
import torch
import cv2
# from pose2seg.modeling.build_model import Pose2Seg


def pose_seg(img, poses):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = torch.load('models/pose2seg-full.pth').cuda().eval()
    else:
        model = torch.load('models/pose2seg-full.pth', map_location=torch.device('cpu')).eval()
    masks = []
    for pose in poses:
        masks.append(model([img.astype(float)], [[pose]])[0][0])
    return np.array(masks)


def mask2poly(img):
    contours, _ = cv2.findContours(
        img * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    epsilon = 0.002 * cv2.arcLength(contours, True)
    approx = cv2.approxPolyDP(contours, epsilon, True).reshape(-1, 2)
    # hull = cv2.convexHull(contours[0])
    # return str(approx.reshape(-1,2))
    return ' '.join('{:d},{:d}'.format(point[0], point[1]) for point in approx)

def mask_generate(masks, options):
    options = np.array(list(options.values())).astype(np.uint8)
    mask = np.tensordot(options, masks, 1).astype(bool).astype(np.uint8)
    return mask