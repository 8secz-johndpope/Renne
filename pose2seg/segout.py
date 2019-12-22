import numpy as np
import torch
import cv2
# from pose2seg.modeling.build_model import Pose2Seg


def pose_seg(img, poses):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = torch.load('models/pose2seg-full.pth').cuda().eval()
    else:
        model = torch.load('models/pose2seg-full.pth',
                           map_location=torch.device('cpu')).eval()
    masks = []
    for pose in poses:
        masks.append(model([img.astype(float)], [[pose]])[0][0])
    masks = deal_masks(masks)

    masks = denoise(masks, 25)

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


def denoise(masks, size):
    """
    掩模去噪，图形学方式，腐蚀、取极大、膨胀，消除孤立点和细线
    输入：masks 掩模模型
          size  模板大小
    输出：masks 处理后的掩模
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    for maskID in range(len(masks)):
        # 腐蚀
        masks[maskID] = cv2.erode(masks[maskID], kernel)
        contours, _ = cv2.findContours(
            masks[maskID], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area = []
        # 计算每个闭合区域的面积，非极大抑制
        for contour in contours:
            area.append(cv2.contourArea(contour))
        masks[maskID] = np.zeros(masks[maskID].shape)
        cv2.drawContours(masks[maskID], contours, np.argmax(area),
                         [1, 1, 1], cv2.FILLED)
        # 膨胀
        masks[maskID] = cv2.dilate(masks[maskID], kernel).astype(np.uint8)
    return masks


def deal_masks(masks):
    lenth = len(masks)-1  # 根据掩模数进行两两消除重复
    if lenth == 0:
        return masks
    else:
        for i in range(lenth):
            for j in range(i, lenth):
                masks[i], masks[j+1] = deal_mask(masks[i], masks[j+1])

    return masks


def deal_mask(mask1, mask2):  # 消除两掩模的重叠部分
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    overlap_area = count_overlap_area(mask1, mask2)  # 计算重叠区域
    if np.sum(mask1) >= np.sum(mask2):
        mask1 = delete_overlap_area2D(mask1, overlap_area)
    else:
        mask2 = delete_overlap_area2D(mask2, overlap_area)
    return mask1.astype(np.uint8), mask2.astype(np.uint8)


def count_overlap_area(area1, area2):
    overlap_area = np.array(area1 * area2)
    return overlap_area


def delete_overlap_area2D(mask, overlap_area):
    overlap_factor = np.ones(overlap_area.shape)
    overlap_factor = overlap_factor - overlap_area
    mask = np.array(mask) * overlap_factor
    return mask
