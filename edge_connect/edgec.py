import numpy as np
import torch
import cv2
from skimage.feature import canny

from edge_connect.src import config, models

def edge_init():
    configs = config.Config('models/config.yml')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        configs['DEVICE'] = torch.device("cuda")
    else:
        configs['DEVICE'] = torch.device("cpu")
    configs['PATH'] = './models'
    edge_model = models.EdgeModel(configs).to(configs['DEVICE'])
    inpaint_model = models.InpaintingModel(configs).to(configs['DEVICE'])

    edge_model.load()
    inpaint_model.load()
    edge_model.eval()
    inpaint_model.eval()
    return edge_model, inpaint_model

def inpaint(img_ori, mask_ori, edge_model, inpaint_model):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img, img_gray, mask, edge = preprocess(img_ori, mask_ori)
    img = torch.unsqueeze(torch.from_numpy(img).float(), 0)
    img_gray = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(img_gray).float(), 0), 0).to(device)
    mask = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(mask).float(), 0), 0).to(device)
    edge = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(edge).float(), 0), 0).to(device)

    edge = edge_model(img_gray, edge, mask).detach()
    outputs = inpaint_model(img.to(device), edge, mask)
    img_out = (outputs * 255.0).int().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8).squeeze()
    img_out = cv2.resize(cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR), (img_ori.shape[1], img_ori.shape[0]))
    mask_ori = (mask_ori > 0).astype(np.uint8)[:, :, np.newaxis]
    img_out = (img_out * mask_ori) + (img_ori * (1 - mask_ori))

    return img_out



def preprocess(img_ori, mask_ori):
    # img = cv2.imread(img_path)

    img = cv2.resize(img_ori, (512, 512), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask_ori, (512, 512), interpolation=cv2.INTER_AREA)
    mask = (mask > 0)
    edge = canny(img_gray, sigma=2, mask=~mask).astype(np.float)

    return img.transpose(2, 0, 1).astype(np.float) / 255.0, img_gray.astype(np.float) / 255.0, mask.astype(np.float), edge.astype(np.float)



if __name__ == "__main__":
    EDGE, INPAINT = edge_init()
    IMG = cv2.imread('test/street1.jpg')
    MASK = cv2.imread('test/737403967419482656_edmask.jpg', cv2.IMREAD_GRAYSCALE)
    inpaint(IMG, MASK, EDGE, INPAINT)
