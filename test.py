import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
from torchvision import transforms
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, default='./PolypPVT.pth')
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolypPVT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_path = './input/'
    save_path = './result_map/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name in os.listdir(data_path):
        _images = os.path.join(data_path, name)
        _save = os.path.join(save_path, name.replace('.jpg', '.png'))
        img = cv2.imread(_images)
        img_size = img.shape

        image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        images = transform(image).unsqueeze(0).to(device).cuda()
        P1, P2 = model(images)
        P = (P1 + P2)
        res = P.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(_save, res * 255)
        contours = cv2.findContours(res.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]

        for contour in contours:
            cv2.drawContours(image, contour, -1, (255, 0, 0), 3)
        image = cv2.resize(image, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(_save, image)
