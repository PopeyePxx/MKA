import os
import argparse
from tqdm import tqdm
from PIL import Image
import cv2
import torch
import numpy as np
from models.keypoint import Net as model
from utils.viz import viz_pred_test_kp
from utils.util import set_seed, process_gt, normalize_map
from sklearn.metrics import precision_score, recall_score
from utils.evaluation import cal_kl, cal_sim, cal_nss
import re
parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/')
parser.add_argument('--model_file', type=str, default= '/.pth')
parser.add_argument('--save_path', type=str, default='./save_preds')
parser.add_argument("--divide", type=str, default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=448)
parser.add_argument('--resize_size', type=int, default=512)
#### test
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=True)

args = parser.parse_args()
torch.cuda.set_device('cuda:' + args.gpu)
if args.divide == "Seen":
    aff_list = ['hold', "press", "click", "clamp", "grip", "open"]
else:
    aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                "swing", "take_photo", "throw", "type_on", "wash"]

if args.divide == "Seen":
    args.num_classes = 6
else:
    args.num_classes = 25

args.test_root = os.path.join(args.data_root, "Seen", "testset", "ego")
args.mask_root = args.test_root

if args.viz:
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

def normalize_map(atten_map, crop_size):
    atten_map = cv2.resize(atten_map, dsize=(crop_size, crop_size))
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)
    return atten_norm

if __name__ == '__main__':
    set_seed(seed=0)
    from data.datatest_func import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)
    model = model(aff_classes=args.num_classes)  # yf_func
    model = model.cuda()
    model.eval()

    KLs = []
    SIM = []
    NSS = []
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    model.load_state_dict(torch.load(args.model_file))

    GT_path = args.divide + "_gt.t7"
    if not os.path.exists(GT_path):
        process_gt(args)
    # GT_masks = torch.load(args.divide + "_gt.t7")
    GT_masks = torch.load('/home/yf/funcgra/Seen_vcr_gt.t7')

    for step, (image, label, mask_path, point) in enumerate(tqdm(TestLoader)):
        names = mask_path[0].split("/")
        bianhao = names[-1].split('.')
        key = names[-3] + "_" + names[-2] + "_" + bianhao[-2] + "_heatmap." + bianhao[-1]
        img_name = key.split(".")[0]
        tensor_x, tensor_y = point
        points_as_tensors = [torch.stack([x, y]) for x, y in zip(tensor_x, tensor_y)]
        ego_pred, kp = model.func_test_forward(image.cuda(), label.long().cuda(), points_as_tensors) # 3, h,w
        ego_pred0 = np.array(ego_pred.squeeze().data.cpu())
        ego_pred1 = normalize_map(ego_pred0, args.crop_size)
        # # ---------------yf--------------------- #

        names = re.split(r'[/.]', mask_path[0])
        key = names[-4] + "_" + names[-3] + "_" + names[-2] + "_heatmap." + names[-1]
        GT_mask = GT_masks[key]
        GT_mask = GT_mask / 255.0

        GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))

        kld, sim, nss = cal_kl(ego_pred1, GT_mask), cal_sim(ego_pred1, GT_mask), cal_nss(ego_pred1, GT_mask)
        KLs.append(kld)
        SIM.append(sim)
        NSS.append(nss)

        if args.viz:
            viz_pred_test_kp(args, image, ego_pred1, GT_mask, aff_list, label, img_name, kp)

    mKLD = sum(KLs) / len(KLs)
    mSIM = sum(SIM) / len(SIM)
    mNSS = sum(NSS) / len(NSS)

    print(f"KLD = {round(mKLD, 3)}\nSIM = {round(mSIM, 3)}\nNSS = {round(mNSS, 3)}")
        # -----------------metric of grasp type---------------#

    print('ddd')
