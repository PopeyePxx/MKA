import os
import sys
import time
import shutil
import logging
import argparse
import torch
import torch.nn as nn
from models.keypoint_vcr_fuse_clip_sla_sam import Net as model  # yf
from utils.evaluation import cal_kl, cal_sim, cal_nss
from utils.viz import viz_pred_test
from utils.util import set_seed, process_gt, normalize_map, get_optimizer
import numpy as np
import re
import cv2

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='/data1/yf/yinshi/')
parser.add_argument('--model_file', type=str, default='/home/yf/code/funcgra/save_models/20241112_152401/aff_model_epoch_1.pth')
parser.add_argument('--save_root', type=str, default='save_models')
parser.add_argument("--divide", type=str, default="Seen")
##  image
parser.add_argument('--crop_size', type=int, default=448)
parser.add_argument('--resize_size', type=int, default=512)
##  dataloader
parser.add_argument('--num_workers', type=int, default=1)
##  train
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--show_step', type=int, default=10)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', default=False)
parser.add_argument('--proto_weight', default=0.5)
parser.add_argument('--jacobian_weight', default=0.3)
parser.add_argument('--cors_weight', default=0.2)

#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=1)

args = parser.parse_args()
torch.cuda.set_device('cuda:' + args.gpu)
lr = args.lr

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

args.exocentric_root = os.path.join(args.data_root, args.divide, "trainset", "exocentric")
args.egocentric_root = os.path.join(args.data_root, args.divide, "trainset", "egocentric")
args.test_root = os.path.join(args.data_root, args.divide, "testset", "egocentric0")
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
args.save_path = os.path.join(args.save_root, time_str)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
dict_args = vars(args)

shutil.copy('./models/keypoint_vcr_fuse_clip_sla_sam.py', args.save_path)
shutil.copy('./train_vcr_fuse.py', args.save_path)

str_1 = ""
for key, value in dict_args.items():
    str_1 += key + "=" + str(value) + "\n"

logging.basicConfig(filename='%s/run.log' % args.save_path, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info(str_1)

if __name__ == '__main__':

    set_seed(seed=0)

    from data.datatrain import TrainData

    trainset = TrainData(exocentric_root=args.exocentric_root,
                         egocentric_root=args.egocentric_root,
                         resize_size=args.resize_size,
                         crop_size=args.crop_size, divide=args.divide)

    TrainLoader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    from data.datatest_func import TestData

    testset = TestData(image_root=args.test_root,
                       crop_size=args.crop_size,
                       divide=args.divide, mask_root=args.mask_root)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model(aff_classes=args.num_classes)
    model = model.cuda()

    # model.load_state_dict(torch.load(args.model_file))

    model.train()

    optimizer, scheduler = get_optimizer(model, args)
    best_kld = 1000
    print('Train begining!')
    for epoch in range(args.epochs):
        model.train()
        logger.info('Start epoch %d/%d, LR = %f' % (epoch + 1, args.epochs, scheduler.get_last_lr()[0]))

        for step, (
        exocentric_image, egocentric_image, aff_label, point, exocentric_image_path, egocentric_image_path) in enumerate(TrainLoader):
            # Move data to GPU
            aff_label = aff_label.cuda().long()
            exo = exocentric_image.cuda()
            ego = egocentric_image.cuda()
            num_exo = exo.shape[1]
            tensor_x, tensor_y = point
            points_as_tensors = [torch.stack([x, y]) for x, y in zip(tensor_x, tensor_y)]
            loss_proto_global, exo_aff_logits = model(exo, ego, aff_label, points_as_tensors,
                                                      (epoch, args.warm_epoch))
            exo_aff_loss = torch.zeros(1).cuda()
            for n in range(num_exo):
                a = exo_aff_logits[:, n]  # 16,6
                exo_aff_loss += nn.CrossEntropyLoss().cuda()(exo_aff_logits[:, n], aff_label)

            exo_aff_loss /= num_exo
            # Compute loss
            loss_dict = {'loss_proto_global': loss_proto_global,
                         'exo_aff_loss': exo_aff_loss}
            loss = sum(loss_dict.values())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training info
            if (step + 1) % args.show_step == 0:
                log_str = 'epoch: %d/%d + %d/%d | ' % (epoch + 1, args.epochs, step + 1, len(TrainLoader))
                log_str += ' | '.join(['%s: %.3f' % (k, v.item()) for k, v in loss_dict.items()])
                logger.info(log_str)

        # Scheduler step
        scheduler.step()
        KLs = []
        SIM = []
        NSS = []
        model.eval()
        GT_path = args.divide + "_gt.t7"
        if not os.path.exists(GT_path):
            process_gt(args)
        GT_masks = torch.load('/home/yf/funcgra/Seen_vcr_gt.t7')

        for step, (image, label, mask_path, point) in enumerate(TestLoader):
            names = mask_path[0].split("/")
            bianhao = names[-1].split('.')
            key = names[-3] + "_" + names[-2] + "_" + bianhao[-2] + "_heatmap." + bianhao[-1]
            img_name = key.split(".")[0]
            tensor_x, tensor_y = point
            points_as_tensors = [torch.stack([x, y]) for x, y in zip(tensor_x, tensor_y)]
            ego_pred = model.func_test_forward(image.cuda(), label.long().cuda(), points_as_tensors)  # 3, h,w
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
                viz_pred_test(args, image, ego_pred1, GT_mask, aff_list, label, img_name)

        mKLD = sum(KLs) / len(KLs)
        mSIM = sum(SIM) / len(SIM)
        mNSS = sum(NSS) / len(NSS)
        logger.info(
            "epoch=" + str(epoch + 1) + " mKLD = " + str(round(mKLD, 3))
            + " mSIM = " + str(round(mSIM, 3)) + " mNSS = " + str(round(mNSS, 3))
            + " bestKLD = " + str(round(best_kld, 3)))

        if mKLD < best_kld:
            best_kld = mKLD
            model_name = 'best_aff_model_' + str(epoch + 1) + '_' + str(round(best_kld, 3)) \
                         + '_' + str(round(mSIM, 3)) \
                         + '_' + str(round(mNSS, 3)) \
                         + '.pth'
            torch.save(model.state_dict(), os.path.join(args.save_path, model_name))
