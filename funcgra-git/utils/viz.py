import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from utils.util import normalize_map, overlay_mask, overlay_mask_yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端


# visualize the prediction of the first batch
def viz_pred_train(args, ego, exo, masks, aff_list, aff_label, epoch, step):
    # mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)
    # std = torch.as_tensor([0.228, 0.223, 0.229], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)
    #
    # ego_0 = ego[0].squeeze(0) * std + mean
    # ego_0 = ego_0.detach().cpu().numpy() * 255
    # ego_0 = Image.fromarray(ego_0.transpose(1, 2, 0).astype(np.uint8))
    #
    # exo_img = []
    # num_exo = exo.shape[1]
    # for i in range(num_exo):
    #     name = 'exo_' + str(i)
    #     locals()[name] = exo[0][i].squeeze(0) * std + mean
    #     locals()[name] = locals()[name].detach().cpu().numpy() * 255
    #     locals()[name] = Image.fromarray(locals()[name].transpose(1, 2, 0).astype(np.uint8))
    #     exo_img.append(locals()[name])
    #
    # exo_cam = masks['exo_aff'][0]
    #
    # sim_maps, exo_sim_maps, part_score, ego_pred = masks['pred']  # ego_pred = gt_ego_cam
    # num_clu = sim_maps.shape[1]
    # part_score = np.array(part_score[0].squeeze().data.cpu())
    #
    # ego_pred = np.array(ego_pred[0].squeeze().data.cpu())
    # ego_pred = normalize_map(ego_pred, args.crop_size)
    # ego_pred = Image.fromarray(ego_pred)
    # ego_pred = overlay_mask(ego_0, ego_pred, alpha=0.5)
    #
    # ego_sam = masks['ego_sam']
    # ego_sam = np.array(ego_sam[0].squeeze().data.cpu())
    # ego_sam = normalize_map(ego_sam, args.crop_size)
    # ego_sam = Image.fromarray(ego_sam)
    # ego_sam = overlay_mask(ego_0, ego_sam, alpha=0.1)
    #
    # aff_str = aff_list[aff_label[0].item()]
    #
    # for i in range(num_exo):
    #     name = 'exo_aff' + str(i)
    #     locals()[name] = np.array(exo_cam[i].squeeze().data.cpu())
    #     locals()[name] = normalize_map(locals()[name], args.crop_size)
    #     locals()[name] = Image.fromarray(locals()[name])
    #     locals()[name] = overlay_mask(exo_img[i], locals()[name], alpha=0.5)

    # for i in range(num_clu):
    #     name = 'sim_map' + str(i)
    #     locals()[name] = np.array(sim_maps[0][i].squeeze().data.cpu())
    #     locals()[name] = normalize_map(locals()[name], args.crop_size)
    #     locals()[name] = Image.fromarray(locals()[name])
    #     locals()[name] = overlay_mask(ego_0, locals()[name], alpha=0.5)
    #
    #     # Similarity maps for the first exocentric image
    #     name = 'exo_sim_map' + str(i)
    #     locals()[name] = np.array(exo_sim_maps[0, 0][i].squeeze().data.cpu())
    #     locals()[name] = normalize_map(locals()[name], args.crop_size)
    #     locals()[name] = Image.fromarray(locals()[name])
    #     locals()[name] = overlay_mask(locals()['exo_' + str(0)], locals()[name], alpha=0.5)

    # Exo&Ego plots
    # fig, ax = plt.subplots(4, max(num_clu, num_exo), figsize=(16, 16))
    # for axi in ax.ravel():
    #     axi.set_axis_off()
    # for k in range(num_exo):
    #     ax[0, k].imshow(eval('exo_aff' + str(k)))
    #     ax[0, k].set_title("exo_" + aff_str)
    # for k in range(num_clu):
    #     ax[1, k].imshow(eval('sim_map' + str(k)))
    #     ax[1, k].set_title('PartIoU_' + str(round(part_score[k], 2)))
    #     ax[2, k].imshow(eval('exo_sim_map' + str(k)))
    #     ax[2, k].set_title('sim_map_' + str(k))
    # ax[3, 0].imshow(ego_pred)
    # ax[3, 0].set_title(aff_str)
    # ax[3, 1].imshow(ego_sam)
    # ax[3, 1].set_title('Saliency')
    #
    # os.makedirs(os.path.join(args.save_path, 'viz_train'), exist_ok=True)
    # fig_name = os.path.join(args.save_path, 'viz_train', 'cam_' + str(epoch) + '_' + str(step) + '.jpg')
    # plt.tight_layout()
    # plt.savefig(fig_name)
    # plt.close()


    mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)
    std = torch.as_tensor([0.228, 0.223, 0.229], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)

    # 处理 Ego 图像
    ego_0 = ego[0].squeeze(0) * std + mean
    ego_0 = ego_0.detach().cpu().numpy() * 255
    ego_0 = Image.fromarray(ego_0.transpose(1, 2, 0).astype(np.uint8))

    exo_img = []
    num_exo = exo.shape[1]
    for i in range(num_exo):
        exo_i = exo[0][i].squeeze(0) * std + mean
        exo_i = exo_i.detach().cpu().numpy() * 255
        exo_i = Image.fromarray(exo_i.transpose(1, 2, 0).astype(np.uint8))
        exo_img.append(exo_i)

    exo_cam = masks['exo_aff'][0][0]
    part_proto = masks['exo_aff'][1][0]
    part_func_proto = masks['exo_aff'][2]
    ego_pred = masks['pred'][3]  # ego_pred = gt_ego_cam

    # 处理 Ego 预测图像
    ego_pred = np.array(ego_pred[0].squeeze().data.cpu())
    ego_pred = normalize_map(ego_pred, args.crop_size)
    ego_pred = Image.fromarray(ego_pred, mode='L')
    ego_pred = overlay_mask(ego_0, ego_pred, alpha=0.5)

    # 处理 Exo 功能图像并叠加
    exo_aff_imgs = []
    part_proto_imgs = []
    part_func_proto_imgs = []

    for i in range(num_exo):
        exo_aff = np.array(exo_cam[i].squeeze().data.cpu())
        exo_aff = normalize_map(exo_aff, args.crop_size)
        exo_aff = Image.fromarray(exo_aff)
        exo_aff = overlay_mask(exo_img[i], exo_aff, alpha=0.5)
        exo_aff_imgs.append(exo_aff)

        if part_proto.numel() > 0:
            part_proto_img = np.array(part_proto.squeeze().data.cpu())
            part_proto_img = normalize_map(part_proto_img, args.crop_size)
            part_proto_img = (part_proto_img * 255).astype(np.uint8)
            part_proto_img = Image.fromarray(part_proto_img, mode='L')
            part_proto_img = overlay_mask(exo_img[i], part_proto_img, alpha=0.5)
            part_proto_imgs.append(part_proto_img)
        else:
            part_proto_imgs.append(None)

        if len(part_func_proto) > 0:
            part_func_proto_img = np.array(part_func_proto[0][i].squeeze().data.cpu())
            part_func_proto_img = normalize_map(part_func_proto_img, args.crop_size)
            part_func_proto_img = Image.fromarray(part_func_proto_img, mode='L')
            part_func_proto_img = overlay_mask(exo_img[i], part_func_proto_img, alpha=0.5)
            part_func_proto_imgs.append(part_func_proto_img)
        else:
            part_func_proto_imgs.append(None)

    # 绘制图像
    fig, ax = plt.subplots(3, num_exo + 1, figsize=(16, 12))
    for axi in ax.ravel():
        axi.set_axis_off()

    # 显示原始 Exo 和 Ego 图像
    for k in range(num_exo):
        ax[0, k].imshow(exo_img[k])
        ax[0, k].set_title(f"Original Exo {k}")
    ax[0, num_exo].imshow(ego_0)
    ax[0, num_exo].set_title("Original Ego")

    # 显示叠加的 Exo 功能图像和 Ego 预测图像
    for k in range(num_exo):
        ax[1, k].imshow(exo_aff_imgs[k])
        ax[1, k].set_title(f"Exo Cam {k}")
    ax[1, num_exo].imshow(ego_pred)
    ax[1, num_exo].set_title("Ego Prediction")

    # 显示 part_proto 和 part_func_proto 叠加图像
    for k in range(num_exo):
        if part_proto_imgs[k] is not None:
            ax[2, k].imshow(part_proto_imgs[k])
            ax[2, k].set_title(f"Part Proto {k}")
        if part_func_proto_imgs[k] is not None:
            ax[2, k].imshow(part_func_proto_imgs[k])
            ax[2, k].set_title(f"Part Func Proto {k}")

    os.makedirs(os.path.join(args.save_path, 'viz_train'), exist_ok=True)
    fig_name = os.path.join(args.save_path, 'viz_train', f'cam_{epoch}_{step}.jpg')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

    # mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)
    # std = torch.as_tensor([0.228, 0.223, 0.229], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)
    #
    # # 处理 Ego 图像
    # ego_0 = ego[0].squeeze(0) * std + mean
    # ego_0 = ego_0.detach().cpu().numpy() * 255
    # ego_0 = Image.fromarray(ego_0.transpose(1, 2, 0).astype(np.uint8))
    #
    # exo_img = []
    # num_exo = exo.shape[1]
    # for i in range(num_exo):
    #     exo_i = exo[0][i].squeeze(0) * std + mean
    #     exo_i = exo_i.detach().cpu().numpy() * 255
    #     exo_i = Image.fromarray(exo_i.transpose(1, 2, 0).astype(np.uint8))
    #     exo_img.append(exo_i)
    #
    # exo_cam = masks['exo_aff'][0][0]
    # ego_pred = masks['pred'][3]  # ego_pred = gt_ego_cam
    #
    # # 处理 Ego 预测图像
    # ego_pred = np.array(ego_pred[0].squeeze().data.cpu())
    # ego_pred = normalize_map(ego_pred, args.crop_size)
    # ego_pred = Image.fromarray(ego_pred)
    # ego_pred = overlay_mask(ego_0, ego_pred, alpha=0.5)
    #
    # # 处理 Exo 功能图像并叠加
    # exo_aff_imgs = []
    # for i in range(num_exo):
    #     exo_aff = np.array(exo_cam[i].squeeze().data.cpu())
    #     exo_aff = normalize_map(exo_aff, args.crop_size)
    #     exo_aff = Image.fromarray(exo_aff)
    #     exo_aff = overlay_mask(exo_img[i], exo_aff, alpha=0.5)
    #     exo_aff_imgs.append(exo_aff)
    #
    # # 绘制图像
    # fig, ax = plt.subplots(2, num_exo + 1, figsize=(16, 8))
    # for axi in ax.ravel():
    #     axi.set_axis_off()
    #
    # # 显示原始 Exo 和 Ego 图像
    # for k in range(num_exo):
    #     ax[0, k].imshow(exo_img[k])
    #     ax[0, k].set_title(f"Original Exo {k}")
    # ax[0, num_exo].imshow(ego_0)
    # ax[0, num_exo].set_title("Original Ego")
    #
    # # 显示叠加的 Exo 功能图像和 Ego 预测图像
    # for k in range(num_exo):
    #     ax[1, k].imshow(exo_aff_imgs[k])
    #     ax[1, k].set_title(f"Exo Cam {k}")
    # ax[1, num_exo].imshow(ego_pred)
    # ax[1, num_exo].set_title("Ego Prediction")
    #
    # os.makedirs(os.path.join(args.save_path, 'viz_train'), exist_ok=True)
    # fig_name = os.path.join(args.save_path, 'viz_train', f'cam_{epoch}_{step}.jpg')
    # plt.tight_layout()
    # plt.savefig(fig_name)
    # plt.close()


def viz_pred_test_3(args, image, ego_preds, aff_list, aff_label, img_name, epoch=None):
    mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.228, 0.223, 0.229], dtype=image.dtype, device=image.device).view(-1, 1, 1)

    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    aff_str = aff_list[aff_label.item()]

    # 创建一个图形和子图
    fig, ax = plt.subplots(1, 4, figsize=(12, 6))  # 4个子图，1个原始图像和3个区域
    for axi in ax.ravel():
        axi.set_axis_off()

    ax[0].imshow(img)
    ax[0].set_title('Original Image')

    # 遍历并可视化每个区域
    for i, ego_pred in enumerate(ego_preds):
        func_ego_cam0 = Image.fromarray(ego_pred)
        func_ego_cam, center_x, center_y = overlay_mask_yf(img, func_ego_cam0, alpha=0.3)
        ax[i + 1].imshow(func_ego_cam)
        ax[i + 1].set_title(f'Area {i + 1}: {aff_str}')

    os.makedirs(os.path.join(args.save_path, 'viz_test1'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test1', "epoch" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test1', img_name + '.jpg')
    plt.savefig(fig_name)
    plt.close()


def viz_pred_test(args, image, ego_pred, GT_mask, aff_list, aff_label, img_name, kp, epoch=None):
    mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.228, 0.223, 0.229], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))


    aff_str = aff_list[aff_label.item()]

    func_ego_cam0 = Image.fromarray(ego_pred)
    func_ego_cam, center_x, center_y = overlay_mask_yf(img, func_ego_cam0, alpha=0.5)

    gt = Image.fromarray(GT_mask)
    gt_result = overlay_mask(img, gt, alpha=0.5)

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()

    ax[0].imshow(img)
    ax[0].set_title('ego')
    ax[1].imshow(func_ego_cam)
    ax[1].set_title(aff_str)
    ax[2].imshow(gt_result)
    ax[2].set_title('GT')
    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test', "epoch" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')
    plt.savefig(fig_name)
    plt.close()

def viz_pred_test_kp(args, image, ego_pred, GT_mask, aff_list, aff_label, img_name, kp, epoch=None):
    mean = torch.as_tensor([0.592, 0.558, 0.523], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.228, 0.223, 0.229], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    aff_str = aff_list[aff_label.item()]

    func_ego_cam0 = Image.fromarray(ego_pred)
    func_ego_cam, center_x, center_y = overlay_mask_yf(img, func_ego_cam0, alpha=0.5)

    gt = Image.fromarray(GT_mask)
    gt_result = overlay_mask(img, gt, alpha=0.5)

    # 获取关键点坐标并标记点
    kp = kp.squeeze(0).detach().cpu().numpy()  # 转为 (3, 2) 的 numpy 数组
    kp_x = kp[:, 1]
    kp_y = kp[:, 0]

    # 根据规则标号点
    idx_right = np.argmax(kp_x)  # 最右边的点
    remaining_idx = [i for i in range(3) if i != idx_right]
    idx_top = remaining_idx[np.argmin(kp_y[remaining_idx])]  # 左边较上面的点
    idx_bottom = remaining_idx[np.argmax(kp_y[remaining_idx])]  # 左边较下面的点
    labels = np.zeros(3, dtype=int)
    labels[idx_right] = 2
    labels[idx_top] = 0
    labels[idx_bottom] = 1

    # 绘制图片
    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()

    ax[0].imshow(img)
    ax[0].set_title('ego')

    ax[1].imshow(func_ego_cam)
    ax[1].set_title(aff_str)
    for i in range(3):
        x, y = kp_x[i], kp_y[i]
        ax[1].scatter(x, y, c='red', s=50)  # 绘制点
        ax[1].add_patch(plt.Rectangle((x - 10, y - 10), 20, 20, edgecolor='blue', facecolor='none', lw=1))  # 绘制框
        ax[1].text(x, y - 15, str(labels[i]), color='white', fontsize=8, ha='center', va='center', bbox=dict(facecolor='blue', edgecolor='none', boxstyle='round,pad=0.3'))

    ax[2].imshow(gt_result)
    ax[2].set_title('GT')

    # 保存可视化结果
    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test', "epoch" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')
    plt.savefig(fig_name)
    plt.close()
