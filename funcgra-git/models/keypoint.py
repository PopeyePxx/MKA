import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from fast_pytorch_kmeans import KMeans
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import cv2
from models.model_util import *
from models.ego_keypoint_mutilayer import KeypointProposer
from models.dino.utils import load_pretrained_weights
from models.dino import vision_transformer as vits
import yaml
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端

def get_config(config_path=None):
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_file_dir, 'configs/config.yaml')
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Net(nn.Module):
    def __init__(self, aff_classes=6):
        super(Net, self).__init__()
        self.aff_classes = aff_classes
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.vit_feat_dim = 384
        self.stride = 16
        self.patch = 16
        self.vit_model = vits.__dict__['vit_small'](patch_size=self.patch, num_classes=0)
        load_pretrained_weights(self.vit_model, '', None, 'vit_small', self.patch)
        # --- learning parameters --- #
        # 初始化mean和std
        self.mean = torch.tensor([0.592, 0.558, 0.523]).view(-1, 1, 1).cuda()
        self.std = torch.tensor([0.228, 0.223, 0.229]).view(-1, 1, 1).cuda()
        self.cel_margin = 0.05
        global_config = get_config(config_path="./configs/config.yaml")
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.get_keypoints = self.keypoint_proposer.get_keypoints
        self.keypoint_weight = nn.Parameter(torch.randn(6, 12))  # 可学习的权重

        sam_checkpoint = "/home/yf/funcgra/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        # 定义保存路径
        self.save_dir = "/home/yf/funcgra/save_seg"
        os.makedirs(self.save_dir, exist_ok=True)
        self.aff_proj = Mlp(in_features=self.vit_feat_dim, hidden_features=int(self.vit_feat_dim * 4),
                            act_layer=nn.GELU, drop=0.)
        self.aff_exo_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_fc = nn.Conv2d(self.vit_feat_dim, self.aff_classes, 1)
        self.cluster_num = 3
        self.aff_cam_thd = 0.6
        self.part_iou_thd = 0.6
        self.proj = nn.Linear(self.vit_feat_dim, self.vit_feat_dim).cuda()

    def forward(self, exo, ego, task_labels, point, epoch):
        bs, num_exo = exo.shape[0], exo.shape[1]
        exo0 = exo.flatten(0, 1)  # b*num_exo x 3 x 224 x 224
        with torch.no_grad():
            _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # 提取 EGO 特征
            _, exo_key, exo_attn = self.vit_model.get_last_key(exo0)  # 提取 EXO 特征
            ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1).detach()
            exo_desc = exo_key.permute(0, 2, 3, 1).flatten(-2, -1).detach()

        exo_proj = exo_desc[:, 1:] + self.aff_proj(exo_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)  # bs,c,28,28
        exo_desc = self._reshape_transform(exo_desc[:, 1:, :], self.patch, self.stride)  # bs*3,c,28,28
        exo_proj = self._reshape_transform(exo_proj, self.patch, self.stride)
        b, c, h, w = ego_desc.shape
        ego_cls_attn = ego_attn[:, :, 0, 1:].reshape(b, 6, h, w)
        ego_cls_attn = (ego_cls_attn > ego_cls_attn.flatten(-2, -1).mean(-1, keepdim=True).unsqueeze(-1)).float()
        head_idxs = [0, 1, 3]
        ego_sam = ego_cls_attn[:, head_idxs].mean(1)
        ego_sam = normalize_minmax(ego_sam)
        ego_sam_flat = ego_sam.flatten(-2, -1)
        # --- LOCATE --- #
        exo_proj = self.aff_exo_proj(exo_proj)
        aff_cam = self.aff_fc(exo_proj)  # b*num_exo x 36 x h x w
        aff_logits = self.gap(aff_cam).reshape(b, num_exo, self.aff_classes)
        aff_cam_re = aff_cam.reshape(b, num_exo, self.aff_classes, h, w)

        gt_aff_cam = torch.zeros(b, num_exo, h, w).cuda()
        for b_ in range(b):
            gt_aff_cam[b_, :] = aff_cam_re[b_, :, task_labels[b_]]

        # --- Clustering extracted descriptors based on CAM --- #
        ego_desc_flat = ego_desc.flatten(-2, -1)  # b x 384 x hw
        exo_desc_re_flat = exo_desc.reshape(b, num_exo, c, h, w).flatten(-2, -1)
        sim_maps = torch.zeros(b, self.cluster_num, h * w).cuda()
        exo_sim_maps = torch.zeros(b, num_exo, self.cluster_num, h * w).cuda()
        part_score = torch.zeros(b, self.cluster_num).cuda()
        part_proto = torch.zeros(b, c).cuda()
        for b_ in range(b):
            exo_aff_desc = []
            for n in range(num_exo):
                tmp_cam = gt_aff_cam[b_, n].reshape(-1)
                tmp_max, tmp_min = tmp_cam.max(), tmp_cam.min()
                tmp_cam = (tmp_cam - tmp_min) / (tmp_max - tmp_min + 1e-10)
                tmp_desc = exo_desc_re_flat[b_, n]  # c,hw
                tmp_top_desc = tmp_desc[:, torch.where(tmp_cam > self.aff_cam_thd)[0]].T  # n,c
                exo_aff_desc.append(tmp_top_desc)
            exo_aff_desc = torch.cat(exo_aff_desc, dim=0)  # (n1 + n2 + n3) x c, 168, 384

            if exo_aff_desc.shape[0] < self.cluster_num:
                continue

            kmeans = KMeans(n_clusters=self.cluster_num, mode='euclidean', max_iter=300)
            kmeans.fit_predict(exo_aff_desc.contiguous())
            clu_cens = F.normalize(kmeans.centroids, dim=1)  # 3, 384

            # save the exocentric similarity maps for visualization in training
            for n_ in range(num_exo):
                exo_sim_maps[b_, n_] = torch.mm(clu_cens, F.normalize(exo_desc_re_flat[b_, n_],
                                                                      dim=0))  # b,3,3,196,计算外在感知区域中每个像素与聚类中心之间的相似度

            # find object part prototypes and background prototypes
            sim_map = torch.mm(clu_cens, F.normalize(ego_desc_flat[b_], dim=0))  # self.cluster_num x hw
            tmp_sim_max, tmp_sim_min = torch.max(sim_map, dim=-1, keepdim=True)[0], \
                torch.min(sim_map, dim=-1, keepdim=True)[0]
            sim_map_norm = (sim_map - tmp_sim_min) / (tmp_sim_max - tmp_sim_min + 1e-12)

            sim_map_hard = (sim_map_norm > torch.mean(sim_map_norm, 1, keepdim=True)).float()
            sam_hard = (ego_sam_flat > torch.mean(ego_sam_flat, 1, keepdim=True)).float()

            inter = (sim_map_hard * sam_hard[b_]).sum(1)
            union = sim_map_hard.sum(1) + sam_hard[b_].sum() - inter
            p_score = (inter / sim_map_hard.sum(1) + sam_hard[b_].sum() / union) / 2

            sim_maps[b_] = sim_map
            part_score[b_] = p_score

            if p_score.max() < self.part_iou_thd:
                continue
            part_proto[b_] = clu_cens[torch.argmax(p_score)]  # 8,384
        # --- LOCATE --- #
        # # --- Ego 部分：生成三个定位图 ---
        N = 12  # 定义期望的最大关键点数
        sample_ego_kp_features = []
        for i in range(bs):
            ego_f = self.get_ego_mask(ego[i], point[i])  # 获取 rgb, seg
            sam_mask = ego_f['seg']  # 获取分割掩码 (H, W)
            ego_keypoints, masks, candidate_rigid_group_ids = self.get_keypoints(ego[i], sam_mask)
            ego_keypoints = torch.from_numpy(ego_keypoints).cuda()
            # 如果关键点数量小于 N，则随机选取并偏移一些距离扩充到 N 个关键点
            num_keypoints = ego_keypoints.shape[0]
            if num_keypoints < N:
                  additional_keypoints = []
                  while len(additional_keypoints) + num_keypoints < N:
                      random_idx = torch.randint(0, num_keypoints, (1,)).item()
                      offset = torch.randn(2).cuda() * 0.01  # 添加一个小的随机偏移
                      new_keypoint = ego_keypoints[random_idx] + offset
                      additional_keypoints.append(new_keypoint)
                  additional_keypoints = torch.stack(additional_keypoints, dim=0)
                  ego_keypoints = torch.cat((ego_keypoints, additional_keypoints), dim=0)  # 扩充到 N 个关键点
            # 确保最终的关键点数量为 N
            ego_keypoints = ego_keypoints[:N]  # (N, 2)

            # 计算关键点权重的 softmax
            selected_weights = self.keypoint_weight[task_labels[i]]  # (N)
            selected_weights = selected_weights.unsqueeze(-1)  # (N,1)
            # 计算加权得分，假设 ego_keypoints 的形状是 (16, 9, 2)
            weighted_scores = (ego_keypoints * selected_weights).sum(dim=-1)  # (6)
            # 选择得分最高的 3 个关键点的索引
            top_k_indices = torch.topk(weighted_scores, k=3, dim=-1).indices  # (3)
            # 根据索引选择关键点
            final_keypoints = ego_keypoints[top_k_indices]  # (3, 2)
            # 提取最终关键点特征
            ego_kp_feat = self.get_keypoint_features(ego_desc[i], final_keypoints)  # 从EGO中提取特征
            sample_ego_kp_features.append(ego_kp_feat)
        ego_kp_features = torch.stack(sample_ego_kp_features)  # b,3,c

        # --- prototype guidance loss --- #
        loss_proto_global = torch.zeros(1).cuda()
        valid_batch = 0

        if epoch[0] > epoch[1]:
            for b_ in range(bs):
                if not part_proto[b_].equal(torch.zeros(c).cuda()):
                    embedding = ego_kp_features.mean(dim=1)
                    loss_proto_global += torch.max(
                        1 - F.cosine_similarity(embedding[b_], part_proto[b_], dim=0) - self.cel_margin,
                        torch.zeros(1).cuda())
                    valid_batch += 1
            loss_proto_global = loss_proto_global / (valid_batch + 1e-15)
        return loss_proto_global, aff_logits

    @torch.no_grad()
    def func_test_forward(self, ego, aff_label, point):
        bs = ego.shape[0]
        N = 12  # 定义期望的最大关键点数
        sample_ego_kp_features = []
        for i in range(bs):
            # 每个样本的关键点特征列表
            # 获取候选关键点
            ego_f = self.get_ego_mask(ego[i], point[i])  # 获取 rgb, seg
            sam_mask = ego_f['seg']  # 获取分割掩码 (H, W)
            ego_keypoints, masks, candidate_rigid_group_ids = self.get_keypoints(ego[i], sam_mask)
            ego_keypoints = torch.from_numpy(ego_keypoints).cuda()
            # 如果关键点数量小于 N，则随机选取并偏移一些距离扩充到 N 个关键点
            num_keypoints = ego_keypoints.shape[0]
            if num_keypoints < N:
                additional_keypoints = []
                while len(additional_keypoints) + num_keypoints < N:
                    random_idx = torch.randint(0, num_keypoints, (1,)).item()
                    offset = torch.randn(2).cuda() * 0.01  # 添加一个小的随机偏移
                    new_keypoint = ego_keypoints[random_idx] + offset
                    additional_keypoints.append(new_keypoint)
                additional_keypoints = torch.stack(additional_keypoints, dim=0)
                ego_keypoints = torch.cat((ego_keypoints, additional_keypoints), dim=0)  # 扩充到 N 个关键点
            # 确保最终的关键点数量为 N
            ego_keypoints = ego_keypoints[:N]  # (N, 2)

            # 遍历掩码中的每个类别（跳过类别0）
            # 计算关键点权重的 softmax
            selected_weights = self.keypoint_weight[aff_label[i]]  # (6)
            selected_weights = selected_weights.unsqueeze(-1)  # (6,1)
            # 计算加权得分，假设 ego_keypoints 的形状是 (16, 9, 2)
            weighted_scores = (ego_keypoints * selected_weights).sum(dim=-1)  # (6)
            # 选择得分最高的 3 个关键点的索引
            top_k_indices = torch.topk(weighted_scores, k=3, dim=-1).indices  # (3)
            # 根据索引选择关键点
            final_keypoints = ego_keypoints[top_k_indices]  # (3, 2)
            heatmap = self._project_keypoints_to_img(final_keypoints)
            heatmap = torch.tensor(heatmap)
            # 将 final_keypoints 添加到列表中
            sample_ego_kp_features.append(heatmap)

        all_affgrounding = torch.stack(sample_ego_kp_features)
        return all_affgrounding

    def create_cluster_color_map(self):
        # 这里定义如何生成你的 cluster_color_map
        # 例如，返回一个随机生成的颜色映射
        return np.random.rand(256, 256, 3) * 25

    def generate_irregular_polygon(self, center, num_vertices=8, radius=20):
        """生成一个不规则多边形的顶点"""
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False) + np.random.uniform(-0.2, 0.2, num_vertices)
        r = radius + np.random.uniform(-5, 5, num_vertices)  # 添加一些随机半径变化
        points = np.array([
            (int(center[1] + r[i] * np.cos(angles[i])), int(center[0] + r[i] * np.sin(angles[i])))
            for i in range(num_vertices)
        ])
        return points

    def create_filled_map(self,shapes, image_size):
        """根据多边形顶点填充图形内部，顶点内的值为1，其他为0"""
        filled_map = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)  # 注意numpy数组是先行后列（先y后x）

        for shape in shapes:
            points = shape['points']
            vertices = np.array([points], dtype=np.int32)
            cv2.fillPoly(filled_map, vertices, 1)

        return filled_map

    def compute_heatmap(self, filled_map, image_size, k_ratio, transpose=False):
        """Compute the heatmap from a filled map."""
        if filled_map.dtype != np.float32:
            filled_map = filled_map.astype(np.float32)

        k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
        if k_size % 2 == 0:
            k_size += 1

        heatmap = cv2.GaussianBlur(filled_map, (k_size, k_size), 0)

        # if np.sum(heatmap) > 0:
        #     heatmap /= np.sum(heatmap)

        if transpose:
            heatmap = heatmap.transpose()

        return heatmap

    def generate_heatmap_around_keypoints(self, keypoints, image_size, num_vertices=8, radius=20, k_ratio=10):
        """根据关键点生成热力图"""
        shapes = []

        for keypoint in keypoints:
            polygon = self.generate_irregular_polygon(keypoint.cpu().numpy().astype(int), num_vertices, radius)
            shapes.append({'points': polygon})

        filled_map = self.create_filled_map(shapes, image_size)
        heatmap = self.compute_heatmap(filled_map, image_size, k_ratio)

        return heatmap

    def _project_keypoints_to_img(self, candidate_keypoints):
        # 直接生成 cluster_color_map

        # 生成热力图
        heatmap = self.generate_heatmap_around_keypoints(candidate_keypoints, (448, 448), num_vertices=5, radius=50, k_ratio=3)

        return heatmap

    def normalize_feature_map(self, feature_map):
        """
        对特征图进行归一化，值范围归一化到 [0, 1]。
        """
        feature_map_min = feature_map.min()
        feature_map_max = feature_map.max()
        normalized_map = (feature_map - feature_map_min) / (feature_map_max - feature_map_min + 1e-10)
        return normalized_map

    def _reshape_transform(self, tensor, patch_size, stride):
        height = (448 - patch_size) // stride + 1
        width = (448 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result

    # 定义反标准化函数
    def denormalize(self, tensor, mean, std):
        # 根据特定的均值和标准差进行去归一化
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor


    def get_ego_mask(self, ego, point):
        denormalized_tensor = self.denormalize(ego, mean=(0.592, 0.558, 0.523), std=(0.228, 0.223, 0.229))
        ego0_numpy = denormalized_tensor.permute(1, 2, 0).cpu().numpy()
        ego0 = (ego0_numpy * 255).astype(np.uint8)

        self.predictor.set_image(ego0)
        input_point = point.unsqueeze(0).numpy().astype(np.float32)
        input_label = np.array([1] * len(input_point))
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        labeled_masks = np.zeros_like(masks[0], dtype=np.int32)
        for i, mask in enumerate(masks):
            labeled_masks[mask] = i + 1

        unique_mask_count = len(np.unique(labeled_masks)) - 1
        np.random.seed(42)  # 确保随机生成可复现
        if unique_mask_count < 3:
            while len(np.unique(labeled_masks)) - 1 < 3:
                new_input_point = input_point[np.random.randint(0, input_point.shape[0], 1), :]
                new_input_label = np.array([1])
                masks, _, _ = self.predictor.predict(
                    point_coords=new_input_point,
                    point_labels=new_input_label,
                    multimask_output=True
                )
                for i, mask in enumerate(masks):
                    labeled_masks[mask] = unique_mask_count + i + 1

        ret = {
            "rgb": ego0,
            "seg": labeled_masks
        }
        return ret

    def get_keypoint_features(self, feature_map, keypoints, scale=16, r=5):
        """
        从特征图中提取关键点及其周围感兴趣区域的特征，假设 keypoints 是 3 个关键点。

        Args:
            feature_map: 特征图，形状为 (H, W, feature_dim)
            keypoints: 关键点坐标列表，形状为 (3, 2)，假设有 3 个关键点，坐标范围为原图尺寸
            original_size: 原始图像的尺寸 (height, width)
            r: 感兴趣区域的半径

        Returns:
            keypoint_features: 提取的 3 个关键点及其周围区域特征，形状为 (3, feature_dim)
        """
        # 将关键点映射到低分辨率
        keypoints = keypoints / scale
        keypoint_features = []
        # 遍历每个关键点
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            # 边界检查，防止越界
            x = min(max(x, 0), feature_map.shape[2] - 1)
            y = min(max(y, 0), feature_map.shape[1] - 1)

            # 提取局部区域特征
            local_region = feature_map[:,
                           max(0, y - r):min(y + r, feature_map.shape[1]),
                           max(0, x - r):min(x + r, feature_map.shape[2])]
            # 计算局部区域的均值特征
            local_region_mean = local_region.mean(dim=(1, 2))  # 输出形状 (C)
            keypoint_features.append(local_region_mean)
        # 组合为 (3, feature_dim) 的张量
        keypoint_features = torch.stack(keypoint_features)  # 堆叠为张量，形状为 (3, feature_dim)
        # 将特征通过线性映射层投影到统一空间
        keypoint_features = self.proj(keypoint_features)  # (N, output_dim)
        return keypoint_features

    @torch.no_grad()
    def test_forward(self, ego, aff_label):
        _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # attn: b x 6 x (1+hw) x (1+hw)
        ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1)
        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.shape
        ego_proj2 = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj2)

        # ------------yf_hand_pred------------------
        ego_proj_hand = self.hand_ego_proj(ego_proj)
        ego_pred_hand = self.aff_fc_hand(ego_proj_hand)
        hand_pred = self.gap_hand(ego_pred_hand).view(b, 14)
        hand_pred = self.softmax(hand_pred)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            a = aff_label[b_]
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]

        return gt_ego_cam, hand_pred

    def _reshape_transform(self, tensor, patch_size, stride):
        height = (448 - patch_size) // stride + 1
        width = (448 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result

    def get_config(self, config_path=None):
        if config_path is None:
            this_file_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(this_file_dir, 'configs/config.yaml')
        assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config