import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from sklearn.cluster import MeanShift
import torch.nn as nn
import matplotlib.pyplot as plt

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

class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.mean_shift = MeanShift(bandwidth=self.config['min_dist_bt_keypoints'], bin_seeding=True, n_jobs=32)
        self.patch_size = 14  # dinov2
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # self.feature_transformer = FeatureTransformer(3, 3).to(self.device)
        self.merge_weight = nn.Parameter(torch.zeros(3)).to(self.device)
        input_dim = 768
        out_dim = 512
        self.lln_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)]).to(self.device)
        self.lln_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(3)]).to(self.device)
        self.lln_norm_1 = nn.LayerNorm(out_dim).to(self.device)
        self.embedder = Mlp(in_features=input_dim, hidden_features=int(out_dim), out_features=out_dim,
                            act_layer=nn.GELU, drop=0.).to(self.device)

    def get_keypoints(self, rgb, masks):
        masks, features_flat = self.get_multiscale_keypoints(rgb, masks)
        candidate_pixels, candidate_rigid_group_ids, cluster_color_map = self._cluster_features(
            features_flat, masks)
        merged_indices = self._merge_clusters(candidate_pixels)
        candidate_keypoints = candidate_pixels[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]

        # sort candidates by locations
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]
        return candidate_keypoints, masks, candidate_rigid_group_ids

    def _annotate_keypoints(self, image, candidate_pixels):
        # 创建副本以避免修改原始图像
        annotated_image = image.copy()

        # 遍历每个候选像素坐标，并在图像上画圈标注
        for pixel in candidate_pixels:
            x, y = int(pixel[0]), int(pixel[1])
            cv2.circle(annotated_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色圆圈，半径5像素

        return annotated_image

    def _preprocess(self, rgb, masks0):
        masks = [masks0 == uid for uid in np.unique(masks0) if uid !=0]

        # Last N features from DINO
        dino_out_ego = self.dino_model.get_intermediate_layers(rgb.unsqueeze(0), n=3, return_class_token=False)
        merge_weight = torch.softmax(self.merge_weight, dim=0)

        dino_dense_ego = 0
        for i, feat in enumerate(dino_out_ego):
            feat_ = self.lln_linear[i](feat[0])
            feat_ = self.lln_norm[i](feat_)
            dino_dense_ego += feat_ * merge_weight[i]

        ego_desc = self.lln_norm_1(self.embedder(dino_dense_ego))  # bs,1024,512
        ego_desc = ego_desc.unsqueeze(0).permute(0, 2, 1).reshape(1, 512, 32, 32)
        # self.visualize_feature_map_max(ego_desc)
        return masks, ego_desc

    def visualize_feature_map_max(self, feature_map,  save_path="feature_map_max.png"):
        """
        Visualize the feature map by taking the maximum activation across channels.

        Args:
            feature_map: Tensor of shape (1, 512, 32, 32)
        """
        feature_map = feature_map.squeeze(0).cpu().detach().numpy()  # Shape: (512, 32, 32)

        # Max projection across channels
        feature_map_max = np.max(feature_map, axis=0)  # Shape: (32, 32)

        # Normalize for visualization
        feature_map_max = (feature_map_max - feature_map_max.min()) / (feature_map_max.max() - feature_map_max.min())

        # Create figure
        plt.figure(figsize=(6, 6))
        plt.imshow(feature_map_max, cmap="jet")
        plt.axis("off")

        # Save the image without displaying
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()  # Close the figure to prevent display

    # 可视化 ego_desc

    def get_multiscale_keypoints(self, rgb, masks):
        # Preprocessing
        masks, ego_desc = self._preprocess(rgb, masks)

        # Extract features at different scales
        scale_1_desc = self._get_features_at_scale(ego_desc, scale_factor=1.0)  # Original scale
        scale_2_desc = self._get_features_at_scale(ego_desc, scale_factor=2.0)  # Downscaled features (e.g., half size)
        scale_3_desc = self._get_features_at_scale(ego_desc, scale_factor=3.0)  # Upscaled features

        # Resize each scale's features to 448x448
        scale_1_desc_resized = self._resize_to_448(scale_1_desc)
        scale_2_desc_resized = self._resize_to_448(scale_2_desc)
        scale_3_desc_resized = self._resize_to_448(scale_3_desc)

        # Merge features from different scales
        multiscale_desc = self._merge_multiscale_features(scale_1_desc_resized, scale_2_desc_resized,
                                                          scale_3_desc_resized)

        return masks, multiscale_desc

    def _get_features_at_scale(self, ego_desc, scale_factor):
        """
        Extract features at a specific scale by resizing the feature map.
        :param ego_desc: The original feature map.
        :param scale_factor: The scale factor for resizing.
        :return: Resized features.
        """
        # Resize the feature map to the desired scale (upscale or downscale)
        size = int(ego_desc.shape[2] * scale_factor), int(ego_desc.shape[3] * scale_factor)
        scaled_desc = interpolate(ego_desc, size=size, mode='bilinear', align_corners=False)
        return scaled_desc

    def _resize_to_448(self, feature_desc):
        """
        Resize the feature map to 448x448.
        :param feature_desc: Feature map at any scale.
        :return: Resized feature map.
        """
        resized_desc = interpolate(feature_desc, size=(448, 448), mode='bilinear', align_corners=False)
        return resized_desc

    def _merge_multiscale_features(self, scale_1_desc, scale_2_desc, scale_3_desc):
        """
        Merge features from different scales.
        This can be done by concatenating or using weighted sums.
        :param scale_1_desc: Features at the original scale.
        :param scale_2_desc: Features at the second scale.
        :param scale_3_desc: Features at the third scale.
        :return: Merged multi-scale features.
        """
        # Concatenate features from different scales along the feature dimension (channel dimension)
        merged_desc = scale_1_desc+scale_2_desc+scale_3_desc
        merged_desc = merged_desc.squeeze(0)

        return merged_desc

    def _project_keypoints_to_img(self, rgb, candidate_pixels, cluster_color_map, candidate_rigid_group_ids, masks,
                                  features_flat):
        projected = rgb.copy()
        # 将彩色聚类掩码叠加到原始图像上
        alpha = 0.4  # 透明度
        projected = cv2.addWeighted(projected, 1-alpha, cluster_color_map, alpha, 0)

        # overlay keypoints on the image
        for keypoint_count, pixel in enumerate(candidate_pixels):
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)
            # draw a box
            box_width = 30 + 10 * (text_length - 1)
            box_height = 30
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                          (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                          (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
            # draw text
            org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
            color = (255, 0, 0)
            cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            keypoint_count += 1
        return projected

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, ego_desc):
        interpolated_feature_grid = interpolate(ego_desc,
                                                # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(448, 448),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(
            0)  # float32 [H, W, feature_dim] 448,448,512
        return interpolated_feature_grid

    # def _cluster_features(self, points, features_flat, masks):
    def _cluster_features(self, features_flat, masks):
        candidate_pixels = []
        candidate_rigid_group_ids = []

        cluster_color_map = np.zeros((448, 448, 3), dtype=np.uint8)  # 用于保存颜色的掩码

        num_clusters = self.config['num_candidates_per_mask']  # 每个掩码的聚类数

        features_flat0 = torch.tensor(features_flat, device=self.device)

        # 将结果展平成 (200704, 384) 以便后续使用
        transformed_features_flat = features_flat0.view(-1, 512)  # (200704, 384)
        for rigid_group_id, binary_mask in enumerate(masks):
            # 忽略过大的掩码
            print('np.mean(binary_mask)', np.mean(binary_mask))
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue
            if np.sum(binary_mask) < self.config['min_mask_pixels']:
                continue

            # 将掩码展平，与 features_flat 对齐
            binary_mask_flat = binary_mask.reshape(-1)

            # 对所有特征计算范数，并对掩码为 0 的部分设为 0
            obj_features_flat = transformed_features_flat.clone()  # 复制 features_flat 以免修改原始数据
            obj_features_flat[binary_mask_flat == 0] = 0
            # 提取前景特征
            feature_pixels = torch.tensor(np.argwhere(binary_mask), device=self.device)  # 掩码中非零像素的坐标

            # 使用主成分分析（PCA）对前景特征进行降维
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])  # 降到3维
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])
            # 将掩码为 0 的区域设置为黑色
            features_pca[binary_mask_flat == 0] = 0  # 对于背景部分的特征设为 0
            features_pca = features_pca[binary_mask.reshape(-1)]  # 提取属于该掩码区域的特征

            X = features_pca
            # 降维后的特征空间上对物体进行聚类

            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config['num_candidates_per_mask'],
                distance='euclidean'
            )

            for cluster_id in range(num_clusters):
                member_idx = cluster_ids_x == cluster_id
                member_pixels = feature_pixels[member_idx]
                cluster_center = cluster_centers[cluster_id][:3].cuda()

                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)

                # 计算与聚类中心最近的点，作为关键点
                closest_idx = torch.argmin(dist)
                candidate_pixels.append(member_pixels[closest_idx].cpu().numpy())  # 转回CPU
                candidate_rigid_group_ids.append(rigid_group_id)
            # 如果没有找到候选点，从 features_flat 中选择最接近掩码质心的点
        if len(candidate_pixels) == 0:
            print("Warning: No candidate pixels found. Using nearest valid point.")
            for binary_mask in masks:
                if np.sum(binary_mask) >= self.config['min_mask_pixels']:
                    # 计算前景区域质心
                    coords = np.argwhere(binary_mask)
                    centroid = np.mean(coords, axis=0)

                    # 找到最接近质心的点
                    distances = np.linalg.norm(coords - centroid, axis=1)
                    closest_idx = np.argmin(distances)
                    candidate_pixels.append(coords[closest_idx])  # 返回最接近的前景点
                    candidate_rigid_group_ids.append(-1)  # 使用特殊标志

        candidate_pixels = np.array(candidate_pixels)  # 关键点的像素坐标
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)  # 每个关键点对应的物体标识符
        return candidate_pixels, candidate_rigid_group_ids, cluster_color_map

    def _merge_clusters(self, candidate_keypoints):
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices

    def _cluster_features_with_surf_and_dino(self, points, rgb_image, features_flat, masks):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []

        cluster_color_map = np.zeros((rgb_image.shape[1], rgb_image.shape[1], 3), dtype=np.uint8)  # 用于保存颜色的掩码

        num_clusters = self.config['num_candidates_per_mask']  # 每个掩码的聚类数
        base_colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)  # 为每个聚类生成基准 RGB 颜色

        # 初始化 SURF 特征检测器
        # surf = cv2.xfeatures2d.SURF_create(hessianThreshold=6)

        for rigid_group_id, binary_mask in enumerate(masks):
            # 忽略过大的掩码
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue

            # 应用掩码到图像
            masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=binary_mask.astype(np.uint8))
            feature_points = points[binary_mask]
            # 使用 SURF 检测特征点和描述符
            # keypoints, descriptors = surf.detectAndCompute(masked_image, None)
            # 使用 ORB 代替 SURF
            orb = cv2.ORB_create()

            # 检测 ORB 特征点和描述符
            keypoints, descriptors = orb.detectAndCompute(masked_image, None)

            if descriptors is None:
                continue  # 如果没有检测到特征点，跳过这个物体

            # 提取 SURF 特征点的像素坐标
            feature_pixels = np.array([kp.pt for kp in keypoints], dtype=np.int32)

            # 获取对应 DINOv2 特征平面中的局部特征
            dino_local_features = self._get_dino_local_features(features_flat, feature_pixels, features_flat.shape)

            # 结合 SURF 描述符和 DINOv2 特征
            combined_features = self._combine_surf_and_dino_features(descriptors, dino_local_features)
            obj_features_flat = torch.from_numpy(combined_features)

            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])  # 降到3维
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])

            # 掩码中非零像素的kongjian坐标

            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch = (feature_points_torch - feature_points_torch.min(0)[0]) / (
                    feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([features_pca, feature_points_torch[:features_pca.shape[0], :]],
                          dim=-1)  # X 是由特征和空间坐标组合而成的特征向量

            # 在特征空间中进行聚类
            cluster_ids_x, cluster_centers = kmeans(
                X,  # 输入特征
                num_clusters=self.config['num_candidates_per_mask'],  # 聚类数
                distance='euclidean',  # 使用欧几里得距离
                device=self.device  # 计算设备（如 GPU 或 CPU）
            )
            # cluster_ids_x, cluster_centers = self._cluster_features_with_kmeans(combined_features, feature_pixels)

            # 为每个聚类区域分配颜色
            for cluster_id in range(num_clusters):
                base_color = base_colors[cluster_id]  # 选择聚类的基准颜色
                member_idx = cluster_ids_x == cluster_id
                member_pixels = feature_pixels[member_idx]
                cluster_center = cluster_centers[cluster_id][:3]

                # 计算每个像素到聚类中心的距离，并使用距离来调整颜色深浅

                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                dist_normalized = dist / dist.max()  # 归一化距离

                # 将颜色应用到颜色掩码的相应位置，基于距离插值颜色
                for pixel, d in zip(member_pixels, dist_normalized):
                    adjusted_color = (1 - d) * torch.tensor(base_color, device=self.device)  # 距离越大，颜色越浅
                    adjusted_color = adjusted_color.to(torch.uint8).cpu()  # 转换为uint8类型的NumPy数组
                    cluster_color_map[pixel[0], pixel[1]] = adjusted_color  # 将颜色应用到RGB通道

                # 将颜色应用到颜色掩码的相应位置，基于距离插值颜色
                for pixel, d in zip(member_pixels, dist_normalized):
                    adjusted_color = (1 - d) * base_color
                    cluster_color_map[pixel[1], pixel[0]] = adjusted_color

                # 计算与聚类中心最近的点，作为关键点
                closest_idx = np.argmin(dist)
                candidate_keypoints.append(feature_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids, cluster_color_map

    def _get_dino_local_features(self, features_flat, feature_pixels, feature_shape):
        """
        提取 DINOv2 特征平面中，SURF 特征点周围的局部特征。
        """
        H, W, feature_dim = 640, 480, 384
        dino_local_features = []

        for pixel in feature_pixels:
            x, y = pixel
            # 根据 SURF 特征点的位置，在 DINOv2 提取的特征平面中提取对应的局部特征
            local_feature = features_flat[y * W + x]
            dino_local_features.append(local_feature.numpy())

        return np.array(dino_local_features)

    def _combine_surf_and_dino_features(self, surf_descriptors, dino_features):
        """
        将 SURF 特征和 DINOv2 的局部特征结合在一起。
        """
        # 可以选择将 SURF 描述符和 DINO 特征直接拼接，或者通过加权方式结合两种特征
        combined_features = np.hstack((surf_descriptors, dino_features))

        return combined_features

