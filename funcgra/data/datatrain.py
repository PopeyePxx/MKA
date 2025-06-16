import os
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms
import json


class TrainData(data.Dataset):
    def __init__(self, exocentric_root, egocentric_root, resize_size=512, crop_size=448, divide="Seen"):

        self.exocentric_root = exocentric_root
        self.egocentric_root = egocentric_root
        self.point_list = []
        self.image_list = []
        self.gt_r_list = []
        self.exo_image_list = []
        self.resize_size = resize_size
        self.crop_size = crop_size
        # self.crop_size_func = (768, 512)
        if divide == "Seen":
            # --------yf-------------
            self.aff_list = ['hold', 'press', 'click', 'clamp', 'grip', 'open']
            self.obj_list = ['screwdriver', 'plug', 'kettle', 'hammer', 'spraybottle', 'stapler', 'flashlight',
                             'bottle', 'cup', 'mouse', 'knife', 'pliers', 'spatula', 'scissors', 'doorhandle',
                             'lightswitch', 'drill', 'valve']
            # --------yf-------------
        else:

            self.aff_list = ['hold', 'press', 'click', 'clamp', 'grip', 'open']
            self.obj_list = ['screwdriver', 'plug', 'kettle', 'hammer', 'spraybottle', 'stapler', 'flashlight',
                             'bottle', 'cup', 'mouse', 'knife', 'pliers', 'spatula', 'scissors', 'doorhandle',
                             'lightswitch', 'drill', 'valve']

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),  # 如果只提供一个整数而不是元组，图像将被等比例缩放，使得较小的边等于该整数，同时保持图像的宽高比不变。
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.592, 0.558, 0.523),
                                 std=(0.228, 0.223, 0.229))]) # 进行标准化，即减去均值并除以标准差

        # self.transform = transforms.Compose([
        #     transforms.Resize(resize_size),  # 如果只提供一个整数而不是元组，图像将被等比例缩放，使得较小的边等于该整数，同时保持图像的宽高比不变。
        #     transforms.RandomCrop(crop_size),  # 进行随机裁剪
        #     transforms.RandomHorizontalFlip(),  # 以50%的概率水平翻转图像
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.592, 0.558, 0.523),
        #                          std=(0.228, 0.223, 0.229))])  # 进行标准化，即减去均值并除以标准差
        # image list for egocentric images
        files = os.listdir(self.exocentric_root)
        for file in files:
            file_path = os.path.join(self.exocentric_root, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path = os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)
                for img in images:
                    if not img.endswith(('.txt')):
                       img_path = os.path.join(obj_file_path, img)
                       self.image_list.append(img_path)

    def __getitem__(self, item):
        # 加载外视角图像（exocentric images）和标签
        exocentric_image_path = self.image_list[item]

        # 提取外视角图像的任务标签和物体名称
        names = exocentric_image_path.split("/")
        aff_name, object = names[-3], names[-2]
        exocentric_image = self.load_img(exocentric_image_path)

        aff_label = self.aff_list.index(aff_name)

        # 加载来自选定任务标签和物体的自视角图像
        ego_path = os.path.join(self.egocentric_root, aff_name, object)
        obj_images = os.listdir(ego_path)

        # 获取图片和对应的 .json 文件
        image_files = [img for img in obj_images if not img.endswith('.json')]
        json_files = [json for json in obj_images if json.endswith('.json')]

        assert len(image_files) == len(json_files), "Image and JSON files do not match!"

        # 随机选择索引
        idx = random.randint(0, len(image_files) - 1)

        # 获取对应的图片和点文件路径
        egocentric_image_path = os.path.join(ego_path, image_files[idx])
        point_path = os.path.join(ego_path, json_files[idx])

        # 加载图片和点数据
        egocentric_image = self.load_img(egocentric_image_path)

        # -----------yf-----------------
        point = self.load_json(point_path)
        # -----------yf-----------------
        # pick one available affordance, and then choose & load exo-centric images
        num_exo = 3
        exo_dir = os.path.dirname(exocentric_image_path)
        all_files = os.listdir(exo_dir)
        exocentrics = [file for file in all_files if not file.endswith('.txt')]

        exo_img_name = [os.path.basename(exocentric_image_path)]
        exocentric_images = [exocentric_image]

        # 设置加载额外外向中心图像的数量（num_exo）
        for i in range(num_exo - 1):
            exo_img_ = random.choice(exocentrics)

            tmp_exo = self.load_img(os.path.join(exo_dir, exo_img_))
            exo_img_name.append(exo_img_)
            exocentric_images.append(tmp_exo)

        exocentric_images = torch.stack(exocentric_images, dim=0)

        return exocentric_images, egocentric_image, aff_label, point, exocentric_image_path, egocentric_image_path

    def transform_keypoints(self, keypoints, start_x, start_y):
        # 计算缩放比例
        scale_x = self.resize_size  # new_width / original_width
        scale_y = self.resize_size  # new_height / original_height

        # 更新关键点坐标
        keypoints_transformed = keypoints.clone()  # 克隆以避免修改原始关键点
        keypoints_transformed[0::3] *= scale_x  # 更新 x 坐标
        keypoints_transformed[1::3] *= scale_y  # 更新 y 坐标

        # 根据裁剪的位置调整关键点
        keypoints_transformed[0::3] -= start_x  # 更新 x 坐标
        keypoints_transformed[1::3] -= start_y  # 更新 y 坐标

        # 确保关键点仍在图像范围内
        keypoints_transformed = torch.clamp(keypoints_transformed, min=0)  # 防止负值

        return keypoints_transformed

    def load_img(self, path):
        img = Image.open(path).convert('RGB')

        img = self.transform(img)
        return img

    def load_json(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        shape = data['shapes']
        all_points = shape[0]['points'] # 将所有点合并
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        original_size = (image_width, image_height)
        resize_point = self.resize_points(all_points, original_size, self.crop_size)
        return resize_point

    def resize_points(self, points, original_size, crop_size):
        """
        根据图像Resize操作调整点坐标。
        :param points: 原始点坐标列表，形如 [[x1, y1], [x2, y2], ...]
        :param original_size: 原始图像尺寸 (width, height)
        :param crop_size: 目标图像尺寸 (new_width = new_height = crop_size)
        :return: 调整后的点坐标列表
        """
        original_width, original_height = original_size
        scale_x = crop_size / original_width
        scale_y = crop_size / original_height
        x, y = points[0]
        # 调整每个点
        resized_points = [y * scale_y, x * scale_x]
        return resized_points

    def __len__(self):

        return len(self.image_list)
