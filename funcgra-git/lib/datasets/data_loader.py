##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret, JingyiXie
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import torch
from torch.utils import data

import lib.datasets.tools.transforms as trans
import lib.datasets.tools.cv2_aug_transforms as cv2_aug_trans
import lib.datasets.tools.pil_aug_transforms as pil_aug_trans
from lib.datasets.loader.default_loader import DefaultLoader, CSDataTestLoader
from lib.datasets.loader.ade20k_loader import ADE20KLoader
from lib.datasets.loader.lip_loader import LipLoader
from lib.datasets.loader.offset_loader import DTOffsetLoader
from lib.datasets.tools.collate import collate
from lib.utils.tools.logger import Logger as Log

from lib.utils.distributed import get_world_size, get_rank, is_distributed

import json
import os
from collections import namedtuple
from torchvision import transforms
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import json
import os
from collections import namedtuple
import torchvision
import torch.utils.data as data
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class AFF_data(data.Dataset):

    def __init__(self,image_root,crop_size=224,select="exocentric"):
        self.images = []
        self.targets = []
        self.crop_size=crop_size

        files=os.listdir(image_root)
        
        for file in files:
            file_path=os.path.join(image_root,file)
            sub_files=os.listdir(file_path)
            for sub_file in sub_files:
                sub_path=os.path.join(file_path,sub_file)

                images=os.listdir(sub_path)
                for img in images:
                    if img[-4:]=="json":
                        continue
                    img_path=os.path.join(sub_path,img)
                    #label_path=img_path.replace("images","masks")
                    label_path=img_path.replace("jpg","json")
                    if os.path.exists(label_path):
                        self.images.append(img_path)
                        self.targets.append(label_path)
        if select=="all":
            image_root=image_root.replace("exocentric","egocentric")
            files = os.listdir(image_root)
            for file in files:
                file_path = os.path.join(image_root, file)
                sub_files = os.listdir(file_path)
                for sub_file in sub_files:
                    sub_path = os.path.join(file_path, sub_file)

                    images = os.listdir(sub_path)
                    for img in images:
                        if img[-4:] == "json":
                            continue
                        img_path = os.path.join(sub_path, img)
                        # label_path=img_path.replace("images","masks")
                        label_path = img_path.replace("jpg", "json")
                        if os.path.exists(label_path):
                            self.images.append(img_path)
                            self.targets.append(label_path)
        
        self.transform = transforms.Compose([transforms.Resize((crop_size,crop_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
        
        
    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert('RGB')
        k_ratio = 3
        #heatmaps=image
        target_file=self.targets[index]
        target=read_heatmap_json(target_file,k_ratio=k_ratio,crop_size=self.crop_size)
        
        image=self.transform(image)
    
        
        return image, torch.stack(target,dim=0)

    def __len__(self):
        return len(self.images)

class AFF_data_test(data.Dataset):

    def __init__(self,image_root,crop_size=224,select="exo"):

        self.images = []
        self.targets = []
        self.transform = transforms.Compose([transforms.Resize((crop_size,crop_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
        self.select=select

        files=os.listdir(image_root)
        for file in files:
            file_path=os.path.join(image_root,file)
            sub_files=os.listdir(file_path)
            for sub_file in sub_files:
                sub_path=os.path.join(file_path,sub_file)
                images=os.listdir(sub_path)
                for img in images:
                    if img[-4:]=="json":
                        continue
                    img_path=os.path.join(sub_path,img) 
                    
                    self.images.append(img_path)
                    
        if self.select=="all":
            image_root=image_root.replace("exocentric","egocentric")
            files=os.listdir(image_root)
            for file in files:
                file_path=os.path.join(image_root,file)
                sub_files=os.listdir(file_path)
                for sub_file in sub_files:
                    sub_path=os.path.join(file_path,sub_file)
                    images=os.listdir(sub_path)
                    for img in images:
                        if img[-4:]=="json":
                            continue
                        img_path=os.path.join(sub_path,img)
                        
                        self.images.append(img_path)
                            
    def __getitem__(self, index):

        path=self.images[index]
        image = Image.open(self.images[index]).convert('RGB')
        
        image= self.transform(image)
    
        return image,path

    def __len__(self):
        return len(self.images)

def read_heatmap_json(json_file,k_ratio=3,crop_size=224):
    with open(json_file, 'r') as load_f:
            json_data = json.load(load_f)

            k_size = int(np.sqrt(json_data['image_height'] * json_data['image_width']) / k_ratio)
            if k_size % 2 == 0:
                k_size += 1
            # Compute the heatmap using the Gaussian filter.

            #### heatmap1
            heatmap1 = np.zeros((json_data['image_height'], json_data['image_width']),dtype=np.float32)
            point1s = json_data['kps1']
            for point in point1s:
                x = point[1]
                y = point[0]
                row = int(x)
                col = int(y)
                try:
                    heatmap1[row, col] += 1.0
                except:
                    # resize pushed it out of bounds somehow
                    row = min(max(row, 0), json_data['image_width'] - 1)
                    col = min(max(col, 0), json_data['image_height'] - 1)
                    heatmap1[row, col] += 1.0
            heatmap1 = cv2.GaussianBlur(heatmap1, (k_size, k_size), 0)
            heatmap1 = (heatmap1 - np.min(heatmap1)) / (np.max(heatmap1) - np.min(heatmap1) + 1e-10)
            heatmap1=cv2.resize(heatmap1,(crop_size,crop_size))
            map1=torch.from_numpy(heatmap1)
            
            #### heatmap2
            heatmap2 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
            point2s = json_data['kps2']
            for point in point2s:
                x = point[1]
                y = point[0]
                row = int(x)
                col = int(y)
                try:
                    heatmap2[row, col] += 1.0
                except:
                    # resize pushed it out of bounds somehow
                    row = min(max(row, 0), json_data['image_width'] - 1)
                    col = min(max(col, 0), json_data['image_height'] - 1)
                    heatmap2[row, col] += 1.0
            heatmap2 = cv2.GaussianBlur(heatmap2, (k_size, k_size), 0)
            heatmap2 = (heatmap2 - np.min(heatmap2)) / (np.max(heatmap2) - np.min(heatmap2) + 1e-10)
            heatmap2=cv2.resize(heatmap2,(crop_size,crop_size))
            map2=torch.from_numpy(heatmap2)
            
           
            #### heatmap3
            heatmap3 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
            point3s = json_data['kps3']
            for point in point3s:
                x = point[1]
                y = point[0]
                row = int(x)
                col = int(y)
                try:
                    heatmap3[row, col] += 1.0
                except:
                    # resize pushed it out of bounds somehow
                    row = min(max(row, 0), json_data['image_width'] - 1)
                    col = min(max(col, 0), json_data['image_height'] - 1)
                    heatmap3[row, col] += 1.0
            heatmap3 = cv2.GaussianBlur(heatmap3, (k_size, k_size), 0)
            heatmap3 = (heatmap3 - np.min(heatmap3)) / (np.max(heatmap3) - np.min(heatmap3) + 1e-10)
            heatmap3=cv2.resize(heatmap3,(crop_size,crop_size))
            map3=torch.from_numpy(heatmap3)
            
            #### heatmap4
            heatmap4 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
            point4s = json_data['kps4']
            for point in point4s:
                x = point[1]
                y = point[0]
                row = int(x)
                col = int(y)
                try:
                    heatmap4[row, col] += 1.0
                except:
                    # resize pushed it out of bounds somehow
                    row = min(max(row, 0), json_data['image_width'] - 1)
                    col = min(max(col, 0), json_data['image_height'] - 1)
                    heatmap4[row, col] += 1.0
            heatmap4 = cv2.GaussianBlur(heatmap4, (k_size, k_size), 0)
            heatmap4 = (heatmap4 - np.min(heatmap4)) / (np.max(heatmap4) - np.min(heatmap4) + 1e-10)
            heatmap4=cv2.resize(heatmap4,(crop_size,crop_size))
            map4=torch.from_numpy(heatmap4)
            
            heatmap5 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
            point5s = json_data['kps5']
            for point in point5s:
                x = point[1]
                y = point[0]
                row = int(x)
                col = int(y)
                try:
                    heatmap5[row, col] += 1.0
                except:
                    # resize pushed it out of bounds somehow
                    row = min(max(row, 0), json_data['image_width'] - 1)
                    col = min(max(col, 0), json_data['image_height'] - 1)
                    heatmap5[row, col] += 1.0
            heatmap5 = cv2.GaussianBlur(heatmap5, (k_size, k_size), 0)
            heatmap5 = (heatmap5 - np.min(heatmap5)) / (np.max(heatmap5) - np.min(heatmap5) + 1e-10)
            heatmap5=cv2.resize(heatmap5,(crop_size,crop_size))
            map5=torch.from_numpy(heatmap5)
            
            heatmap6 = np.zeros((json_data['image_height'], json_data['image_width']), dtype=np.float32)
            point6s = json_data['kps6']
            for point in point6s:
                x = point[1]
                y = point[0]
                row = int(x)
                col = int(y)
                try:
                    heatmap6[row, col] += 1.0
                except:
                    # resize pushed it out of bounds somehow
                    row = min(max(row, 0), json_data['image_width'] - 1)
                    col = min(max(col, 0), json_data['image_height'] - 1)
                    heatmap6[row, col] += 1.0
            heatmap6 = cv2.GaussianBlur(heatmap6, (k_size, k_size), 0)
            heatmap6 = (heatmap6 - np.min(heatmap6)) / (np.max(heatmap6) - np.min(heatmap6) + 1e-10)
            heatmap6=cv2.resize(heatmap6,(crop_size,crop_size))
            map6=torch.from_numpy(heatmap6)
    return [map1,map2,map3,map4,map5,map6]




















class DataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        from lib.datasets.tools import cv2_aug_transforms
        self.aug_train_transform = cv2_aug_transforms.CV2AugCompose(self.configer, split='train')
        self.aug_val_transform = cv2_aug_transforms.CV2AugCompose(self.configer, split='val')

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ])

        self.label_transform = trans.Compose([
            trans.ToLabel(),
            trans.ReLabel(255, -1), ])

    def get_dataloader_sampler(self, klass, split, dataset):

        from lib.datasets.loader.multi_dataset_loader import MultiDatasetLoader, MultiDatasetTrainingSampler

        root_dir = self.configer.get('data', 'data_dir')
        if isinstance(root_dir, list) and len(root_dir) == 1:
            root_dir = root_dir[0]

        kwargs = dict(
            dataset=dataset,
            aug_transform=(self.aug_train_transform if split == 'train' else self.aug_val_transform),
            img_transform=self.img_transform,
            label_transform=self.label_transform,
            configer=self.configer
        )

        if isinstance(root_dir, str):
            loader = klass(root_dir, **kwargs)
            multi_dataset = False
        elif isinstance(root_dir, list):
            loader = MultiDatasetLoader(root_dir, klass, **kwargs)
            multi_dataset = True
            Log.info('use multi-dataset for {}...'.format(dataset))
        else:
            raise RuntimeError('Unknown root dir {}'.format(root_dir))

        if split == 'train':
            if is_distributed() and multi_dataset:
                raise RuntimeError('Currently multi dataset doesn\'t support distributed.')

            if is_distributed():
                sampler = torch.utils.data.distributed.DistributedSampler(loader)
            elif multi_dataset:
                sampler = MultiDatasetTrainingSampler(loader)
            else:
                sampler = None

        elif split == 'val':

            if is_distributed():
                sampler = torch.utils.data.distributed.DistributedSampler(loader)
            else:
                sampler = None

        return loader, sampler

    def get_trainloader(self):
        if self.configer.exists('data', 'use_edge') and self.configer.get('data', 'use_edge') == 'ce2p':
            """
            ce2p manner:
            load both the ground-truth label and edge.
            """
            Log.info('use edge (follow ce2p) for train...')
            klass = LipLoader

        elif self.configer.exists('data', 'use_dt_offset') or self.configer.exists('data', 'pred_dt_offset'):
            """
            dt-offset manner:
            load both the ground-truth label and offset (based on distance transform).
            """
            Log.info('use distance transform offset loader for train...')
            klass = DTOffsetLoader

        elif self.configer.exists('train', 'loader') and \
            (self.configer.get('train', 'loader') == 'ade20k' 
             or self.configer.get('train', 'loader') == 'pascal_context'
             or self.configer.get('train', 'loader') == 'pascal_voc'
             or self.configer.get('train', 'loader') == 'coco_stuff'):
            """
            ADE20KLoader manner:
            support input images of different shapes.
            """
            Log.info('use ADE20KLoader (diverse input shape) for train...')
            klass = ADE20KLoader
        else:
            """
            Default manner:
            + support input images of the same shapes.
            + support distributed training (the performance is more un-stable than non-distributed manner)
            """
            Log.info('use the DefaultLoader for train...')
            klass = DefaultLoader
        loader, sampler = self.get_dataloader_sampler(klass, 'train', 'train')
        trainloader = data.DataLoader(
            loader,
            batch_size=self.configer.get('train', 'batch_size') // get_world_size(), pin_memory=True,
            num_workers=self.configer.get('data', 'workers') // get_world_size(),
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=self.configer.get('data', 'drop_last'),
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('train', 'data_transformer')
            )
        )
        return trainloader
            

    def get_valloader(self, dataset=None):
        dataset = 'val' if dataset is None else dataset

        if self.configer.exists('data', 'use_dt_offset') or self.configer.exists('data', 'pred_dt_offset'):
            """
            dt-offset manner:
            load both the ground-truth label and offset (based on distance transform).
            """   
            Log.info('use distance transform based offset loader for val ...')
            klass = DTOffsetLoader

        elif self.configer.get('method') == 'fcn_segmentor':
            """
            default manner:
            load the ground-truth label.
            """   
            Log.info('use DefaultLoader for val ...')
            klass = DefaultLoader
        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None

        loader, sampler = self.get_dataloader_sampler(klass, 'val', dataset)
        valloader = data.DataLoader(
            loader,
            sampler=sampler,
            batch_size=self.configer.get('val', 'batch_size') // get_world_size(), pin_memory=True,
            num_workers=self.configer.get('data', 'workers'), shuffle=False,
            collate_fn=lambda *args: collate(
                *args, trans_dict=self.configer.get('val', 'data_transformer')
            )
        )
        return valloader

    def get_testloader(self, dataset=None):
            dataset = 'test' if dataset is None else dataset
            if self.configer.exists('data', 'use_sw_offset') or self.configer.exists('data', 'pred_sw_offset'):
                Log.info('use sliding window based offset loader for test ...')
                test_loader = data.DataLoader(
                    SWOffsetTestLoader(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                                       img_transform=self.img_transform,
                                       configer=self.configer),
                    batch_size=self.configer.get('test', 'batch_size'), pin_memory=True,
                    num_workers=self.configer.get('data', 'workers'), shuffle=False,
                    collate_fn=lambda *args: collate(
                        *args, trans_dict=self.configer.get('test', 'data_transformer')
                    )
                )
                return test_loader

            elif self.configer.get('method') == 'fcn_segmentor':
                Log.info('use CSDataTestLoader for test ...')
                test_loader = data.DataLoader(
                    CSDataTestLoader(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                                     img_transform=self.img_transform,
                                     configer=self.configer),
                    batch_size=self.configer.get('test', 'batch_size'), pin_memory=True,
                    num_workers=self.configer.get('data', 'workers'), shuffle=False,
                    collate_fn=lambda *args: collate(
                        *args, trans_dict=self.configer.get('test', 'data_transformer')
                    )
                )
                return test_loader


if __name__ == "__main__":
    pass
