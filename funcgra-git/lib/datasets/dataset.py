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