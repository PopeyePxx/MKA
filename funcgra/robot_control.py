import serial
import struct
import socket
import numpy as np
import time
import pyrealsense2 as rs
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import cv2
import torch
import numpy as np
from models.keypoint import Net as model
from utils.viz import viz_pred_test_kp
from utils.util import set_seed, process_gt, normalize_map
from torchvision import transforms
from PIL import Image
from sympy import symbols

# 机械臂IP地址和端口，不变量
HOST = "192.168.3.6"
PORT = 30003

# 定义机械臂的常量
tool_acc = 0.4  # Safe: 0.5
tool_vel = 0.05  # Safe: 0.2
PI = 3.141592653589
fmt1 = '<I'
fmt2 = '<6d'
BUFFER_SIZE = 1108
buffsize = 1108

#--------------camera----------------------#

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)
def get_aligned_images():

    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame
#--------------camera----------------------#
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    ##  path
    parser.add_argument('--data_root', type=str, default='/data1/yf/test_vis/')
    parser.add_argument('--model_file', type=str,
                        default='/ .pth')
    parser.add_argument('--save_path', type=str, default='./save_preds')
    parser.add_argument("--divide", type=str, default="Seen")
    ##  image
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--resize_size', type=int, default=256)
    #### test
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument('--test_num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--viz', action='store_true', default=True)

    args = parser.parse_args()
    return args

def find_grasstype_values(filename, grasstype):

    return None

def compute_derivative(forces, prev_forces, dt):

    return [(f - pf) / dt for f, pf in zip(forces, prev_forces)]

import numpy as np

# 1.机械臂末端位姿转换为4*4齐次变换矩阵T1
def euler_to_rotation_matrix(roll, pitch, yaw):

    return R

def transformation_matrix(roll, pitch, yaw, x, y, z):


    return transformation_matrix

def calculate_result_matrix(matrix3):


    return result_matrix[0][0], result_matrix[1][0], result_matrix[2][0]


class Robot:
    def __init__(self, host=None, port=None):


    # 控制机械臂末端位姿
    def robot_pose_control(self, target_tcp):  # (x,y,z,rx,ry,rz)

        self.tcp_socket.send(str.encode(tcp_command))

    # 控制机械臂关节角
    def robot_angle_control(self, target_tcp):

        self.tcp_socket.send(str.encode(tcp_command))

    # 接收机械臂信息
    def robot_msg_recv(self):

            return new_data1_str, new_data2_str

    def byte_swap(self, data):
        return data[::-1]

    def robot_close(self):
        self.tcp_socket.close()

class InspireHandR:

    def __init__(self):


    # 把数据分成高字节和低字节
    def data2bytes(self, data):

        return rdata

    # 把十六进制或十进制的数转成bytes
    def num2str(self, num):

        # print(str)
        return str

    # 求校验和
    def checknum(self, data, leng):

        return result

    # 设置电机驱动位置
    def setpos(self, pos1, pos2, pos3, pos4, pos5, pos6):

            return




    # 设置弯曲角度
    def setangle(self, angle1, angle2, angle3, angle4, angle5, angle6):

            return



    # 设置小拇指弯曲角度
    def setlittleangle(self, angle1):

            return


    # 设置食指弯曲角度
    def setringangle(self, angle1):

            return



    # 设置中指弯曲角度
    def setmiddleangle(self, angle1):

            return

    # 设置食指弯曲角度
    def setindexangle(self, angle1):
            return


    # 设置大拇指弯曲角度
    def setthumbangle(self, angle1):

            return

    # 设置侧摆角度
    def setswingangle(self, angle1):

            return

    # 设置力控阈值 安全值200
    def setpower(self, power1, power2, power3, power4, power5, power6):
            return


    # 设置运动速度 安全值200
    def setspeed(self, speed1, speed2, speed3, speed4, speed5, speed6):
            return


    # 读取驱动器实际的位置值
    def get_setpos(self):

        return setpos

    # 读取设置角度
    def get_setangle(self):

        return setangle

    # 读取驱动器设置的力控阈值
    def get_setpower(self):

        return setpower

    # 读取驱动器实际的位置值
    def get_actpos(self):

        return actpos

    # 读取力度信息
    def get_actforce(self):

        return actforce

    # 读取实际的角度值
    def get_actangle(self):

        return actangle

    # 读取小拇指实际的受力
    def get_little_actforce(self):

        return actforce

    # 读取无名指实际的受力
    def get_ring_actforce(self):

        return actforce

    # 读取中指实际的受力
    def get_middle_actforce(self):

        return actforce

    # 读取食指实际的受力
    def get_index_actforce(self):

        return actforce

    # 读取大拇指实际的受力
    def get_thumb_actforce(self):

        return actforce

    # 读取手掌实际的受力
    def get_palm_actforce(self):

        return actforce

    # 读取电流
    def get_current(self):

        return current

    # 读取故障信息
    def get_error(self):

        return error

    # 读取状态信息
    def get_status(self):

        return status

    # 读取温度信息
    def get_temp(self):

        return temp

    # 清除错误
    def set_clear_error(self):


    # 保存参数到FLASH
    def set_save_flash(self):

    # 力传感器校准
    def gesture_force_clb(self):


    # 设置上电速度
    def setdefaultspeed(self, speed1, speed2, speed3, speed4, speed5, speed6):

            return



    # 设置上电力控阈值
    def setdefaultpower(self, power1, power2, power3, power4, power5, power6):

            return




    def soft_setpos(self, pos1, pos2, pos3, pos4, pos5, pos6):


    def reset(self):

        return

    def reset_0(self):

        return

    def hand_close(self):


def transform_image(image, crop_size):

    return transformed_img

if __name__ == "__main__":
    # hand = InspireHandR()
    # robot = Robot()
    # init_pose = [202.23, 115.03, 564.5, -0.82, 1.57, -0.25]
    # robot.robot_pose_control(init_pose)
    # _, pose = robot.robot_msg_recv()
    args = parse_args()
    model = model(aff_classes=6)
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))

    intr, depth_intrin, color_image0, depth_image, aligned_depth_frame = get_aligned_images()

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
        depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # Stack both images horizontally
    images = np.hstack((color_image0, depth_colormap))
    label = torch.tensor([1])
     # 将 BGR 转为 RGB
    img = Image.fromarray(color_image0, 'RGB')
    color_image = transform_image(img, 448)
    ego_pred, kp = model.func_test_forward(color_image, label.long())
    ego_pred0 = np.array(ego_pred.squeeze())
    ego_pred1 = normalize_map(ego_pred0, args.crop_size)

    viz_pred_test_kp(args, color_image, ego_pred1, kp)
    # -------------------grasp type--------------------
    graspindex = 1
    graspfile = 'E:/shiyan/grasptype_yinshi'
    values0 = find_grasstype_values(graspfile, graspindex)  # croase grasp
    angles = np.array(values0)
    angles = [int(angle) for angle in angles]

    aff_list = ['hold', "press", "click", "clamp", "grip", "open"]

    #物体上功能手指对应位置
    func_center = kp[0]
    f_x, f_y = func_center[0].item(), func_center[1].item()
    func_center = [f_x, f_y]
    # 将目标像素xy坐标转换至相机坐标系下
    # 获取对齐的图像与相机内参
    dis_fun = aligned_depth_frame.get_distance(f_x, f_y)
    camera_xyz_fun = rs.rs2_deproject_pixel_to_point(
        depth_intrin, func_center, dis_fun) # 计算相机坐标系的xyz

    matrix3_fun = np.array([[camera_xyz_fun[0]], [camera_xyz_fun[1]], [camera_xyz_fun[2]], [1]])
    fun_global = calculate_result_matrix(matrix3_fun)

    # 物体上小拇指对应位置
    xiao_center = kp[1]
    xiao_x, xiao_y = xiao_center[0].item(), xiao_center[1].item()
    xiao_center = [xiao_x, xiao_y]
    # 将目标像素xy坐标转换至相机坐标系下
    # 获取对齐的图像与相机内参
    dis_xiao = aligned_depth_frame.get_distance(xiao_x, xiao_y)
    camera_xyz_xiao = rs.rs2_deproject_pixel_to_point(
        depth_intrin, xiao_center, dis_xiao)  # 计算相机坐标系的xyz

    matrix3_xiao = np.array([[camera_xyz_xiao[0]], [camera_xyz_xiao[1]], [camera_xyz_xiao[2]], [1]])
    xiao_global = calculate_result_matrix(matrix3_xiao)

    # 物体上手腕对应投影点
    shouwan_center = kp[2]
    shouwan_x, shouwan_y = shouwan_center[0].item(), shouwan_center[1].item()
    shouwan_center = [shouwan_x, shouwan_y]
    # 将目标像素xy坐标转换至相机坐标系下
    # 获取对齐的图像与相机内参
    dis_shouwan = aligned_depth_frame.get_distance(shouwan_x, shouwan_y)
    camera_xyz_shouwan = rs.rs2_deproject_pixel_to_point(
        depth_intrin, shouwan_center, dis_shouwan)  # 计算相机坐标系的xyz

    matrix3_shouwan = np.array([[camera_xyz_shouwan[0]], [camera_xyz_shouwan[1]], [camera_xyz_shouwan[2]], [1]])
    shouwan_global = calculate_result_matrix(matrix3_shouwan)

    #实际的物体



    tool_pos = []

