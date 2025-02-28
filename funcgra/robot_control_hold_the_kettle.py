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
    parser.add_argument('--data_root', type=str, default='/')
    parser.add_argument('--model_file', type=str,
                        default='/.pth')
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

class InspireHandR:

def transform_image(image, crop_size):

    return transformed_img


if __name__ == "__main__":
    hand = InspireHandR()
    robot = Robot()
    #time.sleep(5)
    init_pose = [262.85, 194.76, 666.52, 0.0777, -4.8657, -0.3986]
    robot.robot_pose_control(init_pose)
    #time.sleep(5)
    #读取当前末端位姿
    _, pose = robot.robot_msg_recv()
    # # -------------------grasp type--------------------
    graspindex = 1
    graspfile = 'E:/shiyan/grasptype_yinshi'
    values0 = find_grasstype_values(graspfile, graspindex)  # croase grasp
    angles = np.array(values0)
    angles = [int(angle) for angle in angles]

    aff_list = ['hold', "press", "click", "clamp", "grip", "open"]

    # #  -------------------计算末端与手之间的变换矩阵----------
    # 量出来的物体上功能手指对应关键点 利用预先得到的数据求解手与末端之间的变换矩阵
    kettle_fun_global_liang = tuple([490.32 + 150, 267.83 + 40, 421.57 + 40])
    kettle_fun_global_liang = np.array(kettle_fun_global_liang)
    # 量出来的物体上小拇指对应关键点
    kettle_xiao_global_liang = tuple([490.32 + 150, 267.83 + 40, 421.57 - 40])
    kettle_xiao_global_liang = np.array(kettle_xiao_global_liang)
    # 量出来的物体上手腕对应真实关键点
    kettle_shouwan_true_global_liang = tuple([490.32, 267.83, 421.57])
    kettle_shouwan_true_global_liang = np.array(kettle_shouwan_true_global_liang)
    # 计算手与工具间的变换矩阵,先求解手与末端之间的变换矩阵
    # 建立物体坐标系，用于初始化计算
    # 两个基轴
    origin_o_x_liang = kettle_fun_global_liang - kettle_shouwan_true_global_liang
    origin_o_y_liang = kettle_xiao_global_liang - kettle_shouwan_true_global_liang
    # 标准化
    normalized_o_x_liang = origin_o_x_liang / np.linalg.norm(origin_o_x_liang)
    normalized_o_y_liang = origin_o_y_liang / np.linalg.norm(origin_o_y_liang)

    origin_o_z_liang = np.cross(normalized_o_x_liang, normalized_o_y_liang)
    normalized_o_z_liang = origin_o_z_liang / np.linalg.norm(origin_o_z_liang)

    # 物体坐标系保留x,z
    final_o_y_liang = np.cross(normalized_o_z_liang, normalized_o_x_liang)

    # 建立I系与O系之间的旋转矩阵 R I <-O
    rotation_I_O_liang = np.column_stack((normalized_o_x_liang, final_o_y_liang, normalized_o_z_liang))

    # 求O系与H系之间的旋转矩阵
    # 判断rotation_
    # 末端应该有一个坐标系，记为E
    # 利用量出来的手部关键点(R I<-H)以及读取出来的面板上的数据(T I<-E)求解R E<-H
    # 先利用已知的数据计算基座与末端之间的旋转矩阵 R I<-E
    rotation_I_E_liang = euler_to_rotation_matrix(1.1563, 1.1614, 1.2048)
    # 再计算R E<-H
    rotation_E_I_liang = rotation_I_E_liang.T
    # 先给个值之后记得去掉
    # rotation_I_H= np.eye(3)
    # rotation_I_O = np.eye(3)
    # rotation_E_H = np.dot(rotation_E_I,rotation_I_H)
    # 直接利用测量好的物体一开始对应的位置算出手与末端之间的变换矩阵
    rotation_E_H_liang = np.dot(rotation_E_I_liang, rotation_I_O_liang)



    # # ------------------用相机求解物体上关键点对应的在基座坐标系下的坐标-------
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
    #
    # #物体上功能手指对应位置
    # kettle_func_center = kp[0]
    # kettle_f_x, kettle_f_y = kettle_func_center[0].item(), kettle_func_center[1].item()
    # # kettle_f_x = int(round(kettle_f_x * (640 / 448)))
    # # kettle_f_y = int(round(kettle_f_y * (480 / 448)))
    # kettle_f_y1 = kettle_f_y
    # kettle_f_x1 = kettle_f_x
    # kettle_f_x = int(round(kettle_f_y1 * (640 / 448)))
    # kettle_f_y = int(round(kettle_f_x1 * (480 / 448)))
    # kettle_func_center = [kettle_f_x, kettle_f_y]
    # # 将目标像素xy坐标转换至相机坐标系下
    # # 获取对齐的图像与相机内参
    # kettle_dis_fun = aligned_depth_frame.get_distance(kettle_f_x, kettle_f_y)
    # kettle_camera_xyz_fun = rs.rs2_deproject_pixel_to_point(
    #     depth_intrin, kettle_func_center, kettle_dis_fun) # 计算相机坐标系的xyz
    #
    # kettle_matrix3_fun = np.array([[kettle_camera_xyz_fun[0]], [kettle_camera_xyz_fun[1]], [kettle_camera_xyz_fun[2]], [1]])
    # kettle_fun_global = np.array(calculate_result_matrix(kettle_matrix3_fun))
    #
    #
    #
    # # 物体上小拇指对应位置
    # kettle_xiao_center = kp[1]
    # kettle_xiao_x, kettle_xiao_y = kettle_xiao_center[0].item(), kettle_xiao_center[1].item()
    # kettle_xiao_x1 = kettle_xiao_x
    # kettle_xiao_y1 = kettle_xiao_y
    # kettle_xiao_x = int(kettle_xiao_y1/448*640)
    # kettle_xiao_y = int(kettle_xiao_x1/448*480)
    # kettle_xiao_center = [kettle_xiao_x, kettle_xiao_y]
    # # 将目标像素xy坐标转换至相机坐标系下
    # # 获取对齐的图像与相机内参
    # kettle_dis_xiao = aligned_depth_frame.get_distance(kettle_xiao_x, kettle_xiao_y)
    # kettle_camera_xyz_xiao = rs.rs2_deproject_pixel_to_point(
    #     depth_intrin, kettle_xiao_center, kettle_dis_xiao)  # 计算相机坐标系的xyz
    #
    # kettle_matrix3_xiao = np.array([[kettle_camera_xyz_xiao[0]], [kettle_camera_xyz_xiao[1]], [kettle_camera_xyz_xiao[2]], [1]])
    # kettle_xiao_global = np.array(calculate_result_matrix(kettle_matrix3_xiao))
    #
    #
    #
    # # 物体上手腕对应投影点
    # kettle_shouwan_center = kp[2]
    # kettle_shouwan_x, kettle_shouwan_y = kettle_shouwan_center[0].item(), kettle_shouwan_center[1].item()
    # kettle_shouwan_x1 = kettle_shouwan_x
    # kettle_shouwan_y1 = kettle_shouwan_y
    # kettle_shouwan_x = int(kettle_shouwan_y1/448*640)
    # kettle_shouwan_y = int(kettle_shouwan_x1/448*480)
    # kettle_shouwan_center = [kettle_shouwan_x, kettle_shouwan_y]
    # # 将目标像素xy坐标转换至相机坐标系下
    # # 获取对齐的图像与相机内参
    # kettle_dis_shouwan = aligned_depth_frame.get_distance(kettle_shouwan_x, kettle_shouwan_y)
    # kettle_camera_xyz_shouwan = rs.rs2_deproject_pixel_to_point(
    #     depth_intrin, kettle_shouwan_center, kettle_dis_shouwan)  # 计算相机坐标系的xyz
    #
    # kettle_matrix3_shouwan = np.array([[kettle_camera_xyz_shouwan[0]], [kettle_camera_xyz_shouwan[1]], [kettle_camera_xyz_shouwan[2]], [1]])
    # kettle_shouwan_global = np.array(calculate_result_matrix(kettle_matrix3_shouwan))
    #
    #
    # #实际的物体上的手腕关键点,该点为物体坐标系原点
    # #设p:W_o->W_o'
    # #根据实际情况测量
    # p=tuple([110,40,0])
    # p=np.array(p)
    # kettle_shouwan_true_global = kettle_shouwan_global + p


    #小验证，验证E H的正确性 修改实际的值
    kettle_fun_global = tuple([491.48 + 150, 255.31 + 50, 417.16 + 40])
    kettle_fun_global = np.array(kettle_fun_global)
    kettle_xiao_global = tuple([491.48 + 150, 255.31 + 40, 417.16 - 40])
    kettle_xiao_global = np.array(kettle_xiao_global)
    kettle_shouwan_true_global = tuple([491.48, 264.8, 417.16+20])
    kettle_shouwan_true_global = np.array(kettle_shouwan_true_global)



    # #------------------------------------建立物体坐标系--------------------------
    # 两个基轴
    origin_o_x = kettle_fun_global - kettle_shouwan_true_global
    origin_o_y = kettle_xiao_global - kettle_shouwan_true_global
    # 标准化
    normalized_o_x = origin_o_x / np.linalg.norm(origin_o_x)
    normalized_o_y = origin_o_y / np.linalg.norm(origin_o_y)

    origin_o_z = np.cross(normalized_o_x, normalized_o_y)
    normalized_o_z = origin_o_z / np.linalg.norm(origin_o_z)

    # 物体坐标系保留x,z
    final_o_y = np.cross(normalized_o_z, normalized_o_x)

    # 建立I系与O系之间的旋转矩阵 R I <-O
    rotation_I_O = np.column_stack((normalized_o_x, final_o_y, normalized_o_z))


    #
    #
    # #手腕关键点
    # hand_position_w = tuple([float(pose[0]),float(pose[1]),float(pose[2])])
    # hand_position_w = np.array(hand_position_w)
    # # #功能手指关键点，这里为食指对应区域
    # # #根据实际情况加减
    # p_gongneng = tuple([160,40,50])
    # hand_position_shizhi = hand_position_w + np.array(p_gongneng)
    # #
    # # #小拇指关键点
    # # #根据实际情况加减
    # p_xiao = tuple([170,-40,0])
    # hand_position_xiao = hand_position_w + np.array( p_xiao)
    #

    #
    # #建立手部坐标系
    # #两个基轴
    # origin_h_x = hand_position_shizhi - hand_position_w
    # origin_h_y = hand_position_xiao - hand_position_w
    #
    # #标准化
    # normalized_h_x = origin_h_x / np.linalg.norm(origin_h_x)
    # normalized_h_y = origin_h_y / np.linalg.norm(origin_h_y)
    #
    # origin_h_z = np.cross(normalized_h_x,normalized_h_y)
    # normalized_h_z = origin_h_z / np.linalg.norm(origin_h_z)
    #
    # #手部坐标系中保留x,z
    # final_h_y = np.cross(normalized_h_z,normalized_h_x)
    #
    # #建立I系与H系之间的旋转矩阵 R I<-H
    # rotation_I_H = np.column_stack((normalized_h_x,final_h_y,normalized_h_z))



    # #利用R I<-O R E<-H求解当手部关键点与物体上对应关键点重合的时候的R I <- E
    rotation_H_E_liang = rotation_E_H_liang.T
    rotation_I_E_final_liang = np.dot(rotation_I_O,rotation_H_E_liang)
    #
    #将最终旋转矩阵转为欧拉角
    euler_liang = rotation_matrix_to_euler_angles(rotation_I_E_final_liang)
    # b = kettle_shouwan_true_global[0]
    #c = euler[0]
    #平移向量  直接用物体上对应的手腕真实点即可

    #验证R EH是否正确
    # rotation_I_E = euler_to_rotation_matrix(float(pose[3]), float(pose[4]), float(pose[5]))
    # rotation_

    #a=1

    tool_pos = [float(pose[0]) , float(pose[1]) ,float(pose[2]) , euler_liang[0],
                 euler_liang[1], euler_liang[2]]


    # 令机械臂移动到工具位置处
    robot.robot_pose_control(tool_pos)
    #time.sleep(7)
    #机械手平移过去
    tool_pingyi = [kettle_shouwan_true_global[0],kettle_shouwan_true_global[1],kettle_shouwan_true_global[2],euler_liang[0],
                 euler_liang[1], euler_liang[2]]
    robot.robot_pose_control(tool_pingyi)
    time.sleep(17)
    #
    # 执行粗手势
    hand.setangle(*angles)
    #robot.robot_pose_control(pose)
    #
    # —————————————————稳定性抓取判断——————————————————————
    # Sampling rate is 50 Hz, so the interval is 1/50 seconds
    sampling_interval = 1/10   # seconds
    # Store the previous forces
    pow = hand.get_actforce()
    prev_forces = [pow[0], pow[1], pow[2], pow[3], pow[4], pow[5]]
    forces_derivative = 100 * prev_forces  # Initialize to infinity

    start_time = time.time()  # Start the timer for angle adjustments
    last_force_check = time.time()  # Timer for force measurements
    while True:
        # Update finger angles continuously


        # Optional: short sleep to prevent excessive CPU usage, can be adjusted or removed
        time.sleep(0.01)
