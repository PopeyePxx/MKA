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
    parser.add_argument('--data_root', type=str, default='//')
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
    init_pose = [184.94, 254.96, 377.92, 0.8728, 1.442, 0.3738]
    robot.robot_pose_control(init_pose)
    #time.sleep(5)
    #读取当前末端位姿
    _, pose = robot.robot_msg_recv()
    # # -------------------grasp type--------------------
    graspindex = 2
    graspfile = 'E:/shiyan/grasptype_yinshi'
    values0 = find_grasstype_values(graspfile, graspindex)  # croase grasp
    angles = np.array(values0)
    angles = [int(angle) for angle in angles]

    aff_list = ['hold', "press", "click", "clamp", "grip", "open"]

    # #  -------------------计算末端与手之间的变换矩阵----------
    # 量出来的物体上功能手指对应关键点 利用预先得到的数据求解手与末端之间的变换矩阵
    flashlightfun_global_liang = tuple([490.32 + 150, 267.83 + 40, 421.57 + 40])
    flashlightfun_global_liang = np.array(flashlightfun_global_liang)
    # 量出来的物体上小拇指对应关键点
    flashlightxiao_global_liang = tuple([490.32 + 150, 267.83 + 40, 421.57 - 40])
    flashlightxiao_global_liang = np.array(flashlightxiao_global_liang)
    # 量出来的物体上手腕对应真实关键点
    flashlightshouwan_true_global_liang = tuple([490.32, 267.83, 421.57])
    flashlightshouwan_true_global_liang = np.array(flashlightshouwan_true_global_liang)
    # 计算手与工具间的变换矩阵,先求解手与末端之间的变换矩阵
    # 建立物体坐标系，用于初始化计算
    # 两个基轴
    origin_o_x_liang = flashlightfun_global_liang - flashlightshouwan_true_global_liang
    origin_o_y_liang = flashlightxiao_global_liang - flashlightshouwan_true_global_liang
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

    # ------------------用相机求解物体上关键点对应的在基座坐标系下的坐标-------
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

    #a = 1
    # #物体上功能手指对应位置
    flashlightfunc_center = kp[0]
    flashlightf_x, flashlightf_y = flashlightfunc_center[0].item(), flashlightfunc_center[1].item()
    # flashlightf_x = int(round(flashlightf_x * (640 / 448)))
    # flashlightf_y = int(round(flashlightf_y * (480 / 448)))
    flashlightf_y1 = flashlightf_y
    flashlightf_x1 = flashlightf_x
    flashlightf_x = int(round(flashlightf_y1 * (640 / 448)))
    flashlightf_y = int(round(flashlightf_x1 * (480 / 448)))
    flashlightfunc_center = [flashlightf_x, flashlightf_y]
    # 将目标像素xy坐标转换至相机坐标系下
    # 获取对齐的图像与相机内参
    flashlightdis_fun = aligned_depth_frame.get_distance(flashlightf_x, flashlightf_y)
    flashlightcamera_xyz_fun = rs.rs2_deproject_pixel_to_point(
        depth_intrin, flashlightfunc_center, flashlightdis_fun) # 计算相机坐标系的xyz

    flashlightmatrix3_fun = np.array([[flashlightcamera_xyz_fun[0]], [flashlightcamera_xyz_fun[1]], [flashlightcamera_xyz_fun[2]], [1]])
    flashlightfun_global = np.array(calculate_result_matrix(flashlightmatrix3_fun))
    flashlightfun_global[0] = flashlightfun_global[0] * 1000
    flashlightfun_global[1] = flashlightfun_global[1] * 1000
    flashlightfun_global[2] = flashlightfun_global[2] * 1000


    # 物体上小拇指对应位置
    flashlightxiao_center = kp[1]
    flashlightxiao_x, flashlightxiao_y = flashlightxiao_center[0].item(), flashlightxiao_center[1].item()
    flashlightxiao_x1 = flashlightxiao_x
    flashlightxiao_y1 = flashlightxiao_y
    flashlightxiao_x = int(flashlightxiao_y1/448*640)
    flashlightxiao_y = int(flashlightxiao_x1/448*480)
    flashlightxiao_center = [flashlightxiao_x, flashlightxiao_y]
    # 将目标像素xy坐标转换至相机坐标系下
    # 获取对齐的图像与相机内参
    flashlightdis_xiao = aligned_depth_frame.get_distance(flashlightxiao_x, flashlightxiao_y)
    flashlightcamera_xyz_xiao = rs.rs2_deproject_pixel_to_point(
        depth_intrin, flashlightxiao_center, flashlightdis_xiao)  # 计算相机坐标系的xyz

    flashlightmatrix3_xiao = np.array([[flashlightcamera_xyz_xiao[0]], [flashlightcamera_xyz_xiao[1]], [flashlightcamera_xyz_xiao[2]], [1]])
    flashlightxiao_global = np.array(calculate_result_matrix(flashlightmatrix3_xiao))
    flashlightxiao_global[0] = flashlightxiao_global[0] * 1000
    flashlightxiao_global[1] = flashlightxiao_global[1] * 1000
    flashlightxiao_global[2] = flashlightxiao_global[2] * 1000


    # 物体上手腕对应投影点
    flashlightshouwan_center = kp[2]
    flashlightshouwan_x, flashlightshouwan_y = flashlightshouwan_center[0].item(), flashlightshouwan_center[1].item()
    flashlightshouwan_x1 = flashlightshouwan_x
    flashlightshouwan_y1 = flashlightshouwan_y
    flashlightshouwan_x = int(flashlightshouwan_y1/448*640)
    flashlightshouwan_y = int(flashlightshouwan_x1/448*480)
    flashlightshouwan_center = [flashlightshouwan_x, flashlightshouwan_y]
    # 将目标像素xy坐标转换至相机坐标系下
    # 获取对齐的图像与相机内参
    flashlightdis_shouwan = aligned_depth_frame.get_distance(flashlightshouwan_x, flashlightshouwan_y)
    flashlightcamera_xyz_shouwan = rs.rs2_deproject_pixel_to_point(
        depth_intrin, flashlightshouwan_center, flashlightdis_shouwan)  # 计算相机坐标系的xyz

    flashlightmatrix3_shouwan = np.array([[flashlightcamera_xyz_shouwan[0]], [flashlightcamera_xyz_shouwan[1]], [flashlightcamera_xyz_shouwan[2]], [1]])
    flashlightshouwan_global = np.array(calculate_result_matrix(flashlightmatrix3_shouwan))
    flashlightshouwan_global[0] = flashlightshouwan_global[0] * 1000
    flashlightshouwan_global[1] = flashlightshouwan_global[1] * 1000
    flashlightshouwan_global[2] = flashlightshouwan_global[2] * 1000

    #实际的物体上的手腕关键点,该点为物体坐标系原点
    #设p:W_o->W_o'
    #根据实际情况测量
    # p=tuple([-120,30,0])
    # p=np.array(p)
    # flashlightshouwan_true_global = flashlightshouwan_global + p


    # #----------------------调整生成的点，以方便下一步操作------------------
    #以点0为基准点，利用点1 和点2，求出对应的手腕关键点以及更新后的点1
    #经测量，实际手部三角形三条边长度(mm)如下：
    #功能点与手腕点
    len_F_W = 160
    #功能点与小拇指关键点
    len_F_L = 70
    #小拇指关键点与手腕关键点
    len_L_W = 160
    #计算生成的点0与点1的向量0->1
    xiangliang_0_1 = flashlightxiao_global - flashlightfun_global
    #归一化
    xiangliang_0_1 = xiangliang_0_1 / np.linalg.norm(xiangliang_0_1)
    #求更新后的点1，作为新的小拇指关键点
    flashlightxiao_global = flashlightfun_global + xiangliang_0_1 * len_F_L
    #取出更新后的小拇指关键点，分别为x_1_pie  y_1_pie  z_1_pie
    x_1_pie = flashlightxiao_global[0]
    y_1_pie = flashlightxiao_global[1]
    z_1_pie = flashlightxiao_global[2]
    #向量 0->1‘
    xiangliang_0_1_pie = flashlightxiao_global - flashlightfun_global
    #取出其中的a_0_1_pie  b_0_1_pie c_0_1_pie
    a_0_1_pie = xiangliang_0_1_pie[0]
    b_0_1_pie = xiangliang_0_1_pie[1]
    c_0_1_pie = xiangliang_0_1_pie[2]
    #向量 0->2
    xiangliang_0_2 = flashlightshouwan_global - flashlightfun_global
    #计算0 1’ 2构成的平面的法向量
    xiangliang_fa_0_1_pie_2 = np.cross(xiangliang_0_1_pie,xiangliang_0_2)
    #单位化
    xiangliang_fa_0_1_pie_2 = xiangliang_fa_0_1_pie_2 / np.linalg.norm(xiangliang_fa_0_1_pie_2)
    #取出法向量中的n1 n2 n3
    n_1 = xiangliang_fa_0_1_pie_2[0]
    n_2 = xiangliang_fa_0_1_pie_2[1]
    n_3 = xiangliang_fa_0_1_pie_2[2]

    #取出功能关键点x_0 y_0 z_0
    x_0 = flashlightfun_global[0]
    y_0 = flashlightfun_global[1]
    z_0 = flashlightfun_global[2]
    #利用之前测量好的手部三角形三条边求解叉乘01' 0W后向量的模m
    cos_jiao_W_0_1_pie = (len_F_L**2 + len_F_W**2 - len_L_W**2) / (2 * len_F_L * len_F_W)
    sin_jiao_W_0_1_pie = np.sqrt(1-cos_jiao_W_0_1_pie**2)
    m = len_F_L * len_F_W * sin_jiao_W_0_1_pie
    #求解手腕关键点
    #利用推到的公式，先算出必要的符号
    A = (x_0**2-x_1_pie**2+y_0**2-y_1_pie**2+z_0**2-z_1_pie**2+len_L_W**2-len_F_W**2)
    E = x_0 - x_1_pie
    F = 2*(y_0 - y_1_pie)+2*c_0_1_pie*(z_0-z_1_pie)/b_0_1_pie
    G = 2*(z_0 - z_1_pie)*(m*n_1+b_0_1_pie*z_0-c_0_1_pie*y_0)/b_0_1_pie
    B = m*n_3 + a_0_1_pie*y_0 - b_0_1_pie * x_0
    #求解出手腕真实的关键点对应的世界坐标
    flashlightshouwan_true_global = tuple([1,1,1])
    flashlightshouwan_true_global = np.array(flashlightshouwan_true_global)
    flashlightshouwan_true_global[0] = a_0_1_pie/b_0_1_pie*(((A-G)/(2*E))+B/b_0_1_pie)/(F/(2*E)+a_0_1_pie/b_0_1_pie)-B/b_0_1_pie
    flashlightshouwan_true_global[1] = (((A-G)/(2*E))+B/b_0_1_pie)/(F/(2*E)+a_0_1_pie/b_0_1_pie)
    flashlightshouwan_true_global[2] = (m*n_1+b_0_1_pie*z_0-c_0_1_pie*y_0)/b_0_1_pie+c_0_1_pie/b_0_1_pie*(((A-G)/(2*E))+B/b_0_1_pie)/(F/(2*E)+a_0_1_pie/b_0_1_pie)

    a = 1
    #小验证，验证E H的正确性
    # flashlightfun_global = tuple([489.21 + 150, 261.93, 421.9 + 60])  639.21  261.93  481.9
    # flashlightfun_global = np.array(flashlightfun_global)
    # flashlightxiao_global = tuple([489.21 + 150,261.93+20 , 421.9 - 10])  639.21  281.93  411.9
    # flashlightxiao_global = np.array(flashlightxiao_global)
    # flashlightshouwan_true_global = tuple([489.21, 261.93, 421.9])
    # flashlightshouwan_true_global = np.array(flashlightshouwan_true_global)   489.21  261.93  421.9

    # #------------------------------------建立物体坐标系--------------------------
    # 两个基轴
    origin_o_x = flashlightfun_global - flashlightshouwan_true_global
    origin_o_y = flashlightxiao_global - flashlightshouwan_true_global
    # 标准化
    normalized_o_x = origin_o_x / np.linalg.norm(origin_o_x)
    normalized_o_y = origin_o_y / np.linalg.norm(origin_o_y)

    origin_o_z = np.cross(normalized_o_x, normalized_o_y)
    normalized_o_z = origin_o_z / np.linalg.norm(origin_o_z)

    # 物体坐标系保留x,z
    final_o_y = np.cross(normalized_o_z, normalized_o_x)

    # 建立I系与O系之间的旋转矩阵 R I <-O
    rotation_I_O = np.column_stack((normalized_o_x, final_o_y, normalized_o_z))
   
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
    # b = flashlightshouwan_true_global[0]
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
    tool_pingyi = [flashlightshouwan_true_global[0],flashlightshouwan_true_global[1],flashlightshouwan_true_global[2],euler_liang[0],
                 euler_liang[1], euler_liang[2]]
    robot.robot_pose_control(tool_pingyi)
    time.sleep(10)
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
