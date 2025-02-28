import isaacgym
from isaacgym import gymapi, gymutil
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

# 辅助函数：限制值在给定范围内
def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# 解析命令行参数
args = gymutil.parse_arguments(
    description="Load multiple assets in Isaac Gym",
    custom_parameters=[
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

# 初始化 Isaac Gym
gym = gymapi.acquire_gym()

# 创建仿真环境
sim_params = gymapi.SimParams()
sim_params.physx.solver_type = 1
sim_params.dt = dt = 1.0 / 60.0
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# 配置地面平面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# 加载手的资产
hand_asset_root = "/home/hun/IsaacGym_Preview_4_Package/isaacgym/yinshi_URDF_RL/urdf-five_R/robots"
hand_asset_file = "urdf-five3.urdf"
hand_asset_options = gymapi.AssetOptions()
hand_asset_options.fix_base_link = True
hand_asset_options.armature = 0.01
hand_asset = gym.load_asset(sim, hand_asset_root, hand_asset_file, hand_asset_options)

# 加载电钻的资产
drill_asset_root = "/home/hun/IsaacGym_Preview_4_Package/isaacgym/LDS666888"
drill_asset_file = "LDS666888.urdf"
drill_asset_options = gymapi.AssetOptions()
drill_asset_options.fix_base_link = False  # 根据需要设置是否固定基座
drill_asset_options.armature = 0.01
drill_asset = gym.load_asset(sim, drill_asset_root, drill_asset_file, drill_asset_options)

# 获取DOF名称和属性
dof_names = gym.get_asset_dof_names(hand_asset)
dof_props = gym.get_asset_dof_properties(hand_asset)

# 初始化DOF状态数组
num_dofs = gym.get_asset_dof_count(hand_asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
dof_positions = dof_states['pos']
dof_positions2 = dof_states['pos']
# 获取DOF类型和其他属性
dof_types = [gym.get_asset_dof_type(hand_asset, i) for i in range(num_dofs)]
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
lower_limits2 = dof_props['lower']
upper_limits = dof_props['upper']
upper_limits2 = dof_props['upper']

# 初始化默认位置、速度
defaults = np.zeros(num_dofs)
defaults2 = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
speeds2 = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            lower_limits2[i] = clamp(lower_limits2[i],-math.pi/6, math.pi/6)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
            upper_limits2[i] = clamp(upper_limits2[i], -math.pi/6, math.pi/6)
        # make sure our default position is in range
        if lower_limits[i] > 0.0:
            defaults[i] = lower_limits[i]
        if lower_limits2[i] > 0.0:
             defaults2[i] = lower_limits2[i]
        if upper_limits2[i] < 0.0:
            defaults2[i] = upper_limits2[i]
        elif upper_limits[i] < 0.0:
            defaults[i] = upper_limits[i]

    else:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = -math.pi
            lower_limits2[i] = -math.pi/6
            upper_limits[i] = math.pi
            upper_limits2[i] = math.pi/6
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
    dof_positions[i] = defaults[i]
    dof_positions2[i] = defaults2[i]
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
        speeds2[i] = args.speed_scale * clamp(2 * (upper_limits2[i] - lower_limits2[i]), 0.25 * math.pi/2, 3.0 * math.pi/2)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)



# 设置环境网格参数
num_envs = 1
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# 缓存一些常用的句柄以供后续使用
envs = []
hand_handles = []
drill_handles = []
markers = []

# 创建并填充环境
env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
envs.append(env)

# 设置手的位置和姿态
hand_pose = gymapi.Transform()
hand_pose.p = gymapi.Vec3(0.85, 1.1, 0.25)  # 设置手的位置
quart_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi)  # 设置手的姿态
quart_z = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi / 2)  # 设置手的姿态
quart_test = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi/2)  # 设置手的姿态
final = quart_x * quart_z
hand_pose.r = final
# 将gymapi.Quat转换为numpy数组以便于计算
def quat_to_array(q):
    return np.array([q.x, q.y, q.z, q.w])

# 将四元数转换为旋转矩阵
final_quat_arr = quat_to_array(final)
quat_test=quat_to_array(quart_test)
#旋转矩阵R I<-H
rotation_matrix_I_H_final = R.from_quat(final_quat_arr).as_matrix()

rotation_test=R.from_quat(quat_test).as_matrix()
hand_handle = gym.create_actor(env, hand_asset, hand_pose, "hand_actor", 50, 1)
hand_handles.append(hand_handle)

# 设置电钻的位置和姿态
drill_pose = gymapi.Transform()
drill_pose.p = gymapi.Vec3(1, 1, 0)  # 设置电钻的位置
drill_handle = gym.create_actor(env, drill_asset, drill_pose, "drill_actor", 50, 1)
drill_handles.append(drill_handle)

# 获取手和电钻的刚体形状属性
hand_shape_props = gym.get_actor_rigid_shape_properties(env, hand_handle)
drill_shape_props = gym.get_actor_rigid_shape_properties(env, drill_handle)

# 设置手部刚体属性
for prop in hand_shape_props:
    prop.filter = 0b0001  # 手部碰撞组为1，二进制表示
    prop.restitution = 0.0  # 设置恢复系数
    prop.friction = 1000.0  # 设置摩擦系数
    prop.contact_offset = 0.0  # 设置接触偏移量
    prop.rest_offset = 0.0000000  # 设置静止偏移量

# # 设置电钻刚体属性
# for prop in drill_shape_props:
#     prop.filter = 0b0010  # 电钻碰撞组为2，二进制表示
#     prop.restitution = 0.1  # 设置恢复系数
#     prop.friction = 1.0  # 设置摩擦系数
#     prop.contact_offset = 0.0  # 设置接触偏移量
#     prop.rest_offset = 0.0  # 设置静止偏移量

# 应用修改后的属性
gym.set_actor_rigid_shape_properties(env, hand_handle, hand_shape_props)
#gym.set_actor_rigid_shape_properties(env, drill_handle, drill_shape_props)

# # 启用自碰撞检测
# for i in range(len(hand_shape_props)):
#     hand_shape_props[i].filter = 0b0001 | (i << 2)  # 每个刚体都有不同的过滤器值
#
# gym.set_actor_rigid_shape_properties(env, hand_handle, hand_shape_props)

# 获取 DOF 属性
dof_props = gym.get_actor_dof_properties(env, hand_handle)

# 获取手部 actor 的 DOF 数量
num_dofs = gym.get_actor_dof_count(env, hand_handle)
print(f"Number of DOFs: {num_dofs}")

# 获取手部 actor 的 DOF 属性
dof_props = gym.get_actor_dof_properties(env, hand_handle)

# 设置每个关节的目标位置，这里以 0.5 弧度为例
target_positions = [0.5] * num_dofs  # 根据实际情况调整目标位置

# 创建一个数组来存储所有目标位置
dof_position_targets = np.array(target_positions, dtype=np.float32)

# 设置阻尼和摩擦系数（示例值）
for i in range(num_dofs):
    dof_props['driveMode'][i] = gymapi.DOF_MODE_POS  # 设置为位置控制模式
    dof_props['stiffness'][i] = 1000.0  # 增加刚度
    dof_props['damping'][i] = 100.0  # 增加阻尼

# 应用修改后的属性
gym.set_actor_dof_properties(env, hand_handle, dof_props)



# 创建图形窗口
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

# 定义摄像机的位置
cam_pos = gymapi.Vec3(1, 1, 0.6)  # 摄像机位置 (x, y, z)
cam_target = gymapi.Vec3(0, 0, 0)  # 目标点位置 (x, y, z)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 打印所有rigid body的名字
def print_rigid_body_names(env, actor_handle):
    num_bodies = gym.get_actor_rigid_body_count(env, actor_handle)
    for i in range(num_bodies):
        body_name = gym.get_actor_rigid_body_names(env, i)
        print(f"Rigid Body {i}: {body_name}")

print_rigid_body_names(env, hand_handle)
print_rigid_body_names(env, drill_handle)

# 创建用于标记的小球资产，并设置其为固定底座
sphere_asset_options = gymapi.AssetOptions()
sphere_asset_options.fix_base_link = True  # 设置为固定底座

# 创建小球资产
sphere_asset = gym.create_sphere(sim, radius=0.01, options=sphere_asset_options)

# 定义红色材质颜色
red_color = gymapi.Vec3(1.0, 0.0, 0.0)  # 红色 (R, G, B)


def add_marker(env, position):
    marker_pose = gymapi.Transform()
    marker_pose.p = gymapi.Vec3(position[0],position[1],position[2])

    # 创建标记并设置其为固定底座
    marker_handle = gym.create_actor(env, sphere_asset, marker_pose, "marker", 50, 1)

    # 设置刚体的颜色
    num_bodies = gym.get_actor_rigid_body_count(env, marker_handle)
    for i in range(num_bodies):
        gym.set_rigid_body_color(env, marker_handle, i, gymapi.MESH_VISUAL, red_color)

    markers.append(marker_handle)


# 获取手和电钻的第一个刚体的位置信息
hand_rb_states = gym.get_actor_rigid_body_states(env, hand_handle, gymapi.STATE_ALL)
drill_rb_states = gym.get_actor_rigid_body_states(env, drill_handle, gymapi.STATE_ALL)

# 打印第一个刚体的位置信息
print("Hand Rigid Body Position:", hand_rb_states['pose']['p'][0])
print("Drill Rigid Body Position:", drill_rb_states['pose']['p'][0])
#手腕关键点
hand_position_W=list(hand_rb_states['pose']['p'][0])
hand_position_W[0]+=0.03
hand_position_W[1]+=0.03
hand_position_W[2]-=0.012
hand_position_W=tuple(hand_position_W)

#功能手指关键点
hand_position_F=list(hand_rb_states['pose']['p'][1])

hand_position_F=tuple(hand_position_F)

#小拇指关键点
hand_position_L=list(hand_rb_states['pose']['p'][7])

hand_position_L=tuple(hand_position_L)

#物体上功能手指关键点
drill_position_F_O=list(drill_rb_states['pose']['p'][6])
drill_position_F_O[0]+=0.025

drill_position_F_O=tuple(drill_position_F_O)

#物体上小拇指关键点
drill_position_L_O=list(drill_rb_states['pose']['p'][8])
drill_position_L_O[0]-=0.025
drill_position_L_O[1]+=0.01
drill_position_L_O[2]+=0.03

drill_position_L_O=tuple(drill_position_L_O)

#物体上手腕关键点
drill_position_W_O_D=list(drill_rb_states['pose']['p'][9])
drill_position_W_O_D[0]+=0.035
drill_position_W_O_D[1]-=0.025
drill_position_W_O_D[2]+=0.085
drill_position_W_O_D=tuple(drill_position_W_O_D)

#物体上手腕关键点对应的真实关键点
drill_position_W_O=list(drill_rb_states['pose']['p'][9])
drill_position_W_O[1]-=0.025
drill_position_W_O[2]+=0.09

drill_position_W_O=tuple(drill_position_W_O)


# 在手上添加标记
add_marker(env, hand_position_W)
add_marker(env,hand_position_F)
add_marker(env,hand_position_L)

#在工具上添加标记
add_marker(env, drill_position_F_O)
add_marker(env,drill_position_L_O)
add_marker(env,drill_position_W_O)
add_marker(env,drill_position_W_O_D)


#计算手与工具之间的变换矩阵
#先计算旋转矩阵
#建立物体坐标系，
# 两个基轴
origin_o_x=np.array(drill_position_F_O)-np.array(drill_position_W_O)
origin_o_y=np.array(drill_position_L_O)-np.array(drill_position_W_O)
#标准化
normalized_o_x=origin_o_x/np.linalg.norm(origin_o_x)
normalized_o_y=origin_o_y/np.linalg.norm(origin_o_y)

origin_o_z=np.cross(normalized_o_x,normalized_o_y)
normalized_o_z=origin_o_z/np.linalg.norm(origin_o_z)

#物体坐标系中保留x z
final_o_y=np.cross(normalized_o_z,normalized_o_x)
#建立I系与O系的旋转矩阵R I<-O
rotation_I_O=np.column_stack((normalized_o_x,final_o_y,normalized_o_z))

#建立手部坐标系
#两个基轴
origin_h_x=np.array(hand_position_F)-np.array(hand_position_W)
origin_h_y=np.array(hand_position_L)-np.array(hand_position_W)
#标准化
normalized_h_x=origin_h_x/np.linalg.norm(origin_h_x)
normalized_h_y=origin_h_y/np.linalg.norm(origin_h_y)

origin_h_z=np.cross(normalized_h_x,normalized_h_y)
normalized_h_z=origin_h_z/np.linalg.norm(origin_h_z)
#手部坐标系中保留x,z
final_h_y=np.cross(normalized_h_z,normalized_h_x)
#建立I系与H系间的R I<-H
rotation_I_H=np.column_stack((normalized_h_x,final_h_y,normalized_h_z))

#求O系和H系间的R O<-H
#检查rotation_I_O是否可逆
if np.linalg.det(rotation_I_O)==0:
    print('矩阵rotation_I_O不可逆')
else:
 rotation_I_O_ni=rotation_I_O.T
 rotation_O_H=np.dot(rotation_I_H,rotation_I_O_ni)

#求解O系和H系间的平移向量
#平移向量在I下的表示
pingyi_O_H_I=np.array(hand_position_W)-np.array(drill_position_W_O)
#平移向量在O系下的表示
pingyi_O_H_O=np.dot(pingyi_O_H_I,rotation_I_O_ni)


# 更新手部位置和姿态
def move_hand_to_target(hand_pose, rotation_O_H, pingyi_O_H_O):
    # 应用平移
    hand_pose.p = gymapi.Vec3(
        hand_pose.p.x - pingyi_O_H_I[0]+0.035,
        hand_pose.p.y - pingyi_O_H_I[1],
        hand_pose.p.z - pingyi_O_H_I[2]
    )
    rotation_O_H_ni=rotation_O_H.T
    rotation_O_H_ni_I=np.dot(rotation_O_H_ni,rotation_I_H)
    #rotation_matrix_O_H_I=np.dot(rotation_matrix_I_H_final,rotation_O_H_ni)
    r_tes=np.dot(rotation_O_H_ni,rotation_matrix_I_H_final)
    #rotation_matrix_O_HH=np.dot()
    # 利用scipy将旋转矩阵转为四元数
    rotation_quat = R.from_matrix(r_tes).as_quat()
    rotation_quat = gymapi.Quat(rotation_quat[0],rotation_quat[1],rotation_quat[2],rotation_quat[3])


    # 更新手部的姿态
    hand_pose.r = rotation_quat

    return hand_pose
    # # 利用scipy将旋转矩阵转为四元数
    # rotation_quat_scipy = R.from_matrix(rotation_O_H).as_quat()
    # rotation_quat = gymapi.Quat(w=rotation_quat_scipy[3], x=rotation_quat_scipy[0], y=rotation_quat_scipy[1],z=rotation_quat_scipy[2])
    #
    # # 将gymapi.Quat转换为numpy数组以便于计算
    # def quat_to_array(q):
    #     return np.array([q.x, q.y, q.z, q.w])
    #
    # def array_to_quat(arr):
    #     return gymapi.Quat(w=arr[3], x=arr[0], y=arr[1], z=arr[2])
    #
    # # 将初始四元数final转换为numpy数组
    # initial_quat_arr = quat_to_array(hand_pose.r)
    #
    # # 将新的旋转四元数转换为numpy数组
    # new_quat_arr = quat_to_array(rotation_quat)
    #
    # # 四元数乘法函数
    # def quat_multiply(q1, q2):
    #     w1, x1, y1, z1 = q1
    #     w2, x2, y2, z2 = q2
    #     w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
    #     x = w1 * x2 + x1 * w2 + (y1 * z2 - z1 * y2)
    #     y = w1 * y2 + y1 * w2 + (z1 * x2 - x1 * z2)
    #     z = w1 * z2 + z1 * w2 + (x1 * y2 - y1 * x2)
    #     return np.array([x, y, z, w])
    #
    # # 计算新的四元数（初始四元数与新四元数相乘）
    # combined_quat_arr = quat_multiply(initial_quat_arr, new_quat_arr)
    #
    # # 归一化四元数
    # norm = np.linalg.norm(combined_quat_arr)
    # combined_quat_arr /= norm
    #
    # # 将结果转换回gymapi.Quat
    # combined_quat = array_to_quat(combined_quat_arr)
    #
    # # 更新手部的姿态
    # hand_pose.r = combined_quat

    #return hand_pose

# 绘制坐标系
def draw_coordinate_frame(gym, viewer, env, position, rotation_matrix, color=(1, 1, 1)):
    length = 0.1  # 线段长度
    vertices = []
    colors = []

    for i in range(3):
        axis = rotation_matrix[:, i] * length
        end_point = position + axis

        # 添加起始点和结束点
        vertices.extend([position[0], position[1], position[2], end_point[0], end_point[1], end_point[2]])
        # 设置颜色
        line_color = [0, 0, 0]
        line_color[i] = color[i]  # 设置对应轴的颜色
        colors.extend(line_color * 2)  # 每条线需要两个颜色值（起点和终点）

    # 将列表转换为 NumPy 数组
    vertices = np.array(vertices, dtype=np.float32).reshape(-1, 6)
    colors = np.array(colors, dtype=np.float32).reshape(-1, 3)

    # 使用 add_lines 方法绘制线段
    gym.add_lines(viewer, env, len(vertices), vertices, colors)

def slerp(q1, q2, t):
    """球面线性插值"""
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    cos_half_theta = np.dot(q1, q2)

    if cos_half_theta < 0:
        q2 = -q2
        cos_half_theta = -cos_half_theta

    if cos_half_theta >= 1.0:
        return q1

    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)

    if abs(sin_half_theta) < 1e-6:
        return (q1 + q2) * 0.5

    ratio_a = np.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = np.sin(t * half_theta) / sin_half_theta

    return (q1 * ratio_a + q2 * ratio_b).tolist()



# 动画状态机
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

anim_state = ANIM_SEEK_LOWER
#大拇指侧摆
current_dof = 8
#大拇指伸出
damuzhi_shenchu=9
damuzhi_chuojin=10
zhijian=11
shizhi_guanjie1=0
shizhi_guanjie2=1
zhongzhi_guanjie1=2
zhongzhi_guanjie2=3
wumingzhi_guanjie1=4
wumingzhi_guanjie2=5
xiaomuzhi_guanjie1=6
xiaomuzhi_guanjie2=7


lower_limits[damuzhi_shenchu]-=0.17
lower_limits[current_dof]-=0
lower_limits[damuzhi_chuojin]-=0
lower_limits[zhijian]+=0
lower_limits[shizhi_guanjie1]-=0.8
lower_limits[shizhi_guanjie2]-=0.8
lower_limits[zhongzhi_guanjie1]-=0.8
lower_limits[wumingzhi_guanjie1]-=0.8
lower_limits[xiaomuzhi_guanjie1]-=0.8
lower_limits[zhongzhi_guanjie2]-=0.8
lower_limits[wumingzhi_guanjie2]-=0.8
lower_limits[xiaomuzhi_guanjie2]-=0.8

i=0
# 获取初始手部姿态
initial_hand_pose = gym.get_actor_rigid_body_states(env, hand_handle, gymapi.STATE_POS)['pose'][0]
initial_position = np.array([initial_hand_pose['p']['x'], initial_hand_pose['p']['y'], initial_hand_pose['p']['z']])
initial_rotation = R.from_quat([initial_hand_pose['r']['x'], initial_hand_pose['r']['y'], initial_hand_pose['r']['z'], initial_hand_pose['r']['w']])

# 计算目标姿态
target_hand_pose = move_hand_to_target(hand_pose, rotation_O_H, pingyi_O_H_O)
target_position = np.array([target_hand_pose.p.x, target_hand_pose.p.y, target_hand_pose.p.z])
target_rotation = R.from_quat([target_hand_pose.r.x, target_hand_pose.r.y, target_hand_pose.r.z, target_hand_pose.r.w])

intermediate_point_x=0.8574708
intermediate_point_y=1.03
intermediate_point_z=0.148183
intermediate_point=(intermediate_point_x,intermediate_point_y,intermediate_point_z)
intermediate_point1_x=0.8
intermediate_point1_y=1.1
intermediate_point1_z=0.2
intermediate_point1=(intermediate_point1_x,intermediate_point1_y,intermediate_point1_z)

# add_marker(env,intermediate_point1)
# add_marker(env,tuple(initial_position))
# add_marker(env,intermediate_point)
# add_marker(env,tuple(target_position))
# 定义控制点（例如，起点、中间点、终点）
control_points = np.array([
    [initial_position[0], initial_position[1], initial_position[2]],  # 初始位置
    # [intermediate_point1_x,intermediate_point1_y,intermediate_point1_z],
    # [intermediate_point_x, intermediate_point_y, intermediate_point_z],  # 中间点

    [target_position[0], target_position[1], target_position[2]]  # 目标位置
])

# 创建Cubic Spline插值器
cs = CubicSpline(np.arange(len(control_points)), control_points, bc_type='natural')

# 动画参数
total_steps = 500  # 总过渡帧数
current_step = 0
# 主循环
while not gym.query_viewer_has_closed(viewer):
    # if i<1:
    #     # 更新手部位置和姿态
    #     hand_pose = move_hand_to_target(hand_pose, rotation_O_H, pingyi_O_H_O)
    #
    #     # # 将更新后的变换应用到手部
    #     # gym.set_actor_root_state_tensor(sim, gym.get_sim_index(sim), gym.pack_tensors([hand_pose]))
    #     # 将更新后的变换应用到手部
    #     # 注意：这里需要根据你的具体需求调整如何设置actor的状态
    #     # 例如，可以使用gym.set_actor_root_state_tensor或类似的方法
    #     gym.set_rigid_transform(env, hand_handle, hand_pose)
    #     i=i+1

    # 手部坐标系
    hand_position = np.array(hand_position_W)
    hand_rotation_matrix = rotation_I_H
    draw_coordinate_frame(gym, viewer, env, hand_position, hand_rotation_matrix,
                          color=(1, 0, 0, 0, 1, 0, 0, 0, 1))  # 绿色

    # 电钻坐标系
    drill_position = np.array(drill_position_W_O)
    drill_rotation_matrix = rotation_I_O
    draw_coordinate_frame(gym, viewer, env, drill_position, drill_rotation_matrix,
                          color=(1, 0, 0, 0, 1, 0, 0, 0, 1))  # 红色

    # 获取手部的所有关节信息
    hand_joints = gym.get_actor_dof_properties(env, hand_handle)

    # 步进物理模拟
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if current_step < total_steps:
        # 计算插值比例（0~1）
        alpha = current_step / total_steps

        # 获取当前位置
        current_position = cs(alpha * (len(control_points) - 1))

        # 旋转球面线性插值
        current_rotation_quat = slerp(initial_rotation.as_quat(), target_rotation.as_quat(), alpha)
        current_rotation = R.from_quat(current_rotation_quat)
        # 更新手部姿态
        new_pose = gymapi.Transform()
        new_pose.p = gymapi.Vec3(*current_position)
        new_pose.r = gymapi.Quat(*current_rotation.as_quat())

        # 应用新姿态
        gym.set_rigid_transform(env, hand_handle, new_pose)

        current_step += 1

    # 更新图形
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)




    current_position1 = new_pose.p
    current_position1 = np.array([current_position1.x, current_position1.y, current_position1.z])
    target = np.array([target_position[0], target_position[1], target_position[2]])
    # 设置执行抓取的位置
    wucha = 1e-4

    if np.allclose(current_position1, np.array([0.94555557,0.98007452,0.14838664]), atol=wucha):
        speed = speeds[current_dof]
        speed2 = speeds2[current_dof]
        # animate the dofs
        if anim_state == ANIM_SEEK_LOWER:
            if dof_positions[current_dof] > lower_limits[current_dof]:
                dof_positions[current_dof] -= speed * dt
            else:

                dof_positions[current_dof] = lower_limits[current_dof]

            # if dof_positions[zhijian] > lower_limits[zhijian]:
            #     dof_positions[zhijian] -= speed * dt
            # else:
            #
            #     dof_positions[zhijian] = lower_limits[zhijian]

            if dof_positions[shizhi_guanjie1] > lower_limits[shizhi_guanjie1]:
                dof_positions[shizhi_guanjie1] -= speed * dt
            else:

                dof_positions[shizhi_guanjie1] = lower_limits[shizhi_guanjie1]

            if dof_positions[shizhi_guanjie2] > lower_limits[shizhi_guanjie2]:
                dof_positions[shizhi_guanjie2] -= speed * dt
            else:

                dof_positions[shizhi_guanjie2] = lower_limits[shizhi_guanjie2]

            if dof_positions[shizhi_guanjie2] > lower_limits[shizhi_guanjie2]:
                dof_positions[shizhi_guanjie2] -= speed * dt
            else:

                dof_positions[shizhi_guanjie2] = lower_limits[shizhi_guanjie2]

            if dof_positions[zhongzhi_guanjie1] > lower_limits[zhongzhi_guanjie1]:
                dof_positions[zhongzhi_guanjie1] -= speed * dt
            else:

                dof_positions[zhongzhi_guanjie1] = lower_limits[zhongzhi_guanjie1]

            if dof_positions[zhongzhi_guanjie2] > lower_limits[zhongzhi_guanjie2]:
                dof_positions[zhongzhi_guanjie2] -= speed * dt
            else:

                dof_positions[zhongzhi_guanjie2] = lower_limits[zhongzhi_guanjie2]

            if dof_positions[wumingzhi_guanjie1] > lower_limits[wumingzhi_guanjie1]:
                dof_positions[wumingzhi_guanjie1] -= speed * dt
            else:

                dof_positions[wumingzhi_guanjie1] = lower_limits[wumingzhi_guanjie1]

            if dof_positions[wumingzhi_guanjie2] > lower_limits[wumingzhi_guanjie2]:
                dof_positions[wumingzhi_guanjie2] -= speed * dt
            else:

                dof_positions[wumingzhi_guanjie2] = lower_limits[wumingzhi_guanjie2]

            if dof_positions[xiaomuzhi_guanjie1] > lower_limits[xiaomuzhi_guanjie1]:
                dof_positions[xiaomuzhi_guanjie1] -= speed * dt
            else:

                dof_positions[xiaomuzhi_guanjie1] = lower_limits[xiaomuzhi_guanjie1]

            if dof_positions[xiaomuzhi_guanjie2] > lower_limits[xiaomuzhi_guanjie2]:
                dof_positions[xiaomuzhi_guanjie2] -= speed * dt
            else:

                dof_positions[xiaomuzhi_guanjie2] = lower_limits[xiaomuzhi_guanjie2]

            if dof_positions[damuzhi_shenchu] < lower_limits[damuzhi_shenchu]:
                dof_positions[damuzhi_shenchu] -= speed * dt
            else:

                dof_positions[damuzhi_shenchu] = lower_limits[damuzhi_shenchu]

            # if dof_positions[damuzhi_chuojin] > lower_limits[damuzhi_chuojin]:
            #     dof_positions[damuzhi_chuojin] -= speed * dt
            # else:
            #
            #     dof_positions[damuzhi_chuojin] = lower_limits[damuzhi_chuojin]
            # anim_state = ANIM_SEEK_UPPER
        elif anim_state == ANIM_SEEK_UPPER:
            dof_positions[current_dof] += speed * dt
            if dof_positions[current_dof] >= upper_limits[current_dof]:
                dof_positions[current_dof] = upper_limits[current_dof]
                anim_state = ANIM_SEEK_DEFAULT
        if anim_state == ANIM_SEEK_DEFAULT:
            dof_positions[current_dof] -= speed * dt
            if dof_positions[current_dof] <= defaults[current_dof]:
                dof_positions[current_dof] = defaults[current_dof]
                anim_state = ANIM_FINISHED
        elif anim_state == ANIM_FINISHED:
            dof_positions[current_dof] = defaults[current_dof]
            current_dof = (current_dof + 1) % num_dofs
            anim_state = ANIM_SEEK_LOWER
            print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))


    if args.show_axis:
        gym.clear_lines(viewer)

    gym.set_actor_dof_states(env, hand_handle, dof_states, gymapi.STATE_POS)
    # 同步帧时间以匹配实际时间流速
    gym.sync_frame_time(sim)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)