# Multi-Keypoint Affordance Representation for Functional Dexterous Grasping (MKA)
https://github.com/user-attachments/assets/cc6d6549-6507-46ea-adb4-8d949762594d
## Usage
### Affordance (CMKA)
#### 1. Requirements

Code is tested under Pytorch 1.12.1, python 3.7, and CUDA 11.6

```
pip install -r funcgra/requirements.txt
```
You can download the sam_vit modles [Baidu Pan)](https://pan.baidu.com/s/16zwHbANn3mEhtdji2vZE9Q?pwd=ca8a). The extraction code is: `ca8a`.
#### 2. Dataset

You can download the FAH from [Baidu Pan (3.35G)](https://pan.baidu.com/s/1Naf9GirbG1mB9cSSxc6N_w?pwd=a6nv). The extraction code is: `a6nv`.

#### 3. Train

Run following commands to start training:

```
python train.py --data_root <PATH_TO_DATA>
```
####4.
You can download the pretrained modles [Baidu Pan)](https://pan.baidu.com/s/1vdKDS4yvaody056UUy1VAA?pwd=8qvk). The extraction code is: `8qvk`.

### Simulation (KGT)
#### 1. Requirements
The simulation code was tested on Pytorch 1.13.1, Python 3.7, and CUDA 9.1

```
pip install -r requirements_sim.txt
```
#### 2. Run

```
cd isaacgym/python
python hand_move_to_drill_hold_no_pengzhuang.py
python hand_move_to_drill_press_test.py
python hand_move_to_flashlight_click
python hand_move_to_flashlight_hold.py
```
### Experiments
#### 1. Requirements
The experiments code was tested on Pytorch 1.12.0, Python 3.9, and CUDA 11.3

```
pip install -r requirements_exp.txt
```

#### 2. Run
```
cd fungra/
python robot_control_click_the_flashlight.py
python robot_control_press_the_spraybottle.py
python robot_control_press_the_drill.py
python robot_control_hold_the_kettle.py
```
