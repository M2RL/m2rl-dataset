import logging
from typing import List
import os
import numpy as np
import pandas as pd
import csv
import ast
import re
from dataset.observation import Observation
from dataset.demo import Demo
from PIL import Image
from dataset.yarr.replay_buffer.replay_buffer import ReplayBuffer
from dataset.arm.utils import stack_on_channel, quaternion_to_discrete_euler, normalize_quaternion, discretize_euler, euler_from_quaternion


REMOVE_KEYS = {'controller_axis', 'controller_button', 'controller_hat', 'joint_velocities', 'gripper_open'}
CAMERA_WRIST = 'rs_color'
CAMERA_LS = 'kinect1_color'
CAMERA_RS = 'kinect2_color'

CAMERA_WRIST_depth = 'rs_depth'
CAMERA_LS_depth = 'kinect1_depth'
CAMERA_RS_depth = 'kinect2_depth'

IMAGE_FORMAT  = '%d.png'
TARGET_RES = (128, 128)
def _is_stopped(demo, i, obs, stopped_buffer, delta=0.01):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)    
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def read_rgbd(episode_path, demo):    

    for i in range(len(demo)):
        demo[i].left_shoulder_rgb = np.zeros((3, 128, 128))
        demo[i].right_shoulder_rgb = np.zeros((3, 128, 128))
        demo[i].wrist_rgb = np.zeros((3, 128, 128))
        demo[i].left_shoulder_depth = np.zeros((3, 128, 128))
        demo[i].right_shoulder_depth = np.zeros((3, 128, 128))
        demo[i].wrist_depth = np.zeros((3, 128, 128))

        if os.path.exists(os.path.join(episode_path, '%s' % (CAMERA_LS), IMAGE_FORMAT % i)):
            demo[i].left_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s' % (CAMERA_LS), IMAGE_FORMAT % i)).resize(TARGET_RES)).transpose(2,0,1)
        else:
            print ("path does not exists!!")

        if os.path.exists(os.path.join(episode_path, '%s' % (CAMERA_RS), IMAGE_FORMAT % i)):
            demo[i].right_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s' % (CAMERA_RS), IMAGE_FORMAT % i)).resize(TARGET_RES)).transpose(2,0,1)
        if os.path.exists(os.path.join(episode_path, '%s' % (CAMERA_WRIST), IMAGE_FORMAT % i)):            
            demo[i].wrist_rgb = np.array(Image.open(os.path.join(episode_path, '%s' % (CAMERA_WRIST), IMAGE_FORMAT % i)).resize(TARGET_RES)).transpose(2,0,1)

        # obs[i].front_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        if os.path.exists(os.path.join(episode_path, '%s' % (CAMERA_LS_depth), IMAGE_FORMAT % i)):
            demo[i].left_shoulder_depth = np.array(Image.open(os.path.join(episode_path, '%s' % (CAMERA_LS_depth), IMAGE_FORMAT % i)).resize(TARGET_RES)).transpose(2,0,1)
        if os.path.exists(os.path.join(episode_path, '%s' % (CAMERA_RS_depth), IMAGE_FORMAT % i)):
            demo[i].right_shoulder_depth = np.array(Image.open(os.path.join(episode_path, '%s' % (CAMERA_RS_depth), IMAGE_FORMAT % i)).resize(TARGET_RES)).transpose(2,0,1)
        if os.path.exists(os.path.join(episode_path, '%s' % (CAMERA_WRIST_depth), IMAGE_FORMAT % i)):            
            demo[i].wrist_depth = np.array(Image.open(os.path.join(episode_path, '%s' % (CAMERA_WRIST_depth), IMAGE_FORMAT % i)).resize(TARGET_RES)).transpose(2,0,1)

    return 

def read_robot_pose(task_rgbd_file, robot_pose_file, interface_id=1):
    obs_eef = list()
    obs_rot = list()
    obs_gripper = list()
    controller_axis = list()
    controller_button = list()
    controller_hat = list()
    demo = list()
    joint_velocities = list()
    obs = pd.read_csv(robot_pose_file)

    obs['controller_data'] = obs['controller_data'].apply(eval)
    obs['eef_pos'] = obs['eef_pos'].apply(eval)

    joint_velocities.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for i in range(obs['eef_pos'].shape[0]):       
        x, y, z = obs['eef_pos'][i]['x'], obs['eef_pos'][i]['y'], obs['eef_pos'][i]['z']        
        roll, pitch, yaw = obs['eef_pos'][i]['roll'], obs['eef_pos'][i]['pitch'], obs['eef_pos'][i]['yaw']
        gripper_state = obs['eef_pos'][i]['grip_status']
        
        obs_eef.append([x, y, z])
        obs_rot.append([roll, pitch, yaw])
        if i>0:
            joint_velocities.append([obs_eef[i][0]-obs_eef[i-1][0], obs_eef[i][1]-obs_eef[i-1][1], obs_eef[i][2]-obs_eef[i-1][2], 
                                    obs_rot[i][0]-obs_rot[i-1][0], obs_rot[i][1]-obs_rot[i-1][1], obs_rot[i][2]-obs_rot[i-1][2]])
        
        obs_gripper.append(gripper_state)

        # print (obs['controller_data'][i].keys())

        if interface_id == 1:
            # print ("obs['controller_data'][i] inside spacemouse", obs['controller_data'][i].keys())
            x, y, z = obs['controller_data'][i]['x'], obs['controller_data'][i]['y'], obs['eef_pos'][i]['z']
            roll, pitch, yaw = obs['controller_data'][i]['roll'], obs['controller_data'][i]['pitch'], obs['controller_data'][i]['yaw']
            button_0, button_1 = obs['controller_data'][i]['button_0'], obs['controller_data'][i]['button_1']

            controller_button.append([x, y, z])
            controller_axis.append([roll, pitch, yaw])
            controller_hat.append([button_0, button_1])
        elif interface_id == 2:
            # print ("obs['controller_data'][i] inside joystick", obs['controller_data'][i].keys())
            axis = obs['controller_data'][i]['axis'] 
            button = obs['controller_data'][i]['button']
            hat = obs['controller_data'][i]['hat']

            controller_axis.append(axis)
            controller_button.append(button)
            controller_hat.append(hat)

        elif interface_id == 3:
            axis = obs['controller_data'][i]['axis'] 
            button = obs['controller_data'][i]['button']
            hat = obs['controller_data'][i]['hat']

            controller_axis.append(axis)
            controller_button.append(button)
            controller_hat.append(hat)
        

    
    obs_eef = np.asarray(obs_eef).reshape(-1, 3)
    obs_rot = np.asarray(obs_rot).reshape(-1, 3)
    obs_gripper = np.asarray(obs_gripper).reshape(-1, 1)

    if interface_id == 1:
        controller_axis = np.asarray(controller_axis).reshape(-1, 3)
        controller_button = np.asarray(controller_button).reshape(-1, 3)
        controller_hat = np.asarray(controller_hat).reshape(-1, 2)        
    else:
        controller_axis = np.asarray(controller_axis).reshape(-1, 6)
        controller_button = np.asarray(controller_button).reshape(-1, 12)
        controller_hat = np.asarray(controller_hat).reshape(-1, 2)
    for i in range (len(controller_hat)):
        obs = Observation(gripper_pose=obs_eef[i],
                    gripper_rot=obs_rot[i],
                    gripper_open=obs_gripper[i],
                    controller_axis=controller_axis[i],
                    controller_button=controller_button[i],
                    controller_hat=controller_hat[i],
                    joint_velocities=joint_velocities[i])
        demo.append(obs)

        # demo = Demo(observations=obs)
    read_rgbd(task_rgbd_file, demo)
    keypoints, _ = keypoint_discovery(demo)

    return keypoints, demo


def read_robot_pose_from_vr(task_rgbd_file, robot_pose_file, interface_id="spacemouse"):
    obs_eef = list()
    obs_rot = list()
    ee_ori_vel = list()
    ee_pos_vel = list()
    obs_gripper = list()
    controller_axis = list()
    controller_button = list()
    controller_hat = list()
    demo = list()
    joint_velocities = list()
                
    obs = pd.read_csv(robot_pose_file)

    def parse_list(list_str):
        list_elements = ast.literal_eval(list_str)
        return np.array(list_elements)

    # Parse each column of the DataFrame
    parsed_df = obs.copy()
    # for column in ['ee_pos', 'ee_pos_vel', 'ee_ori_vel', 'ee_quat']:
    #     parsed_df[column] = parsed_df[column].apply(parse_list)

    # For the 'gripper_width' column, convert the single value to a NumPy array

    for i in range(len(parsed_df['ee_pos'])):
        cart = ast.literal_eval(parsed_df['ee_pos'][i])
        # ori = quaternion_to_discrete_euler(ast.literal_eval(parsed_df['ee_quat'][i]), resolution=5)
        ori = euler_from_quaternion(ast.literal_eval(parsed_df['ee_quat'][i]))
        gripper_width = ast.literal_eval(parsed_df['gripper_width'][i])
        pos_vel = ast.literal_eval(parsed_df['ee_pos_vel'][i])
        ori_vel = ast.literal_eval(parsed_df['ee_ori_vel'][i])

        obs_eef.append([cart[0], cart[1], cart[2]])
        obs_rot.append([ori[0], ori[1], ori[2]])
        ee_pos_vel.append([pos_vel[0], pos_vel[1], pos_vel[2]])
        ee_ori_vel.append([ori_vel[0], ori_vel[1], ori_vel[2]])
        obs_gripper.append(gripper_width)
    
    obs_eef = np.array(obs_eef).reshape(-1, 3)
    ee_pos_vel = np.array(ee_pos_vel).reshape(-1, 3)
    ee_ori_vel = np.array(ee_ori_vel).reshape(-1, 3)
    print ("obs_eef ", obs_eef.shape)

    joint_velocities = np.concatenate((ee_pos_vel, ee_ori_vel), axis=-1)
    obs_gripper = np.asarray(obs_gripper).reshape(-1, 1)

    for i in range (len(obs_eef)):
        obs = Observation(gripper_pose=obs_eef[i],
                    gripper_rot=obs_rot[i],
                    gripper_open=obs_gripper[i],
                    controller_axis=obs_eef[i],
                    controller_button=obs_eef[i],
                    controller_hat=obs_eef[i],
                    joint_velocities=joint_velocities[i])
        demo.append(obs)

        # demo = Demo(observations=obs)
    read_rgbd(task_rgbd_file, demo)
    keypoints, _ = keypoint_discovery(demo)
    return keypoints, demo

def keypoint_discovery(demo: Demo,
                       stopping_delta=0.01,
                       method='heuristic') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        prev_keypoint = 0
        
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 50 if stopped else stopped_buffer - 1
            last = i == (len(demo) - 1)
            # If change in gripper, or end of episode.
            if not (prev_keypoint>0 and stopped and obs.gripper_open):
                if i != 0 and (obs.gripper_open != prev_gripper_open or
                            last or stopped):
                    episode_keypoints.append(i)
                    prev_keypoint = i

                prev_gripper_open = obs.gripper_open
        logging.debug('Found original %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        

        segment_length = len(demo) // 10
        episode_sampled_jeypoints = list()
        for i in range(0, len(demo), segment_length):
            episode_sampled_jeypoints.append(i)
            if i not in episode_keypoints:
                episode_keypoints.append(i)
        
        if len(episode_sampled_jeypoints) > 1 and (episode_sampled_jeypoints[-1] - 1) == \
                episode_sampled_jeypoints[-2]:
            episode_sampled_jeypoints.pop(-2)
        logging.debug('Found sampled %d keypoints.' % len(episode_sampled_jeypoints),
                      episode_sampled_jeypoints)
        print ('Found new original %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return sorted(episode_keypoints), episode_sampled_jeypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


# find minimum difference between any two elements in list
def find_minimum_difference(lst):
    minimum = lst[-1]
    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] < minimum:
            minimum = lst[i] - lst[i - 1]
    return minimum


def extract_obs(obs: Observation,
				cameras,
                t: int = 0,
                prev_action = None,
                channels_last: bool = False):

    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in REMOVE_KEYS}
    # remove low-level proprioception variables that are not needed
    # if not channels_last:
    #     # swap channels from last dim to 1st dim
    #     obs_dict = {k: np.transpose(
    #         v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
    #                 for k, v in obs_dict.items()}
    # else:
    #     # add extra dim to depth data
    #     obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
    #                 for k, v in obs_dict.items()}


    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    obs_dict['ignore_collisions'] = np.array(obs.ignore_collisions, dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)
    
    
    # np.concatenate(obs.gripper_pose, obs.gripper_rot, obs.gripper_open) # for the diffusion model

    return obs_dict

def _get_action(
        obs_tp1: Observation,
        obs_tm1: Observation,        
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool):
    #TO DO: How to discretize negative rotation values
    rot = obs_tp1.gripper_rot[:3]
    # if rot[-1] < 0:
    #     rot *= -1
    # rot[rot[0]<=0] = 1e-10
    # rot[rot[1]<=0] = 1e-10
    # rot[rot[2]<=0] = 1e-10

    disc_rot = rot #discretize_euler(rot, rotation_resolution)
    trans_indicies, attention_coordinates = [], []    
    ignore_collisions = int(obs_tm1.ignore_collisions)    

    for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
        trans_indicies.extend(obs_tp1.gripper_pose[:3])
    if not isinstance(disc_rot, list):
        rot_and_grip_indicies = disc_rot.tolist()
    else:
        rot_and_grip_indicies = disc_rot

    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
        [trans_indicies, obs_tp1.gripper_rot, np.array([grip])]), attention_coordinates

# add individual data points to replay
def _add_keypoints_to_replay(        
        replay: ReplayBuffer,
        task: str,
        task_replay_storage_folder: str,
        inital_obs: Observation,
        initial_pose: List[float],
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        voxel_sizes: List[int],
        rotation_resolution: int,
        crop_augmentation: bool,
        description: str = '',
        clip_model = None,
        device = 'cpu'):
    prev_action = None
    obs = inital_obs
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        
        obs_tm1 = demo[max(0, keypoint - 1)]
        trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
            obs_tp1, obs_tm1, voxel_sizes,
            rotation_resolution, crop_augmentation)
        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(obs, cameras, t=k, prev_action=prev_action)
        obs_dict['prev_gripper_pose'] = initial_pose
        prev_action = np.copy(action)
        others = {'demo': True}
        final_obs = {
            'prev_gripper_pose': initial_pose,
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
            'gripper_pose': obs_tp1.gripper_pose,
        }
        initial_pose = demo[keypoint].gripper_pose

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(task, task_replay_storage_folder, action, reward, terminal, timeout, **others)
        obs = obs_tp1

    # final step
    obs_dict_tp1 = extract_obs(obs_tp1, cameras, t=k + 1, prev_action=prev_action)

    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)