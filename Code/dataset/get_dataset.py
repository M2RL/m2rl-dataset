import os 
import numpy as numpy
import pandas as pd
import numpy as np
from dataset.demo import Demo
from dataset.observation import Observation
import glob
from config import config
from typing import List
from dataset.utils import keypoint_discovery, read_robot_pose, read_robot_pose_from_vr, _add_keypoints_to_replay
from dataset.replay_buffer import create_replay, fill_replay
from dataset.yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer

# cameras = ['kinect_left']

def get_low_obs(obs_file):
    obs = pd.read_csv(obs_file)
    obs = np.asarray(obs)


def get_train_dataset(tasks: List, interfaces: List, start_index: int, end_index: int):

    train_replay_buffer = create_replay(batch_size=config.BATCH_SIZE,
                        timesteps=1,
                        disk_saving=True,
                        cameras=config.CAMERAS,
                        voxel_sizes=[100])
    for task in tasks:
        for interface in interfaces:
            EPISODES_FOLDER_TRAIN = f"task_{task}/interface_{interface}"
            data_path_train = os.path.join(config.data_dir, EPISODES_FOLDER_TRAIN)
            train_replay_storage_folder = f"{config.TRAIN_REPLAY_STORAGE_DIR}/{task}/{interface}"
            print (f"start index {start_index} end index {end_index}")

            fill_replay(
                    replay=train_replay_buffer,
                    task=task,
                    interface=interface, 
                    task_replay_storage_folder=train_replay_storage_folder,
                    start_idx=start_index,
                    end_idx=end_index,
                    cameras=config.CAMERAS,
                    rlbench_scene_bounds=config.SCENE_BOUNDS,
                    voxel_sizes=config.VOXEL_SIZES,
                    rotation_resolution=config.rotation_resolution,
                    crop_augmentation=False,
                    data_path=data_path_train,
                    episode_folder=config.EPISODE_FOLDER,
                )
    return train_replay_buffer


def get_test_dataset(tasks: List, interfaces: List, start_index: int, end_index: int):
    test_replay_buffer = create_replay(batch_size=config.BATCH_SIZE,
                        timesteps=1,
                        disk_saving=True,
                        cameras=config.CAMERAS,
                        voxel_sizes=[100])
    for task in tasks:
        for interface in interfaces:
            EPISODES_FOLDER_TEST = f"task_{task}/interface_{interface}"
            data_path_test = os.path.join(config.data_dir, EPISODES_FOLDER_TEST)
            test_replay_storage_folder = f"{config.TEST_REPLAY_STORAGE_DIR}/{task}/{interface}"
            print (f"start index {start_index} end index {end_index}")
            
            fill_replay(
                    replay=test_replay_buffer,
                    task=task,
                    interface=interface, 
                    task_replay_storage_folder=test_replay_storage_folder,
                    start_idx=start_index,
                    end_idx=end_index,
                    cameras=config.CAMERAS,
                    rlbench_scene_bounds=config.SCENE_BOUNDS,
                    voxel_sizes=config.VOXEL_SIZES,
                    rotation_resolution=config.rotation_resolution,
                    crop_augmentation=False,
                    data_path=data_path_test,
                    episode_folder=config.EPISODE_FOLDER,
                )
    return test_replay_buffer


def get_demo(replay, robot_obs_dir, robot_data_file, interface_id="spacemouse"):
    if interface_id == "vr":
        episode_keypoints, demo = read_robot_pose_from_vr(robot_obs_dir, robot_data_file, interface_id)    
    else:
        episode_keypoints, demo = read_robot_pose(robot_obs_dir, robot_data_file, interface_id)    
    # fill_buffer(demo)
    
    voxel_sizes = [100]
    episode_keypoints = episode_keypoints[0]
    rotation_resolution = 5
    for i in range(len(demo) - 1):
        
        obs = demo[i]
        initial_pose = demo[i].gripper_pose
        # np.concatenate([demo[i].gripper_pose, demo[i].gripper_rot, demo[i].gripper_open]) # for the diffusion model
        
        # if our starting point is past one of the keypoints, then remove it
        while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
            episode_keypoints = episode_keypoints[1:]
        if len(episode_keypoints) == 0:
            break

        _add_keypoints_to_replay(
            replay, task, task_replay_storage_folder, obs, initial_pose, demo, episode_keypoints, config.CAMERAS,
            config.VOXEL_SIZES,
            config.rotation_resolution, crop_augmentation=False, device='cuda')

"""train_replay_buffer = create_replay(batch_size=4,
                                    timesteps=1,
                                    disk_saving=True,
                                    cameras=cameras,
                                    voxel_sizes=[100])


test_replay_buffer = create_replay(batch_size=4,
                                    timesteps=1,
                                    disk_saving=True,
                                    cameras=cameras,
                                    voxel_sizes=[100])                
"""