from dataset.yarr.utils.observation_type import ObservationElement
from dataset.yarr.replay_buffer.replay_buffer import ReplayBuffer, ReplayElement
from dataset.utils import keypoint_discovery, read_robot_pose, read_robot_pose_from_vr, _add_keypoints_to_replay
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
import os
import numpy as np
from typing import List
import logging

def create_replay(batch_size: int,
                  timesteps: int,
                  disk_saving: bool, 
                  cameras: list,
                  voxel_sizes,
                  replay_size=1e3):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512
    IMAGE_SIZE1 = 128
    IMAGE_SIZE2 = 128

    # low_dim_state
    observation_elements = []    
    observation_elements.append(
        ObservationElement('prev_gripper_pose', (gripper_pose_size,), np.float32))
        
    
    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:        
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, IMAGE_SIZE1, IMAGE_SIZE2,), np.float32))
        observation_elements.append(
            ObservationElement('%s_depth' % cname, (3, IMAGE_SIZE1, IMAGE_SIZE1,), np.float32))
        # observation_elements.append(
        #     ObservationElement('%s_point_cloud' % cname, (3, IMAGE_SIZE, IMAGE_SIZE,), np.float32)) # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        # observation_elements.append(
        #     ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        # observation_elements.append(
        #     ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,),
                      np.float32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                      np.float32),
        ReplayElement('ignore_collisions', (ignore_collisions_size,),
                      np.int32),
        ReplayElement('gripper_pose', (gripper_pose_size,),
                      np.float32),        
        ReplayElement('gripper_rot', (trans_indicies_size,),
                      np.float32),                              
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), bool),
    ]
    print ("batch size: ", batch_size)
    replay_buffer = UniformReplayBuffer( # all tuples in the buffer have equal sample weighting
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(7,), # 3 translation + 3 rotation + 1 gripper open
        action_dtype=np.float32,
        reward_shape=(),
        disk_saving=disk_saving,
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer



def get_stored_demo(task_rgbd_file, robot_pose_file, interface_id=0):

    robot_pose_file = f"{robot_pose_file}/robot_data.csv"

    if interface_id == 3:
        episode_keypoints, demo = read_robot_pose_from_vr(task_rgbd_file, robot_pose_file, interface_id)    
    else:
        episode_keypoints, demo = read_robot_pose(task_rgbd_file, robot_pose_file, interface_id)    
    
    return demo, episode_keypoints

        

def fill_replay(
    replay: ReplayBuffer,
    task: str,
    interface: str,
    task_replay_storage_folder: str,
    start_idx: int,
    end_idx: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    device="cuda",
):

    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...")
        for d_idx in range(start_idx, end_idx):
            print("Filling demo %d" % d_idx)
            task_rgbd_file = os.path.join(data_path, episode_folder%(d_idx))

            robot_pose_file = os.path.join(data_path, episode_folder%(d_idx))
            if os.path.exists(os.path.join(robot_pose_file, 'robot_data.csv')):
                demo, episode_keypoints = get_stored_demo(task_rgbd_file, robot_pose_file, interface_id=interface)
            else:
                continue
            
            next_keypoint_idx = 0
            for i in range(len(demo) - 1):
                # demo[i].description = desc + " " + demo[i].description  
                obs = demo[i]            
                # if our starting point is past one of the keypoints, then remove it
                initial_pose = demo[i].gripper_pose
                # np.concatenate([demo[i].gripper_pose, demo[i].gripper_rot, demo[i].gripper_open]) # for the diffusion model
                
                # if our starting point is past one of the keypoints, then remove it
                while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                    episode_keypoints = episode_keypoints[1:]
                if len(episode_keypoints) == 0:
                    break
                

                _add_keypoints_to_replay(
                    replay, 
                    task, 
                    task_replay_storage_folder,
                    obs, 
                    initial_pose, 
                    demo, 
                    episode_keypoints, 
                    cameras,
                    voxel_sizes,
                    rotation_resolution, 
                    crop_augmentation=False, 
                    device='cuda'
                    )

        # save TERMINAL info in replay_info.npy

        task_idx = replay._task_index[task]
        with open(
            os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        ) as fp:
            np.save(
                fp,
                replay._store["terminal"][
                    replay._task_replay_start_index[
                        task_idx
                    ] : replay._task_replay_start_index[task_idx]
                    + replay._task_add_count[task_idx].value
                ],
            )

        print("Replay filled with demos.")
