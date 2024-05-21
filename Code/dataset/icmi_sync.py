import os
import argparse
import cv2
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from frame_match import *
import datetime
import multiprocessing
import glob
from natsort import natsorted
from PIL import Image
import threading
# import wandb

def remove_consecutive_repeated_elements(array):
    mask = np.concatenate(([True], np.any(array[1:, 1] != array[:-1, 1], axis=1)))
    filtered_array = array[mask]
    return filtered_array

def make_video(file_path, corr_indices, cam_identifier):
    count = 0
    # Path for writing new images 
    write_folder = file_path + "_synchronized" + cam_identifier
    os.makedirs(write_folder, exist_ok=True)

    # Path for reading all unsynchronized images 
    file_path = file_path + cam_identifier
    camera = file_path.split("/")[-1]
    # output_video_file = file_path.replace(camera, "synchronized_{}.mp4".format(camera))

    image_path = file_path + "/*.png"
    
    for i, image_path in enumerate(natsorted(glob.glob(image_path))):
        
        if i in corr_indices:
            # print ("image_path ", count)
            cv2_image = cv2.imread(image_path)
            pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
            pil_image.save(f'{write_folder}/{count}.png')        
            count +=1

    return
    
def make_only_video(file_path, cam_identifier='kinect1_color'):
    ount = 0
    # Path for writing new images 
    write_folder = file_path + cam_identifier
    os.makedirs(write_folder, exist_ok=True)

    # Path for reading all unsynchronized images 
    file_path = file_path + cam_identifier
    print ("file_path ", file_path)
    camera = file_path.split("/")[-1]
    output_video_file = file_path.replace(camera, "synchronized_{}.mp4".format(camera))

    image_path = file_path + "/*.png"
    first_image_path = natsorted(glob.glob(image_path))[0]

    first_image = cv2.imread(first_image_path)
    fps = 15
    width = int(first_image.shape[1])
    height = int(first_image.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (codec options may vary)
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))


    for i, image_path in enumerate(natsorted(glob.glob(image_path))):
        # if i in corr_indices:
        # print ("image_path ", count)
        cv2_image = cv2.imread(image_path)
        # pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        # pil_image.save(f'{write_folder}/{count}.png')

        # cv2.imwrite()
        video_writer.write(cv2_image)
    
    video_writer.release()


    return

def get_fileName(_trial):
    robot_pose_dir = _trial.replace("icmi_cam_data", "icmi_cam_data/input_and_pose_data") 
    print ("robot_pose_dir ", robot_pose_dir)
    try:
        robot_pose_path = glob.glob(robot_pose_dir + "/*.csv")[0]                
    except:
        robot_pose_path = glob.glob(robot_pose_dir + "/data.pkl")[0]                
    print ("robot pose_path ", robot_pose_path)
    realsense_path = _trial + "/realsense_timestamps.pkl"
    kinect1_path = _trial + "/kinect1_timestamps.pkl"
    kinect2_path = _trial + "/kinect2_timestamps.pkl"
    
    return robot_pose_path, realsense_path, kinect1_path, kinect2_path


def write_robot_data(corr_indices, robot_pose_data, robot_pose_path, file_path):
    print (len(corr_indices))
    write_folder = file_path + "_synchronized" + "/robot_data.csv"
    
    new_robot_pose = list()
    df = pd.read_csv(robot_pose_path)
    print ("old df ",len(df))
    df = df.iloc[:len(corr_indices)]

    df = df.reset_index(drop=True)
    print ("new df ",len(df))

    # for i in range(0, len(corr_indices)):
    #     new_robot_pose.append(robot_pose_data[corr_indices[i]])
    # new_robot_pose = np.asarray(new_robot_pose)
    # new_robot_pose = pd.DataFrame(new_robot_pose)
    df.to_csv(write_folder)

def write_robot_data_from_pickle(corr_indices, robot_pose_data, robot_pose_path, file_path):
    write_folder = file_path + "_synchronized" + "/robot_data.csv"
    
    new_robot_pose = list()
    with open(robot_pose_path, 'rb') as _file:
        data = pickle.load(_file)     

    # data = np.asarray(data)    
    selected_keys = ['timestamps', 'observations']
    filtered_data = {key: data[key] for key in selected_keys}
    filtered_data['timestamps'] = list()
    filtered_data['ee_pos'] = list()
    filtered_data['ee_pos_vel'] = list()
    filtered_data['ee_ori_vel'] = list()
    filtered_data['ee_quat'] = list()
    filtered_data['gripper_width'] = list()

    min_length = min(len(v) for v in filtered_data.values())    

    for i in range(len(corr_indices)):
        filtered_data['timestamps'].append(data['timestamps'][i])
        filtered_data['ee_pos'].append([filtered_data['observations'][i]['robot_state']['ee_pos'][0], filtered_data['observations'][i]['robot_state']['ee_pos'][1], filtered_data['observations'][i]['robot_state']['ee_pos'][2]])
        filtered_data['ee_pos_vel'].append([filtered_data['observations'][i]['robot_state']['ee_pos_vel'][0], filtered_data['observations'][i]['robot_state']['ee_pos_vel'][1], filtered_data['observations'][i]['robot_state']['ee_pos_vel'][2]])
        filtered_data['ee_ori_vel'].append([filtered_data['observations'][i]['robot_state']['ee_ori_vel'][0], filtered_data['observations'][1]['robot_state']['ee_ori_vel'][1], filtered_data['observations'][i]['robot_state']['ee_ori_vel'][2]])
        filtered_data['ee_quat'].append([filtered_data['observations'][i]['robot_state']['ee_quat'][0], filtered_data['observations'][i]['robot_state']['ee_quat'][1], filtered_data['observations'][i]['robot_state']['ee_quat'][2], filtered_data['observations'][i]['robot_state']['ee_quat'][3]])
        filtered_data['gripper_width'].append([filtered_data['observations'][i]['robot_state']['gripper_width'][0]<0.05])
    del (filtered_data['observations'])
    filtered_data = pd.DataFrame.from_dict(filtered_data)
    # filtered_data = pd.DataFrame(filtered_data)

    filtered_data = filtered_data.reset_index(drop=True)
    print ("new df ",len(filtered_data))

    # for i in range(0, len(corr_indices)):
    #     new_robot_pose.append(robot_pose_data[corr_indices[i]])
    # new_robot_pose = np.asarray(new_robot_pose)
    # new_robot_pose = pd.DataFrame(new_robot_pose)
    filtered_data.to_csv(write_folder)

def read_pickle(pickle_file):
    with open(pickle_file, 'rb') as _file:
        data = pickle.load(_file)     

    return np.asarray(data).reshape(-1,1)


def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    data = np.asarray(data)    
    return (data)[1:, 0].reshape(-1,1), data

def read_robot_pickle(pickle_file):
    with open(pickle_file, 'rb') as _file:
        data = pickle.load(_file)     

    # data = np.asarray(data)    
    selected_keys = ['timestamps', 'observations']
    filtered_data = {key: data[key] for key in selected_keys}
    
    min_length = min(len(v) for v in filtered_data.values())
    filtered_data['timestamps'] = filtered_data['timestamps'][:min_length]
    filtered_data['observations'] = filtered_data['observations'][:min_length]
    print (len(filtered_data['timestamps']))
    # print ("data shape ", data.keys(), data['observations'][:10], data['actions'][:10])
    filtered_data = pd.DataFrame(filtered_data)
    return filtered_data['timestamps'].values.reshape(-1,1), filtered_data


def multi_thread_write_images(_trial, corr_indices_k1_w, corr_indices_k1_k1, corr_indices_k1_k2):
    thread1 = threading.Thread(target=make_video, args =(_trial, corr_indices_k1_w, "/rs_color"))
    thread2 = threading.Thread(target=make_video, args =(_trial, corr_indices_k1_k1, "/kinect1_color"))
    thread3 = threading.Thread(target=make_video, args =(_trial, corr_indices_k1_k2, "/kinect2_color"))
    thread4 = threading.Thread(target=make_video, args =(_trial, corr_indices_k1_w, "/rs_depth"))
    thread5 = threading.Thread(target=make_video, args =(_trial, corr_indices_k1_k1, "/kinect1_depth"))
    thread6 = threading.Thread(target=make_video, args =(_trial, corr_indices_k1_k2, "/kinect2_depth"))
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()

    # Wait for all threads to finish
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()

count = 0    
data_dir = "<path-to-dataset>"
# P_*/Task_*/Interface_*/Trial_* 

# make_only_video(cam_identifier='kinect1_color')
for _dir in sorted(glob.glob(data_dir), reverse=True):
    for i in range (1,9):
        task_dir = _dir + "/task_{}".format(i)
        for _task in sorted(glob.glob(task_dir)):
            interface_dir = _task + "/interface_3"
            for _interface in sorted(glob.glob(interface_dir)):
                trial_dir = _interface + "/episode_*"
                for _trial in sorted(glob.glob(trial_dir)):
                    # Assigning path variables
                    if "synchronized" in _trial:
                        continue
                    
                    print ("trial_dir ", _trial)
                    robot_pose_path, realsense_path, kinect1_path, kinect2_path = get_fileName(_trial)
                    
                    if os.path.exists(realsense_path) and os.path.exists(kinect1_path) and os.path.exists(kinect2_path) and os.path.exists(robot_pose_path):
                        print ("reading pickle files")
                        realsense_timestamps = read_pickle(realsense_path)
                        kinect1_timestamps = read_pickle(kinect1_path)
                        kinect2_timestamps = read_pickle(kinect2_path)
                        if _interface.split("/")[-1] == "interface_3":
                            robot_timestamps, robot_pose_data = read_robot_pickle(robot_pose_path)
                            delta_rs = robot_timestamps[0] - realsense_timestamps[0]
                            robot_timestamps = robot_timestamps - delta_rs

                        else:
                            robot_timestamps, robot_pose_data = read_csv(robot_pose_path)

                        query_timestamp = robot_timestamps
                        
                        if len(robot_timestamps)>1:
                            query_timestamp, corr_indices_k1_k1 = loop_queryarray(query_timestamp, kinect1_timestamps, denom=1, interface_3=_interface.split("/")[-1] == "interface_3")
                            query_timestamp, corr_indices_k1_k2 = loop_queryarray(query_timestamp, kinect2_timestamps, denom=1, interface_3=_interface.split("/")[-1] == "interface_3")
                            _, corr_indices_k1_w = loop_queryarray(query_timestamp, realsense_timestamps, denom=1, interface_3=_interface.split("/")[-1] == "interface_3")
                            _, corr_indices_k1_r = loop_queryarray(query_timestamp, robot_timestamps, denom=1, interface_3=_interface.split("/")[-1] == "interface_3")
                            # print (corr_indices_k1_k2[:10], corr_indices_k1_r[:10], corr_indices_k1_w[:10])
                            # multi_thread_write_images(_trial, corr_indices_k1_w, corr_indices_k1_k1, corr_indices_k1_k2)
                            if _interface.split("/")[-1] == "interface_3":
                                write_robot_data_from_pickle(list(set(corr_indices_k1_k1)), robot_pose_data, robot_pose_path, _trial)                    
                            else:
                                write_robot_data(list(set(corr_indices_k1_r)), robot_pose_data, robot_pose_path, _trial)                    

                            print ("timestamps for all the streans ", len(set(corr_indices_k1_w)),len(set(corr_indices_k1_k2)),len(set(corr_indices_k1_r)), len(set(corr_indices_k1_k1)))
                            # print ("writing ", corr_indices_k1_k2[:])
                            
                        
                            
                            

                    else:
                        print ("missing pickle files")
                
