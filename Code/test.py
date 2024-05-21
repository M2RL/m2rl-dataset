import os 
import glob
import time
import torch
from typing import List
from dataset.get_dataset import *
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from config import config
import wandb
from agents.diffuser_agent import *
from agents.bc_agent.agent import *
from agents.bc_agent.pose_predictor import PosePredictor

import argparse

config.wandb_activate = False
parser = argparse.ArgumentParser(description="Process task and interface IDs")
config.BATCH_SIZE = 16
# Add arguments for task_id and interface_id, expecting multiple values for each
parser.add_argument("--task_id", type=int, nargs='+', required=True, help="List of task IDs")
parser.add_argument("--interface_id", type=int, nargs='+', required=True, help="List of interface IDs")

# Parse the arguments
args = parser.parse_args()

# Access and print the parsed arguments
print("Task IDs:", args.task_id)
print("Interface IDs:", args.interface_id)


config.tasks = args.task_id
config.interfaces = args.interface_id

config.model_path = f'./{config.agent}/checkpoint/[3, 4]_[1, 2, 3]_cam_{config.CAMERAS}/model_5000.pth'

run_name = "one_interface_test{}_{}_interface_{}only".format(config.agent, config.tasks, config.interfaces)
if config.wandb_activate == True:
    wandb.init(project="test_diffuser", name=run_name, entity="crl_rl2", dir="./icmi_log/")
    wandb.config.update(config)

def load_test_dataset(tasks: List[str], interfaces: List[int], start_index: int, end_index: int):
    test_replay_buffer = get_test_dataset(tasks, interfaces, start_index, end_index)
    test_wrapped_replay = PyTorchReplayBuffer(test_replay_buffer)
    test_dataset = test_wrapped_replay.dataset()
    return test_dataset


def load_agent(agent_path, agent=None, only_epoch=False):
    checkpoint = torch.load(agent_path, map_location="cpu")
    if config.agent == 'diffuser_agent':
        model = agent._q._qnet.noise_pred_net
    elif config.agent == 'bc_agent':
        model = agent._q._qnet.pred_net
        print ("inside bc_agent ", agent_path)

    model.load_state_dict(checkpoint["model_state"])
        
    return agent

def create_agent():
    # initialize PerceiverActor
    
    if config.agent == 'diffuser_agent':
        mvdit = MoveDit(depth=1, 
                    iterations=1,
                    voxel_size=config.VOXEL_SIZES[0],
                    initial_dim=config.initial_dim,
                    low_dim_size=config.low_dim_size
                    )

        agent = DiffuserActorAgent(
            coordinate_bounds=config.SCENE_BOUNDS,
            perceiver_encoder=mvdit,
            camera_names=config.CAMERAS,
            batch_size=config.BATCH_SIZE,
            voxel_size=config.VOXEL_SIZES[0],
            voxel_feature_size=3,
            num_rotation_classes=72,
            rotation_resolution=5,
            lr=config.lr,
            image_resolution=[config.IMAGE_SIZE, config.IMAGE_SIZE],
            lambda_weight_l2=0.000001,
            transform_augmentation=False,
            optimizer_type=config.optimizer,
        )
        agent.build(training=True, device=config.device)
    elif config.agent == 'bc_agent':
        print ("inside bc_agent ", config.SCENE_BOUNDS)
        pose_pred = PosePredictor(depth=1, 
                    iterations=1,
                    voxel_size=config.VOXEL_SIZES[0],
                    initial_dim=config.initial_dim,
                    low_dim_size=config.low_dim_size,
                    obs_dim = 512*2)

        agent = BCAgent(
            coordinate_bounds=config.SCENE_BOUNDS,
            perceiver_encoder=pose_pred,
            camera_names=config.CAMERAS,
            batch_size=config.BATCH_SIZE,
            voxel_size=config.VOXEL_SIZES[0],
            voxel_feature_size=3,
            num_rotation_classes=72,
            rotation_resolution=5,
            lr=config.lr,
            image_resolution=[config.IMAGE_SIZE, config.IMAGE_SIZE],
            lambda_weight_l2=0.000001,
            transform_augmentation=False,
            optimizer_type=config.optimizer,
        )
        agent.build(training=True, device=config.device)

    agent = load_agent(config.model_path, agent)
    return agent


def save_agent(agent, path, epoch):
    if config.agent == 'diffuser_agent':
        model = agent._q._qnet.noise_pred_net
    else:
        model = agent._q._qnet.pred_net
    optimizer = agent._optimizer
    
    lr_sched = agent.lr_scheduler


    model_state = model.state_dict()
    os.makedirs(path, exist_ok=True)
    save_path = f"{path}/model_{epoch}.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        save_path,
    )
    

def run_experiment():
    print ("config ", config.tasks)
    test_dataset = load_test_dataset(config.tasks, config.interfaces, start_index=config.train_split, end_index=16)
    test_data_iter = iter(test_dataset)


    diff_agent = create_agent()
    

    loss_list = list()
    start_time = time.time()

    for iteration in range(config.TEST_ITERATIONS):
        test_batch = next(test_data_iter)

        test_batch = {k: v.to(config.device) for k, v in test_batch.items() if type(v) == torch.Tensor}

        eval_dict = diff_agent.evaluate(iteration, test_batch)
        
        if iteration % config.LOG_FREQ == 0:
            elapsed_time = (time.time() - start_time) / 60.0
            # print("Total Training Loss: %f | Elapsed Time: %f mins" % (update_dict['total_loss'], elapsed_time))
            if config.wandb_activate == True:
                wandb.log({"iteration": iteration, 
                    "test_loss/total": eval_dict['total_loss'], "test_loss/translation": eval_dict['trans_loss'], 
                    "test_loss/rotation": eval_dict['rot_loss'], "test_loss/gripper": eval_dict['grip_loss']            
                    })
            loss_list.append(eval_dict['trans_loss'].detach().cpu())
            # print("Total Test Loss: %f | Elapsed Time: %f mins" % (eval_dict['total_loss'], elapsed_time))
    
    print (100*sum(loss_list)/config.TEST_ITERATIONS)

            



run_experiment()




