import copy
from arm.utils import stack_on_channel, quaternion_to_discrete_euler, normalize_quaternion, discretize_euler
import torch
import torch.nn as nn

class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 rotation_resolution: float,
                 device,
                 training):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._qnet = (perceiver_encoder)
        self._qnet._dev = device

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        
        coords = q_trans
        rot_and_grip_indicies = None
        q_rot = q_rot_grip[:, :3]
        q_grip = torch.round(q_rot_grip[:, -1]).cpu().unsqueeze(1)
        
        quat = normalize_quaternion(q_rot.detach().cpu().numpy())        
        quat[quat[:, 0]<=0] = 1e-10
        quat[quat[:, 1]<=0] = 1e-10
        quat[quat[:, 2]<=0] = 1e-10

        quat[quat[:, 0]>=360] = 360
        quat[quat[:, 1]>=360] = 360
        quat[quat[:, 2]>=360] = 360
        # quat[quat>360] = 360
        disc_rot = torch.from_numpy(discretize_euler(quat, resolution=5))
        if q_rot_grip is not None:
            rot_and_grip_indicies = torch.cat(
                [disc_rot,
                 q_grip], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self,
                obs,
                proprio,
                action,
                pcd,                
                bounds=None,
                eval=False):

        # batch bounds if necessary
        bs = obs.shape[0]
        if bounds.shape[0] != bs:
            bounds = bounds.repeat(bs, 1)
        
        obs_nan = torch.isnan(obs).any().item()
        pcd_nan = torch.isnan(pcd).any().item()
        proprio_nan = torch.isnan(proprio).any().item()

        # forward pass
        if eval == True:
            with torch.no_grad():
                q_trans, rot_and_grip_q, collision_q = self._qnet.evaluate(obs,
                                                        pcd,
                                                        proprio,
                                                        action,
                                                        bounds)
                noise_trans, noise_rot, noise_col = q_trans, rot_and_grip_q, collision_q
        else:

            q_trans, rot_and_grip_q, collision_q, noise = self._qnet(obs,
                                                            pcd,
                                                            proprio,
                                                            action,
                                                            bounds)
            noise_trans, noise_rot, noise_col = noise[:, :3], noise[:, 3:6], noise[:, 6:]
            
        
        return q_trans, rot_and_grip_q, collision_q, noise_trans, noise_rot, noise_col

    def latents(self):
        return self._qnet.latent_dict