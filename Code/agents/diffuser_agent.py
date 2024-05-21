from arm.optim.lamb import Lamb
from arm.utils import stack_on_channel, quaternion_to_discrete_euler, normalize_quaternion
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from agents.mdeit import MoveDit
from agents.qfunction import QFunction
from agents.mdeit_utils import _preprocess_inputs
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DiffuserActorAgent():
    def __init__(self,
                coordinate_bounds: list,
                perceiver_encoder: nn.Module,
                camera_names: list,
                batch_size: int,
                voxel_size: int,
                voxel_feature_size: int,
                num_rotation_classes: int,
                rotation_resolution: float,
                lr: float = 0.0001,
                image_resolution: list = None,
                lambda_weight_l2: float = 0.0,
                transform_augmentation: bool = True,
                transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                transform_augmentation_rot_resolution: int = 5,
                optimizer_type: str = 'adamW'):

        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._camera_names = camera_names
        self._batch_size = batch_size
        self._voxel_size = voxel_size
        self._voxel_feature_size = voxel_feature_size
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = transform_augmentation_xyz
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type

        self.vae, self.scheduler, self.unet = self.load_model('stabilityai/stable-diffusion-2-base')
        self.vae = self.vae.cuda()

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._mse_loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device
       
        self._q = QFunction(self._perceiver_encoder,                            
                            self._rotation_resolution,
                            device,
                            training).to(device).train(training)

        self.ema_model = EMAModel(
                         parameters=self._q._qnet.noise_pred_net.parameters(),
                         power=0.75
                         )
        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._optimizer_type == 'lamb':
            # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
            self._optimizer = Lamb(
                self._q.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
                betas=(0.9, 0.999),
                adam=False,
            )
        elif self._optimizer_type == 'adam':
            self._optimizer = torch.optim.Adam(
                self._q.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        elif self._optimizer_type == 'adamW':
            self._optimizer = torch.optim.AdamW(
            params=self._q.parameters(),
            lr=1e-4, weight_decay=1e-6)

            # Cosine LR schedule with linear warmup
            self.lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=self._optimizer,
                num_warmup_steps=500,
                num_training_steps=10000
            )
        else:
            raise Exception('Unknown optimizer')

    def _softmax_q(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def huber_loss(self, output, target, delta=1.0):
        huber_loss = F.smooth_l1_loss(output, target, reduction='none')
        return huber_loss.mean()

    def _get_one_hot_expert_actions(self,  # You don't really need this function since GT labels are already in the right format. This is some leftover code from my experiments with label smoothing.
                                    batch_size,
                                    action_trans,
                                    action_rot_grip,
                                    action_ignore_collisions,
                                    device):
        bs = batch_size

        # initialize with zero tensors
        action_trans_one_hot = torch.zeros((bs, self._voxel_size, self._voxel_size, self._voxel_size), dtype=int, device=device)
        action_rot_x_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_y_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_z_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_grip_one_hot  = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
          
          # ignore collision
          gt_ignore_collisions = action_ignore_collisions[b, :]
          action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        # flatten trans
        action_trans_one_hot = action_trans_one_hot.view(bs, -1)

        return action_trans_one_hot, \
               action_rot_x_one_hot, \
               action_rot_y_one_hot, \
               action_rot_z_one_hot, \
               action_grip_one_hot,  \
               action_collision_one_hot
    
    def load_model(self, model_id):
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae")
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet")
        return vae, scheduler, unet

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]//len(self._camera_names)
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = z.reshape(b, -1, z.shape[-3], z.shape[-2],
                      z.shape[-1])  # (bs, 4, 4, 64, 64)

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        z = z.float()
        return z

    def _get_img_pcd(self, obs):
        images = [img for (img, pcd) in obs]
        pcds = [pcd for (img, pcd) in obs]
        images = torch.cat(images, 0)
        pcds = torch.cat(pcds, 0)
        bs = images.shape[0]//len(self._camera_names)
        images = images.view(images.shape[0]//bs, bs, images.shape[1], images.shape[2], images.shape[3])
        pcds = pcds.view(pcds.shape[0]//bs, bs, pcds.shape[1], pcds.shape[2], images.shape[3])
        
        images = images.permute(1, 0, 2, 3, 4)
        pcds = pcds.permute(1, 0, 2, 3, 4)
        images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
        pcds = pcds.reshape(-1, pcds.shape[2], pcds.shape[3], pcds.shape[4])
        
        # images = images.permute(1, 0, 2, 3)
        # pcds = pcds.permute(1, 0, 2, 3)
        # images = images.view(images.shape[0]*images.shape[1], images.shape[2], images.shape[3])
        # pcds = pcds.view(pcds.shape[0]*pcds.shape[1], pcds.shape[2], pcds.shape[3])
        return images, pcds

    def save_checkpoint(self):
        torch.save()
    
    def update(self, step: int, replay_sample: dict, backprop: bool = True) -> dict:
        # sample
        gt_action_trans = replay_sample['trans_action_indicies'][:, -1, :3]#.int()
        # action_trans = action_trans*100
        gt_action_rot_grip = replay_sample['rot_grip_action_indicies'][:, -1]#.int()
        action_ignore_collisions = replay_sample['ignore_collisions'][:, -1].int()
        action_gripper_pose = replay_sample['gripper_pose'][:, -1]        
        prev_gripper_pose = replay_sample['prev_gripper_pose'].squeeze(1)
        # print ("prev action_rot_grip pose ", action_rot_grip.shape, prev_gripper_pose[:, 3:])
        quat = prev_gripper_pose[:, 3:]
        
        proprio_new = torch.cat((prev_gripper_pose[:, :3], quat), dim=-1)
        # proprio_new = torch.cat((proprio_new, torch.randn((proprio_new.shape[0], 2)).cuda()), dim=-1)
        
        # metric scene bounds
        bounds = bounds_tp1 = self._coordinate_bounds
        # inputs        
        # proprio = stack_on_channel(proprio_new)
        # proprio = torch.cat((proprio, proprio), dim=-1)
        
        obs, pcd = _preprocess_inputs(replay_sample)
        # TODO: data augmentation by applying SE(3) pertubations to pcd and actions
        # see https://github.com/peract/peract/blob/main/voxel/augmentation.py#L68 for reference

        # retrive batched image and depth
        images, pcds = self._get_img_pcd(obs)
        

        latent_images = self.encode_image(images, self.vae)
        latent_depth = self.encode_image(pcds, self.vae)

        # Q function
        q_trans, rot_grip_q, collision_q, action_trans, action_rot_grip, noise_col = self._q(latent_images,
                                                               proprio_new,
                                                               torch.cat((gt_action_trans, gt_action_rot_grip), dim=-1 ),
                                                               latent_depth,
                                                               bounds)
        # one-hot expert actions
        bs = self._batch_size
        total_loss = 0.
        if backprop:
            # cross-entropy loss
            trans_loss = self.huber_loss(q_trans.view(bs, -1).float(),
                                                  action_trans.float())

            rot_loss = 0.
            # print ("prediction: ",q_trans[:, :3].view(bs, -1).float(), "gt: ", action_trans.float())
            rot_loss = self.huber_loss(rot_grip_q[:, :3].view(bs, -1).float(),
                                                  action_rot_grip[:, :3].float(),)
            grip_loss = self.huber_loss(collision_q[:, 0].float(), noise_col[:, -1].float())#.item()


            # collision_loss = self._cross_entropy_loss(collision_q,
            #                                           action_collision_one_hot.argmax(-1))

            total_loss = trans_loss + rot_loss + grip_loss  #+ collision_loss
            total_loss = total_loss.mean()
            # backprop
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            self.lr_scheduler.step()
            self.ema_model.step(self._q._qnet.noise_pred_net.parameters())


            total_loss = total_loss.item()
            """
            q_trans, rot_grip_q, collision_q, q_trans, rot_grip_q, collision_q = self._q(latent_images,
                                                                proprio_new,
                                                                proprio_new,
                                                                latent_depth,
                                                                bounds, eval=True)
            total_loss = 0.
            # cross-entropy loss
            trans_loss = self._mse_loss(q_trans.view(bs, -1).float(),
                                                    gt_action_trans.float())

            rot_loss = 0.
            # print ("prediction: ",q_trans[:, :3].view(bs, -1).float(), "gt: ", action_trans.float())
            rot_loss = self._mse_loss(rot_grip_q[:, :3].view(bs, -1).float(),
                                                    gt_action_rot_grip[:, :3].float(),)
            grip_loss = self._mse_loss(collision_q[:, 0].float(), gt_action_rot_grip[:, -1].float())#.item()

            # print ("rot_grip_q ", rot_grip_q[:, -1].float(), )

            total_loss = trans_loss  + rot_loss + grip_loss  #+ collision_loss
            total_loss = total_loss.mean()
            # backprop

            total_loss = total_loss.item()

            """
            

        # choose best action through argmax
        coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = self._q.choose_highest_action(q_trans,
                                                                                                          rot_grip_q,
                                                                                                          collision_q)
        
        # self._q.choose_highest_action(q_trans, rot_grip_q, collision_q)

        # print("images ", images.shape, action_trans.shape)
        


        return {
            'total_loss': total_loss,
            'trans_loss': trans_loss,
            'rot_loss': rot_loss,
            'grip_loss': grip_loss,
            'pred_action': {
                'trans': coords_indicies,                
                'rot_and_grip': rot_and_grip_indicies,
                'collision': ignore_collision_indicies
            },
            'expert_action': {
                'action_trans': action_trans
            }
        }



    def evaluate(self, step: int, replay_sample: dict, backprop: bool = False) -> dict:
        # sample
        write_folder = './icmi_tasks/output_images/'
        action_trans = replay_sample['trans_action_indicies'][:, -1, :3]#.int()
        action_rot_grip = replay_sample['rot_grip_action_indicies'][:, -1]#.int()
        action_ignore_collisions = replay_sample['ignore_collisions'][:, -1].int()
        action_gripper_pose = replay_sample['gripper_pose'][:, -1]        
        prev_gripper_pose = replay_sample['prev_gripper_pose'].squeeze(1)
        # print ("prev action_rot_grip pose ", action_rot_grip.shape, prev_gripper_pose[:, 3:])
        quat = prev_gripper_pose[:, 3:]
        
        proprio_new = torch.cat((prev_gripper_pose[:, :3], quat), dim=-1)
        # proprio_new = torch.cat((proprio_new, torch.randn((proprio_new.shape[0], 2)).cuda()), dim=-1)
        
        # metric scene bounds
        bounds = bounds_tp1 = self._coordinate_bounds
        # inputs        
        # proprio = stack_on_channel(proprio_new)
        # proprio = torch.cat((proprio, proprio), dim=-1)
        
        obs, pcd = _preprocess_inputs(replay_sample)
        # TODO: data augmentation by applying SE(3) pertubations to pcd and actions
        # see https://github.com/peract/peract/blob/main/voxel/augmentation.py#L68 for reference

        # retrive batched image and depth
        transform = T.ToPILImage()
        images, pcds = self._get_img_pcd(obs)
        with torch.no_grad():
            latent_images = self.encode_image(images, self.vae)
            latent_depth = self.encode_image(pcds, self.vae)
            # for idx in range(len(images)):
            #     transform(images[idx]).save(f"{write_folder}/{step}_{idx}.png")
            # Q function
            # print ("self.ema_model ", self.ema_model)
            q_trans, rot_grip_q, collision_q, q_trans, rot_grip_q, collision_q = self._q(latent_images,
                                                                proprio_new,
                                                                proprio_new,
                                                                latent_depth,
                                                                bounds, eval=True)
            # one-hot expert actions
            bs = self._batch_size

            total_loss = 0.
            # cross-entropy loss
            trans_loss = self._mse_loss(q_trans.view(bs, -1).float(),
                                                    action_trans.float())

            rot_loss = 0.
            # print ("prediction: ",rot_grip_q[:, :3].view(bs, -1).float(), "gt: ", action_rot_grip.float())
            rot_loss = self._mse_loss(rot_grip_q[:, :3].view(bs, -1).float(),
                                                    action_rot_grip[:, :3].float(),)
            grip_loss = self._mse_loss(collision_q[:, 0].float(), action_rot_grip[:, -1].float())#.item()

            # print ("rot_grip_q ", rot_grip_q[:, -1].float(), )

            total_loss = trans_loss  + rot_loss + grip_loss  #+ collision_loss
            total_loss = total_loss.mean()
            # backprop

            total_loss = total_loss.item()
                
            # choose best action through argmax
            coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = self._q.choose_highest_action(q_trans,
                                                                                                            rot_grip_q,
                                                                                                            collision_q)
        # self._q.choose_highest_action(q_trans, rot_grip_q, collision_q)
        transform = T.ToPILImage()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)
        gt_x = action_trans[:,0].cpu().numpy()
        gt_y = action_trans[:,1].cpu().numpy()
        gt_z = action_trans[:,2].cpu().numpy()

        pred_x = q_trans[:,0].detach().cpu().numpy()
        pred_y = q_trans[:,1].detach().cpu().numpy()
        pred_z = q_trans[:,2].detach().cpu().numpy()

        ax.scatter(gt_x, gt_y, gt_z, c='red', label='Waypoints', s=100)
        ax.plot(gt_x, gt_y, gt_z, color='blue', label='Path')

        ax.scatter(pred_x, pred_y, pred_z, c='green', label='Waypoints', s=100)
        ax.plot(pred_x, pred_y, pred_z, color='yellow', label='Path')

    
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')

        # Title
        ax.set_title('3D Plot of Robot Rotation')
        plt.savefig(f'{write_folder}/3d_plot_{step}.png')
        plt.close()


        return {
            'total_loss': total_loss,
            'trans_loss': trans_loss,
            'rot_loss': rot_loss,
            'grip_loss': grip_loss,
            'pred_action': {
                'trans': coords_indicies,                
                'rot_and_grip': rot_and_grip_indicies,
                'collision': ignore_collision_indicies
            },
            'expert_action': {
                'action_trans': action_trans
            }
        }




