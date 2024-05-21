import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from agents.mdeit_utils import *

device = 'cuda'
class MoveDit(nn.Module):
    def __init__(
        self,
        depth, 
        iterations,
        voxel_size,
        initial_dim,
        low_dim_size,
        layer=0,    
        num_rotation_classes=72,
        num_grip_classes=2,
        num_collision_classes=2,
        input_axis=3,
        num_latents=512,
        im_channels=64,
        latent_dim=512,
        obs_dim = 16*16*4*2
    ):
        super().__init__()
        
        # self.preprocess_image  
        # self.preprocess_pcd 
        # self.preprocess_proprio
        # self.preprocess_lang 
        action_dim = 7
        num_views = len(CAMERAS)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=100,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        self.single_image_ft = True
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=(obs_dim*num_views) + action_dim
        )


    def load_scheduler(self, model_id):
        
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        return scheduler
        
    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask)

        return prompt_embeds[0].float(), prompt_embeds[1]

    

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)
        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype('uint8')

        return image

    def forward(self,latents,
            depth_latents=None,
            proprio=None,
            action = None,
            description=None,
            **kwargs,):
        
        bs = proprio.shape[0]
        overall_latents = torch.cat((latents, depth_latents), dim=1)

        overall_latents = overall_latents.reshape(overall_latents.shape[0], overall_latents.shape[1], overall_latents.shape[2], -1)        

        noise = torch.randn((action.shape)).to(device)

        overall_latents = overall_latents.reshape(overall_latents.shape[0], -1)
        overall_latents = torch.cat((overall_latents, proprio), dim=1)

        latents = latents.reshape(latents.shape[0], -1)
        timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (bs,), device=device
                ).long()

        noise_z = self.scheduler.add_noise(action, noise, timesteps)
        # predict the noise residual
        # print ("noisy z: ", noise_z.shape)
        noise_pred = self.noise_pred_net(
            noise_z, timesteps, global_cond=overall_latents)

        # L2 loss
        noise_pred = noise_pred.squeeze(1)        
        
        translation, rotation, gripper = noise_pred[:, :3], noise_pred[:, 3:6], noise_pred[:, 6:]
        return translation, rotation, gripper, noise

    
    def evaluate(self,
                latents,
                depth_latents=None,
                proprio=None,
                action=None,
                description=None,
                **kwargs):

        bs = proprio.shape[0]
        overall_latents = torch.cat((latents, depth_latents), dim=1)

        overall_latents = overall_latents.reshape(overall_latents.shape[0], overall_latents.shape[1], overall_latents.shape[2], -1)        
        overall_latents = overall_latents.reshape(overall_latents.shape[0], -1)
        overall_latents = torch.cat((overall_latents, proprio), dim=1)

        latents = latents.reshape(latents.shape[0], -1)

        noised_action = torch.randn((proprio.shape)).to(device)

        
        self.scheduler.set_timesteps(100)
        for k in self.scheduler.timesteps:
            noise_pred = self.noise_pred_net(noised_action, 
                                             timestep=k, 
                                             global_cond=overall_latents)
            noise_pred = noise_pred.squeeze(1)
            noised_action = self.scheduler.step(model_output=noise_pred, 
                                                  timestep=k, 
                                                  sample=noised_action).prev_sample
        translation, rotation, gripper = noised_action[:, :3], noised_action[:, 3:6], noised_action[:, 6:]
        
        return translation, rotation, gripper

