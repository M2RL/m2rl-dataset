import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from agents.mdeit_utils import *

device = 'cuda'
class PosePredictor(nn.Module):
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

        self.single_image_ft = True
        self.pred_net = nn.Sequential(
                            nn.Linear((obs_dim*num_views) + action_dim, num_latents),
                            nn.Linear(num_latents, action_dim)
                            )
                                   


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


        next_wpt = self.pred_net(overall_latents)

        
        translation, rotation, gripper = next_wpt[:, :3], next_wpt[:, 3:6], next_wpt[:, 6:]
        return translation, rotation, gripper, noise

    
    def evaluate(self,
                latents,
                depth_latents=None,
                proprio=None,
                action=None,
                description=None,
                **kwargs):
        with torch.no_grad():
            bs = proprio.shape[0]
            overall_latents = torch.cat((latents, depth_latents), dim=1)

            overall_latents = overall_latents.reshape(overall_latents.shape[0], overall_latents.shape[1], overall_latents.shape[2], -1)        

            noise = torch.randn((action.shape)).to(device)

            overall_latents = overall_latents.reshape(overall_latents.shape[0], -1)
            overall_latents = torch.cat((overall_latents, proprio), dim=1)


            next_wpt = self.pred_net(overall_latents)

            
            translation, rotation, gripper = next_wpt[:, :3], next_wpt[:, 3:6], next_wpt[:, 6:]
        return translation, rotation, gripper, noise


