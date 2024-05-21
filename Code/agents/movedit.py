import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

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
        latent_dim=512
    ):
        super().__init__()
        
        # self.preprocess_image  
        # self.preprocess_pcd 
        # self.preprocess_proprio
        # self.preprocess_lang 
        action_dim = 9
        obs_dim = 16*16*4
        num_views = 4
        self.scheduler = self.load_scheduler('stabilityai/stable-diffusion-2-base')
        self.single_image_ft = True
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*num_views
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
            lang_emb=None,
            description=None,
            **kwargs,):
        
        overall_latents = torch.cat((latents, depth_latents), dim=1)
        overall_latents = overall_latents.reshape(overall_latents.shape[0], overall_latents.shape[1], overall_latents.shape[2], -1)
        prompt_embds = lang_emb
        
        noised_action = torch.randn((proprio.shape)).to(device)

        noise = torch.randn_like(latents)
        latents = latents.reshape(latents.shape[0], -1)
        timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (bs,), device=device
                ).long()

        noise_z = self.scheduler.add_noise(proprio, noised_action, timesteps)
        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noise_z, timesteps, global_cond=latents)

        # L2 loss
        noise_pred = noise_pred.squeeze(1)        
        
        
        return noise_pred[:, :3], noise_pred[:, 3:6], noise_pred[:, 6:]

    
    
    
mvdit = MoveDit(depth=1, 
                iterations=1,
                voxel_size=100,
                initial_dim=512,
                low_dim_size=64)