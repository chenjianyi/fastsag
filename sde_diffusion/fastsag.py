import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import Diffusion, DiffusionEDM
from utils_ import sequence_mask, generate_path, duration_loss, fix_len_compatibility
from wavenet import Wavenet
from unet import Unet2d
from perceiver_io import PerceiverIO
from unet1d import Unet1D
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel, UNet1DModel

class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

class FastSAG(BaseModule):
    def __init__(self, n_spks=1, spk_emb_dim=64, 
                 n_feats=100, dec_dim=64, beta_min=0.05, beta_max=20, pe_scale=1000,
                 vocoder=None, wav2vec=None, mel_L=937, mix_type='wavenet'):
        super(FastSAG, self).__init__()
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.vocoder = vocoder
        self.wav2vec = wav2vec
        self.mel_L = mel_L
        self.mix_type = mix_type

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.decoder = DiffusionEDM(n_feats=n_feats, dim=dec_dim, n_spks=n_spks, spk_emb_dim=spk_emb_dim, 
                beta_min=beta_min, beta_max=beta_max, pe_scale=pe_scale, dim_mults=(1,2,4))

        if mix_type == 'unet1d-v1':
            self.mixed_encoder = Unet1D(dim=64, dim_mults=(1, 2, 4, 8), channels=100, self_condition=True, attn_dim_head=64, attn_heads=8)
        elif mix_type == 'unet1d-v2':
            self.mixed_encoder = UNet1DModel(sample_size=1024, in_channels=self.n_feats+16, out_channels=self.n_feats, layers_per_block=2, block_out_channels=[256, 512,  512])
        elif mix_type == 'wavenet':
            self.mixed_encoder = Wavenet(layers=8, stacks=4, dim=self.n_feats)
        self.semantic_encoder = nn.ModuleList([
            Wavenet(layers=8, stacks=8, dim=768),
            PerceiverIO(dim=768, queries_num=self.mel_L, queries_dim=256, logits_dim=self.n_feats, num_latents=32, latent_dim=256, cross_heads=8, \
                    latent_heads=8, cross_dim_head=128, latent_dim_head=128, weight_tie_layers=False, seq_dropout_prob=0.1, depth=6),
            ])

        self.act_func = torch.nn.Tanh()

    @torch.no_grad()
    def forward(self, x, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0, y=None, use_x_mel=False, use_cfg=False):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        bs, d, T = x['mel'].size()
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        y_lengths = torch.tensor([x['mel'].size(-1)] * bs).to(x['mel'].device)
        L = x['mel'].size(-1)
        y_max_length = 1024  # 1024 > 937

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x['mel'].device)

        #semantic_x2y = self.semantic_encoder[0](x['semantic'])  # perceiver
        semantic_x2y = self.semantic_encoder[0](x['semantic'].permute(0, 2, 1)).permute(0, 2, 1)  # wavenet
        semantic_xy = semantic_x2y + x['semantic']

        use_interpolate = True
        if use_interpolate:
            semantic2mel = F.interpolate(semantic_xy.permute(0, 2, 1).unsqueeze(1), size=(d, T), mode='bilinear', align_corners=False)[:, 0, ...]
        else:
            semantic2mel = self.semantic_encoder[1](semantic_xy).permute(0, 2, 1).float()

        if use_x_mel:
            #x_cond = semantic2mel + x['mel']
            x_cond = x['mel']
        else:
            x_cond = semantic2mel

        pad = torch.zeros(bs, d, y_max_length-T).to(x_cond.device)
        x_cond = torch.cat([x_cond, pad], dim=-1)  # (bs, d, 1024)
        x_mel = torch.cat([x['mel'], pad], dim=-1)  # (bs, d, 1024)

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_cond.device)

        #mu_x = self.mixed_encoder(x_mel)

        if self.mix_type == 'wavenet':
            mu_x = self.mixed_encoder(x_cond, mask=y_mask)   # wavenet
        elif self.mix_type == 'unet1d-v1':
            mu_x = self.mixed_encoder(x_cond, x_self_cond=x_mel)  # unet1d
        elif self.mix_type == 'unet1d-v2':
            mu_x = self.mixed_encoder(x_cond, timestep=0).sample


        #mu_x = self.mixed_encoder(x_cond.float().permute(0, 2, 1)).permute(0, 2, 1)
        #mu_y = self.act_func(mu_x)
        mu_y = mu_x

        encoder_outputs = mu_x[:, :, :L]

        # Sample latent representation from terminal distribution N(mu_y, I)
        #z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        z = torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        #decoder_output = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk, x_cond)
        decoder_output = self.decoder(z, y_mask, mu_y, n_timesteps, spk, cond=None, use_cfg=use_cfg)
        decoder_outputs = decoder_output[:, :, :L]

        loss = None
        if y is not None:
            y_mixed = torch.cat([y['mixed'], pad], dim=-1)  # (bs, d, 1024)
            prior_loss1 = torch.sum(0.5 * ((y_mixed[:, 0: 32, :] - mu_x[:, 0: 32, :]) ** 2 + math.log(2 * math.pi)) * y_mask)
            enc_prior_loss1 = prior_loss1 / (torch.sum(y_mask) * 32)
            prior_loss2 = torch.sum(0.5 * ((y_mixed[:, 32:, :] - mu_x[:, 32:, :]) ** 2 + math.log(2 * math.pi)) * y_mask)
            enc_prior_loss2 = prior_loss2 / (torch.sum(y_mask) * self.n_feats)

            #prior_loss1 = torch.sum(0.5 * ((y_mixed[:, 0: 32, :] - decoder_output[:, 0: 32, :]) ** 2 + math.log(2 * math.pi)) * y_mask)
            #dec_prior_loss1 = prior_loss1 / (torch.sum(y_mask) * 32)
            dec_prior_loss1 = torch.zeros_like(enc_prior_loss2)
            prior_loss2 = torch.sum(0.5 * ((y_mixed[:, 32:, :] - decoder_output[:, :, :]) ** 2 + math.log(2 * math.pi)) * y_mask)
            dec_prior_loss2 = prior_loss2 / (torch.sum(y_mask) * self.n_feats)

            # Compute semantic loss
            semantic_loss = torch.mean((semantic_x2y - y['semantic']) ** 2)

            loss = (semantic_loss, enc_prior_loss1, enc_prior_loss2, dec_prior_loss1, dec_prior_loss2)
 
        return encoder_outputs, decoder_outputs, loss

    def compute_loss(self, x, y, spk=None, out_size=None, use_x_mel=False, cfg=False):
        """
        Computes 2 losses:
            1. prior loss: loss between mel-spectrogram and encoder outputs.
            2. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        bs, d, T = x['mel'].size()
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        y_lengths = torch.tensor([y['mel'].size(-1)] * bs).to(y['mel'].device)
        y_max_length = 1024

        #semantic_x2y = self.semantic_encoder[0](x['semantic'])  # perceiver, transformer
        semantic_x2y = self.semantic_encoder[0](x['semantic'].permute(0, 2, 1)).permute(0, 2, 1)  # wavenet
        semantic_xy = semantic_x2y + x['semantic']  # (bs, T, d)

        use_interpolate = True
        if use_interpolate:
            semantic2mel = F.interpolate(semantic_xy.permute(0, 2, 1).unsqueeze(1), size=(d, T), mode='bilinear', align_corners=False)[:, 0, ...]
        else:
            semantic2mel = self.semantic_encoder[1](semantic_xy).permute(0, 2, 1).float()

        if use_x_mel:
            #x_cond = semantic2mel + x['mel']
            x_cond = x['mel']
        else:
            x_cond = semantic2mel

        pad = torch.zeros(bs, d, 1024-T).to(x_cond.device)
        x_cond = torch.cat([x_cond, pad], dim=-1)  # (bs, d, 1024)
        y_mel = torch.cat([y['mel'], pad], dim=-1)  # (bs, d, 1024)
        x_mel = torch.cat([x['mel'], pad], dim=-1)  # (bs, d, 1024)

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(y['mel'].device)


        if self.mix_type == 'wavenet':
            mu_x = self.mixed_encoder(x_cond, mask=y_mask)   # wavenet
        elif self.mix_type == 'unet1d-v1':
            mu_x = self.mixed_encoder(x_cond, x_self_cond=x_mel)  # unet1d
        elif self.mix_type == 'unet1d-v2':
            mu_x = self.mixed_encoder(x_cond, timestep=0).sample

        #mu_x = self.mixed_encoder(x_mixed.float())
        #mu_x = self.mixed_encoder(x_cond.float(), mask=y_mask)  # wavenet

        #mu_x = self.mixed_encoder(x_cond.float().permute(0, 2, 1)).permute(0, 2, 1)

        #mu_y = self.act_func(mu_x)
        mu_y = mu_x

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y_mel, y_mask, mu_y, spk, cond=None, cfg=cfg)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y_mel - mu_x) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        # Compute semantic loss
        semantic_loss = torch.mean((semantic_x2y - y['semantic']) ** 2)
        
        return semantic_loss, prior_loss, diff_loss, xt
