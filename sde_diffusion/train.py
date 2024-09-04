import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))
sys.path.append('../BigVGAN')

import torch
import argparse

from hubert_kmeans.model import create_hubert_kmeans_from_config
from dataset import SoundDataset, get_dataloader
from BigVGAN.bigvgan_wrapper import BigVGAN 

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fastsag import FastSAG
from utils_ import plot_tensor, save_plot, plot_curve, plot_curve2
from utils_d import norm_spec, denorm_spec

out_size = 2*22050//256

dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train diffusion song')
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=28, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--bivgan_ckpt', default='../weights/bigvgan_24khz_100band')
    parser.add_argument('--ckpt', default='../weights/fastsag.pt')
    parser.add_argument('--data_dir', default='clips_10s')
    parser.add_argument('--data_dir_testset', default='clips_10s_testset')
    parser.add_argument('--results_folder', default='./workspace/exps')
    args = parser.parse_args()

    print('Initializing logger...')
    log_dir = args.results_folder
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ckpt_dir = os.path.join(log_dir, 'ckpt')
    image_dir = os.path.join(log_dir, 'image')
    summary_dir = os.path.join(log_dir, 'tensorboard')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    logger = SummaryWriter(log_dir=summary_dir)

    wav2vec = create_hubert_kmeans_from_config(kmeans_path=None, device='cpu')
    bigvgan = BigVGAN(args.bivgan_ckpt)

    data_key = ('vocal', 'non_vocal')
    normalize = (True, True)
    target_sample_hz = (bigvgan.config.sampling_rate, bigvgan.config.sampling_rate)  #, wav2vec.target_sample_hz)
    mixed_F0 = (False, False)
    print(target_sample_hz)

    print('Initializing data loaders...')
    train_dataset = SoundDataset(args.data_dir, max_length_seconds=10, data_key=data_key, mixed_F0=mixed_F0, \
            normalize=normalize, target_sample_hz=target_sample_hz, seq_len_multiple_of=None)
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    test_dataset = SoundDataset(args.data_dir_testset, max_length_seconds=10, data_key=data_key, mixed_F0=mixed_F0, \
            normalize=normalize, target_sample_hz=target_sample_hz, seq_len_multiple_of=None)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    print('Initializing model...')
    model = FastSAG(1, 64, 100, dec_dim, beta_min, beta_max, pe_scale, \
            vocoder=bigvgan, wav2vec=wav2vec, mix_type='wavenet').cuda()
    if os.path.exists(args.ckpt):
        print('Load checkpoint from: %s' % args.ckpt)
        state_dict = torch.load(args.ckpt)
        model.load_state_dict(state_dict, strict=False)
    #print('Number of encoder + duration predictor parameters: %.2fm' % (model.mel_encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    print('Start training...')
    iteration = 0
    norm_mel = True
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        semantic_losses = []
        prior_losses = []
        diff_losses = []
        N = len(train_dataset)//args.batch_size
         
        with tqdm(train_loader, total=len(train_dataset)//args.batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, y = batch
                x = {k: x[k].cuda() for k in x}
                y = {k: y[k].cuda() for k in y}
                with torch.no_grad():
                    model.wav2vec.eval()
                    model.vocoder.eval()
                    x['semantic'] = model.wav2vec(x['wav'].squeeze(1), return_embed=True)  # (bs, T1, d)
                    y['semantic'] = model.wav2vec(y['wav'].squeeze(1), return_embed=True)
                    if 'mel' not in x or 'mel' not in y:
                        x['mel'] = model.vocoder.get_mel(x['wav'].squeeze(1))  # (bs, d, T)
                        y['mel'] = model.vocoder.get_mel(y['wav'].squeeze(1))

                    if norm_mel:
                        x['mel'] = norm_spec(x['mel'])
                        y['mel'] = norm_spec(y['mel'])
            
                semantic_loss, prior_loss, diff_loss, xt = model.compute_loss(x, y, out_size=out_size, use_x_mel=False, cfg=False)

                bs, d, T = y['mel'].size()


                loss = sum([semantic_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.mixed_encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar('training/semantic_loss', semantic_loss.item(),
                                    global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                
                semantic_losses.append(semantic_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration}/{N} | seman_loss: {semantic_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    progress_bar.set_description(msg)
                if batch_idx % 50 == 0:
                    with open(f'{log_dir}/train.log', 'a') as f:
                        f.write(msg)
                
                iteration += 1
       
                if iteration % 10000 == 0:
                    log_msg = 'Epoch %d: semantic loss = %.3f ' % (epoch, np.mean(semantic_losses))
                    log_msg += '| prior loss = [%.3f]' % np.mean(prior_losses)
                    log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
                    with open(f'{log_dir}/train.log', 'a') as f:
                        f.write(log_msg)

                    save_every = 1
                    if epoch % save_every > 0:
                        continue

                    model.eval()
                    """
                    print('Evaluation...')
                    semantic_losses, enc_prior_losses1, enc_prior_losses2, dec_prior_losses1, dec_prior_losses2, diff_losses = [], [], [], [], [], []
                    with torch.no_grad():
                        for i, item in enumerate(test_loader):
                            x, y = item
                            x = {k: x[k].cuda() for k in x}
                            y = {k: y[k].cuda() for k in y}
                            with torch.no_grad():
                                model.wav2vec.eval()
                                model.vocoder.eval()
                                x['semantic'] = model.wav2vec(x['wav'].squeeze(1), return_embed=True)  # (bs, T1, d)
                                y['semantic'] = model.wav2vec(y['wav'].squeeze(1), return_embed=True)
                                x['mel'] = model.vocoder.get_mel(x['wav'].squeeze(1))  # (bs, d, T)
                                y['mel'] = model.vocoder.get_mel(y['wav'].squeeze(1))  # (bs, d, T)
                                x['mixed'] = torch.cat([x['f0'], x['mixed_F'], x['mel']], dim=1).float()
                                y['mixed'] = torch.cat([y['f0'], y['mixed_F'], y['mel']], dim=1).float()
                            y_enc, y_dec, loss = model(x, n_timesteps=50, y=y)
                            (semantic_loss, enc_prior_loss1, enc_prior_loss2, dec_prior_loss1, dec_prior_loss2) = loss

                            semantic_losses.append(semantic_loss.item())
                            enc_prior_losses1.append(enc_prior_loss1.item())
                            enc_prior_losses2.append(enc_prior_loss2.item())
                            dec_prior_losses1.append(dec_prior_loss1.item())
                            dec_prior_losses2.append(dec_prior_loss2.item())
 
                        log_msg = '[Evaluation] Epoch %d: semantic loss = %.3f ' % (epoch, np.mean(semantic_losses))
                        log_msg += '| enc_prior loss = [%.3f %.3f]' % (np.mean(enc_prior_losses1), np.mean(enc_prior_losses2))
                        log_msg += '| dec_prior loss = [%.3f %.3f]\n' % (np.mean(dec_prior_losses1), np.mean(dec_prior_losses2))
                        with open(f'{log_dir}/train.log', 'a') as f:
                            f.write(log_msg)
                        print(log_msg)
                    """
                    print('Synthesis...')
                    with torch.no_grad():
                        for i, item in enumerate(test_loader):
                            x, y = item
                            x = {k: x[k][0: 1].cuda() for k in x}
                            y = {k: y[k][0: 1].cuda() for k in y}
                            with torch.no_grad():
                                model.wav2vec.eval()
                                model.vocoder.eval()
                                x['semantic'] = model.wav2vec(x['wav'].squeeze(1), return_embed=True)  # (bs, T1, d)
                                y['semantic'] = model.wav2vec(y['wav'].squeeze(1), return_embed=True)
                                x['mel'] = model.vocoder.get_mel(x['wav'].squeeze(1))  # (bs, d, T)
                                y['mel'] = model.vocoder.get_mel(y['wav'].squeeze(1))  # (bs, d, T)
                                if norm_mel:
                                    x['mel'] = norm_spec(x['mel'])
                                    y['mel'] = norm_spec(y['mel'])
                            image = int(iteration / 10000)
                            logger.add_image(f'{image}_{i}/x_ground_truth', plot_tensor(x['mel'].squeeze().cpu()),
                                            global_step=0, dataformats='HWC')
                            save_plot(x['mel'].squeeze().cpu(), f'{log_dir}/image/x_original_{image}_{i}.png')
                            logger.add_image(f'{image}_{i}/y_ground_truth', plot_tensor(y['mel'].squeeze().cpu()),
                                            global_step=0, dataformats='HWC')
                            save_plot(y['mel'].squeeze().cpu(), f'{log_dir}/image/y_original_{image}_{i}.png')

                            y_enc, y_dec, loss = model(x, n_timesteps=50, y=None, use_x_mel=False)
                            if norm_mel:
                                y_enc = denorm_spec(y_enc)
                                y_dec = denorm_spec(y_dec)

                            indx = list(range(0, 32)) + [51, 81, 111]
                            indx2 = list(range(0, 32)) + [19, 49, 79]
                            logger.add_image(f'{image}_{i}/generated_enc',
                                    plot_tensor(y_enc[:, 32:, :].squeeze().cpu()),
                                             global_step=iteration, dataformats='HWC')
                            logger.add_image(f'{image}_{i}/generated_dec',
                                    plot_tensor(y_dec[:, 32:, :].squeeze().cpu()),
                                             global_step=iteration, dataformats='HWC')
                            save_plot(y_enc.squeeze().cpu(), 
                                      f'{log_dir}/image/generated_enc_{image}_{i}.png')
                            save_plot(y_dec.squeeze().cpu(), 
                                      f'{log_dir}/image/generated_dec_{image}_{i}.png')
                            if i >= 5:
                                break

                if iteration % 10000 == 0:
                    ckpt = model.state_dict()
                    torch.save(ckpt, f=f"{log_dir}/ckpt/grad_{iteration}.pt")
