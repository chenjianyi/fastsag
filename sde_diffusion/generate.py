import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys, time
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))
sys.path.append('../BigVGAN')
sys.path.append('../')
print(sys.path)
import random
from scipy.io.wavfile import write

import torch
import torchaudio
from einops import rearrange
import argparse
from torchaudio.functional import resample

from hubert_kmeans.model import create_hubert_kmeans_from_config
from BigVGAN.bigvgan_wrapper import BigVGAN
from fastsag import FastSAG
from utils_ import plot_tensor, save_plot, plot_curve, plot_curve2
from utils_d import zero_mean_unit_var_norm
from utils_d import norm_spec, denorm_spec

dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000

def convert(wav, vocoder, outfile=None):
    mel = vocoder.get_mel(wav.cuda())
    wav = vocoder.get_wav(mel, outfile)
    return wav

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run inference on trained musiclm model')
    parser = argparse.ArgumentParser(description='train diffusion song')
    parser.add_argument('--n_samples', default=1, type=int)
    parser.add_argument('--bivgan_ckpt', default='../weights/bigvgan_24khz_100band')

    parser.add_argument('--ckpt', default='../weights/fastsag.pt')
    parser.add_argument('--data_dir', default='../tmp_data')
    parser.add_argument('--result_dir', default='./output')
    args = parser.parse_args()

    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sample_rate = 24000

    wav2vec = create_hubert_kmeans_from_config(kmeans_path=None, device='cpu')
    bigvgan = BigVGAN(args.bivgan_ckpt)

    model = FastSAG(1, 64, 100, dec_dim, beta_min, beta_max, pe_scale, \
            vocoder=bigvgan, wav2vec=wav2vec, mix_type='wavenet').cuda()
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _list = []
    for _root, _dirs, _files in os.walk(args.data_dir):
        for _file in _files:
            path = os.path.join(_root, _file)
            if '_no_vocals' in path:
                continue
            nv_path = path.replace('_vocals', '_no_vocals')
            _list.append([path, nv_path])
    random.seed(31415926)
    random.shuffle(_list)

    t = 0
    for i, [v_path, nv_path] in enumerate(_list[0: 2000]):
        t1 = time.time()
        vocal_wav, sample_hz = torchaudio.load(v_path)
        nv_wav, nv_sample_hz = torchaudio.load(nv_path)

        ### gt wave
        try:
            v_wav_gt = resample(vocal_wav, sample_hz, sample_rate)
            nv_wav_gt = resample(nv_wav, nv_sample_hz, sample_rate)
        except:
            continue

        v_wav_gt = v_wav_gt[:, 0: sample_rate*10]   # take 10 seconds clips
        nv_wav_gt = nv_wav_gt[:, 0: sample_rate*10]

        norm = True   # normalize wave
        if norm:
            v_wav_gt = zero_mean_unit_var_norm(v_wav_gt)
            nv_wav_gt = zero_mean_unit_var_norm(nv_wav_gt)
        wav_gt = v_wav_gt + nv_wav_gt
       

        ### generation
        x = {}
        x['wav'] = v_wav_gt.cuda()
        norm_mel = True
        with torch.no_grad():
            x['mel'] = model.vocoder.get_mel(x['wav'].squeeze(1))
            x['semantic'] = model.wav2vec(x['wav'].squeeze(1), return_embed=True)  # (bs, T1, d)
        if norm_mel:  # normalize Mel
            x['mel'] = norm_spec(x['mel'])

        for j in range(args.n_samples):
            vocal_wav = vocal_wav + torch.randn_like(vocal_wav) * 0.01  # add noise to vocal_wav
            with torch.no_grad():
                y_enc, y_dec, loss = model(x, n_timesteps=50, y=None, use_cfg=False, use_x_mel=False)
            if norm_mel:
                y_enc = denorm_spec(y_enc)
                y_dec = denorm_spec(y_dec)
            mel = y_dec[:, : ]
            generated_wave = model.vocoder.get_wav(mel, ratio=0.9)
            t2 = time.time()
            t = t + (t2 - t1)

            print("time:", i+1, t2-t1, t/(i+1))

            v_wav_gt2 = convert(v_wav_gt, model.vocoder)
            write('%s/%03d_%s_gen.wav' % (result_dir, i,  j), sample_rate, v_wav_gt2+generated_wave)


        torchaudio.save('%s/%03d_gt.wav' % (result_dir, i), wav_gt, sample_rate)

        v_wav_gt2 = convert(v_wav_gt, model.vocoder)
        nv_wav_gt2 = convert(nv_wav_gt, model.vocoder)
        write('%s/%03d_gt.wav' % (result_dir, i), sample_rate, v_wav_gt2+nv_wav_gt2)


