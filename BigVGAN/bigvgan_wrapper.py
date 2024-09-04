# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from attrdict import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import BigVGAN as Generator
import librosa
import torch.nn as nn

h = None
device = None
torch.backends.cudnn.benchmark = False

class BigVGAN(nn.Module):
    def __init__(self, ckpt_dir):
        super(BigVGAN, self).__init__()
        config_file = os.path.join(ckpt_dir, 'config.json')
        ckpt_path = os.path.join(ckpt_dir, 'g_05000000.zip')
        with open(config_file) as f:
            data = f.read()
            json_config = json.loads(data)
            self.config = AttrDict(json_config)

        self.generator = Generator(self.config)
        state_dict_g = self.load_checkpoint(ckpt_path)['generator']
        self.generator.load_state_dict(state_dict_g)

    def load_checkpoint(self, filepath, device='cpu'):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict
    
    def get_mel(self, x):
        # egs. x: [1, 24000], mel: [1, 100, 937]
        h = self.config
        mel = mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)    
        return mel

    def get_mel_from_path(self, path, device='cpu'):
        wav, sr = librosa.load(path, sr=self.config.sampling_rate, mono=True)
        wav = torch.FloatTensor(wav).to(device)
        x = self.get_mel(wav.unsqueeze(0))
        return x
    
    def scan_checkpoint(cp_dir, prefix):
        pattern = os.path.join(cp_dir, prefix + '*')
        cp_list = glob.glob(pattern)
        if len(cp_list) == 0:
            return ''
        return sorted(cp_list)[-1]

    def recon_wav(self, mel, to_int=True):
        y_g_hat = self.generator(mel)
        audio = y_g_hat.squeeze()
        if to_int:
            audio = audio * MAX_WAV_VALUE
        return audio

    def get_wav(self, mel, outfile=None, ratio=1.0, norm=False):
        with torch.no_grad():
            y_g_hat = self.generator(mel)
        audio = y_g_hat.squeeze()
        if norm:
            a, b = -0.8, 0.8
            k = (b - a) / (audio.max() - audio.min())
            audio = a + k * (audio - audio.min())
            #audio = audio - audio.mean()
            #audio = audio / audio.abs().max()
        audio = audio * MAX_WAV_VALUE * ratio
        audio = audio.cpu().numpy().astype('int16')

        if outfile is not None:
            write(outfile, self.config.sampling_rate, audio)
        return audio
    
def main():
    print('Initializing Inference Process..')

    root_dir = 'bigvgan_24khz_100band'
    config_file = os.path.join(root_dir, 'config.json')
    checkpoint_dir = os.path.join(root_dir, 'g_05000000.zip')

    model = BigVGAN(root_dir).cuda()
    x = torch.randn((1, 240000)).cuda()
    print(x)
    mel = model.get_mel(x)
    print(mel.size())
    wav = model.get_wav(mel)


if __name__ == '__main__':
    main()

