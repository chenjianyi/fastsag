""" from https://github.com/jaywalnut310/glow-tts """

import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], 
                                          [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return loss

def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_curve(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    #ax.plot(librosa.times_like(tensor), tensor[0], label='rms', color=(1, 1, 1))
    #ax.plot(librosa.times_like(tensor), tensor[1], label='zero_rate', color=(0.8, 0.8, 0.8))
    #ax.plot(librosa.times_like(tensor), tensor[3], label='tonnetz(1)', color=(0.3, 0, 0))
    #ax.plot(librosa.times_like(tensor), tensor[5], label='tonnetz(3)', color=(0.6, 0, 0))
    #ax.plot(librosa.times_like(tensor), tensor[7], label='tonnetz(5)', color=(0.9, 0, 0))
    ax.plot(librosa.times_like(tensor), tensor[0], label='Roll-off(0.01)', color='r')
    ax.plot(librosa.times_like(tensor), tensor[3], label='Roll-off(0.3)', color='g')
    ax.plot(librosa.times_like(tensor), tensor[6], label='Roll-off(0.6)', color='b')
    ax.plot(librosa.times_like(tensor), tensor[8], label='Roll-off(0.8)', color='y')
    ax.plot(librosa.times_like(tensor), tensor[10], label='Roll-off(0.99)', color='k')
    ax.legend(loc='upper right')
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_curve2(tensor1, tensor2, tensor3):
    _dict = {0: 'f0', 1: 'rms', 2: 'zero_rate', 32: 'mel(20)', 33: 'mel(50)', 34: 'mel(80)'}
    for i in range(3, 9):
        _dict[i] = 'tonnetz%s' % (i-3)
    for i in range(9, 21):
        _dict[i] = 'chroma%s' % (i-9)
    for i, percent in enumerate(np.linspace(0.01, 0.99, 11)):
        _dict[i+21] = 'roll(%s)' % percent

    plt.style.use('default')
    fig, ax = plt.subplots(5, 3, figsize=(60, 15))
    for i, idx in enumerate([0, 1, 2] + [4, 6, 8] + [10, 14, 19] + [21, 25, 31] + [32, 33, 34]):
        fig_i = int(i / 3)
        fig_j = int(i % 3)
        ax[fig_i, fig_j].plot(librosa.times_like(tensor1), tensor1[idx], label=_dict[idx] + ': gt', color='r')
        ax[fig_i, fig_j].plot(librosa.times_like(tensor2), tensor2[idx], label=_dict[idx] + ': enc', color='g', alpha=0.2)
        if idx >= 32:
            ax[fig_i, fig_j].plot(librosa.times_like(tensor3), tensor3[idx], label=_dict[idx] + ': dec', color='b', alpha=0.2)
        ax[fig_i, fig_j].legend(loc='upper right')
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return
