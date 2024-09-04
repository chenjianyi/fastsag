import os
import librosa

import soundfile as sf
import os
import random
import numpy as np
from tqdm import tqdm


src_root = 'dataset_dir_demucs'
des_root = 'dataset_dir_demucs_clips'

_list = []
suffix = 'wav'
for _root, _dirs, _files in os.walk(src_root):
    for _file in _files:
        path = os.path.join(_root, _file)
        name = os.path.basename(path)
        if 'no_vocals.%s' % suffix in path:
            _list.append(path)
random.shuffle(_list)

def peak_RMS_amplitude(audio):
    result = np.sqrt((audio ** 2).sum() / audio.shape[0])
    result_db = 10 * np.log(result)
    return result_db

def clip(path, vocal_path):
    name = os.path.basename(path).split('_')[0]
    #des_dir = os.path.dirname(path).replace('/renamed_mp3', '/clips_10s')
    des_dir = os.path.join(des_root, 'clips_10s')
    des_dir = os.path.join(des_dir, name)

    audio, sr = librosa.load(path, sr=None, mono=True) #sf.read(path)
    clip_len = sr * 10
    #audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    vocal, sr = librosa.load(vocal_path, sr=None, mono=True)
    #vocal_16k = librosa.resample(vocal, orig_sr=sr, target_sr=16000)

    num_clips = int(audio.shape[0] / clip_len)
    rms_list = []
    for i in range(num_clips):
        audio_clip = audio[i*clip_len: (i+1)*clip_len]
        vocal_clip = vocal[i*clip_len: (i+1)*clip_len]
        rms_audio = peak_RMS_amplitude(audio_clip)
        rms_vocal = peak_RMS_amplitude(vocal_clip)
        rms_list.append([rms_audio, rms_vocal])
        if rms_vocal < -25 or rms_audio < -25:
            continue

        if not os.path.exists(des_dir):
            os.makedirs(des_dir)

        sf.write(os.path.join(des_dir, '%03d_no_vocals.wav' % i), audio_clip, sr, subtype='PCM_24')
        sf.write(os.path.join(des_dir, '%03d_vocals.wav' % i), vocal_clip, sr, subtype='PCM_24')

i = 0
for path in tqdm(_list):
    name = os.path.basename(path)
    name = name.strip('_no_vocals.%s' % suffix)
    #new_path = os.path.join(des_root, '%06d_no_vocals.%s' % (i, suffix))
    new_dir = os.path.join(des_root, 'tmp')
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    new_path = os.path.join(new_dir, '%06d_no_vocals.%s' % (i, suffix))
    cmd1 = "cp '%s' %s" % (path, new_path)

    path2 = path.replace('_no_vocals.%s' % suffix, '_vocals.%s' % suffix)
    new_path2 = new_path.replace('_no_vocals.%s' % suffix, '_vocals.%s' % suffix)
    cmd2 = "cp '%s' %s" % (path2, new_path2)

    if (not os.path.exists(path)) or (not os.path.exists(path2)):
        continue
    try:
        os.system(cmd1)
        os.system(cmd2)
    except:
        print(cmd1)

    try:
        clip(new_path, new_path2)
    except:
        continue
    cmd = 'rm -rf %s ' % new_dir
    os.system(cmd)
    i = i + 1
