import io
import random
import sqlite3
from functools import partial, wraps
from itertools import cycle
from pathlib import Path
import json, pickle

import numpy as np
import os
import torch
import torch.nn.functional as F
import torchaudio
from beartype.door import is_bearable
from beartype.typing import List, Literal, Optional, Tuple, Union
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchaudio.functional import resample
from torch.nn.utils.rnn import pad_sequence

from utils_d import (beartype_jit, curtail_to_multiple, default,
                    float32_to_int16, int16_to_float32,
                    zero_mean_unit_var_norm, exists)
#from F0_predictor.DioF0Predictor import DioF0Predictor

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

def pad_to_longest_fn(data):
    if isinstance(data, tuple) or isinstance(data, list):  # (bs, N, ...) -> (N, bs, ...)
        I, J = len(data), len(data[0])
        new_data = []
        for j in range(J):
            d = []
            for i in range(I):
                d.append(data[i][j])
            new_data.append(d)
        return tuple([pad_sequence(item, batch_first = True) for item in new_data])
    return pad_sequence(data, batch_first = True)

def get_dataloader2(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)

def get_dataloader(ds, **kwargs):
    return DataLoader(ds, **kwargs)

def extract_F0(file_path):
    y, sr = librosa.load(file_path, sr=None)
    frame_length = int(2048 / 22050 * sr)  # 93*3 ms

    #model = DioF0Predictor(sampling_rate=sr, hop_length=256, compute_mixed=False)
    #f0, _ = model.compute_mixed_f0(y, p_len=None)

    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=frame_length)
    f0 = f0.tolist()
    f0_float = [item if not math.isnan(item) else 0 for item in f0]
    f0_int = [round(item) if not math.isnan(item) else 0 for item in f0]
    f0_note = [librosa.hz_to_note(item) if not math.isnan(item) else "NoneNote" for item in f0]
    #f0 = [librosa.note_to_hz(item) if not item == "NoneNote" else 0 for item in f0]
    return f0_int

class SoundDataset(Dataset):
    def __init__(
        self,
        folder,
        exts = ['flac', 'wav', 'mp3'],
        max_length_seconds = 1,
        data_key=None,
        mixed_F0=None,
        normalize = False,
        target_sample_hz = None,
        seq_len_multiple_of = None,
        ignore_files = None,
        ignore_load_errors=True,
        random_crop=True,
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        #self.mixed_F_extractor = DioF0Predictor(sampling_rate=24000, hop_length=256, compute_mixed=True)

        files = []
        ignore_files = default(ignore_files, [])
        num_ignored = 0
        ignore_file_set = set([f.split('/')[-1] for f in ignore_files])
        for ext in exts:
            for file in path.glob(f'**/*.{ext}'):
                if file.name in ignore_file_set:
                    num_ignored += 1
                    #continue
                else:
                    file = str(file)
                    if '_no_vocals.' in file:
                        _dict = {}
                        _dict['non_vocal'] = file
                        _dict['vocal'] = file.replace('_no_vocals.', '_vocals.')
                        #_dict['vocal'] = file
                        #_dict['non_vocal'] = file.replace('_no_vocals.', '_vocals.')
                        files.append(_dict)
        assert len(files) > 0, 'no sound files found'
        print(len(files), '!!!!')
        if num_ignored > 0:
            print(f'skipped {num_ignored} ignored files')

        self.files = files
        self.ignore_load_errors = ignore_load_errors
        self.random_crop = random_crop

        self.target_sample_hz = cast_tuple(target_sample_hz)
        num_outputs = len(self.target_sample_hz)

        self.max_length_seconds = max_length_seconds
        max_length_seconds_tuple = cast_tuple(max_length_seconds, num_outputs)
        self.max_length = tuple([int(s * hz) if exists(s) else None for s, hz in zip(max_length_seconds_tuple, self.target_sample_hz)])
        self.data_key = data_key
        self.mixed_F0 = mixed_F0

        self.normalize = cast_tuple(normalize, num_outputs)

        self.seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

        assert len(self.max_length) == len(max_length_seconds_tuple) == len(
            self.target_sample_hz) == len(self.seq_len_multiple_of) == len(self.normalize)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file = self.files[idx]
            #data, sample_hz = torchaudio.load(file)
        except:
            if self.ignore_load_errors:
                return self[torch.randint(0, len(self), (1,)).item()]
            else:
                raise Exception(f'error loading file {file}')

        return self.process_audio(file, pad_to_target_length=True)

    def process_audio(self, file, pad_to_target_length=True):

        # recursively crop the audio at random in the order of longest to shortest max_length_seconds, padding when necessary.
        # e.g. if max_length_seconds = (10, 4), pick a 10 second crop from the original, then pick a 4 second crop from the 10 second crop
        # also use normalized data when specified

        num_outputs = len(self.target_sample_hz)
        datas = [None for _ in range(num_outputs)]

        data_key = [random.choice(item) if isinstance(item, list) else item for item in self.data_key]

        non_vocal_path = file['non_vocal']
        f_path = non_vocal_path.replace('/clips_10s/', '/clips_feature2/').replace('_no_vocals.wav', '.npy')
        load_feature = True if os.path.exists(f_path) else False
        if load_feature:
            features = np.load(f_path, allow_pickle=True)
            f = {}
            f['vocal'], f['non_vocal'] = features[0], features[1]

        paths = [file[k] for k in data_key]
        start = None
        for i, path in enumerate(paths):
            data, sample_hz = torchaudio.load(path)
            if data.shape[0] > 1:
                data = torch.mean(data, dim=0).unsqueeze(0)

            
            ### add noise
            name = random.randint(0, 1000)
            if data_key[i] == 'vocal':
                stdv = 0.01
                noise = torch.randn_like(data) * stdv
                data = data + noise
            

            
            ### data augmentation
            if random.random() < 0.3 and data_key[i] == 'vocal':
                snr = random.random() / 10
                try:
                    stdv = torch.sqrt((data ** 2).max()) * snr
                except:
                    stdv = 0.01
                    print(path)
                noise = torch.randn_like(data) * stdv
                data = data + noise
            

            temp_data = data
            temp_data_normalized = zero_mean_unit_var_norm(data)
            audio_length = temp_data.size(1)
            target_length = int(self.max_length_seconds * sample_hz)

            if audio_length > target_length:
                max_start = audio_length - target_length
                if start is None:
                    start = torch.randint(0, max_start, (1, )) if self.random_crop else 0
                temp_data = temp_data[:, start:start + target_length]
                temp_data_normalized = temp_data_normalized[:, start:start + target_length]
            else:
                if pad_to_target_length:
                    temp_data = F.pad(temp_data, (0, target_length - audio_length), 'constant')
                    temp_data_normalized = F.pad(temp_data_normalized, (0, target_length - audio_length), 'constant')
            datas[i] = temp_data_normalized if self.normalize[i] else temp_data

        # resample if target_sample_hz is not None in the tuple
        data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in zip(datas, self.target_sample_hz))
        # quantize non-normalized audio to a valid waveform
        data_tuple = tuple(d if self.normalize[i] else int16_to_float32(float32_to_int16(d)) for i, d in enumerate(data_tuple))

        output = []
        for i, data in enumerate(data_tuple):
            data = data.squeeze(0)
            _dict = {}
            _dict['wav'] = data
            if load_feature and self.mixed_F0[i] == True:
                _dict['mel'] = f[data_key[i]]['mel']
                _dict['f0'] = f[data_key[i]]['f0']
                _dict['mixed_F'] = f[data_key[i]]['mixed_F']

            vocal_flag = (data_key[i] == 'vocal') and random.random() < 0.3
            if (not load_feature or vocal_flag) and self.mixed_F0[i] == True:
                f0, mixed_F = self.mixed_F_extractor.compute_mixed_f0(data.numpy())
                _dict['f0'] = f0
                _dict['mixed_F'] = mixed_F

            output.append(_dict)
        return output

        output = []

        # process each of the data resample at different frequencies individually

        for data, max_length, seq_len_multiple_of in zip(data_tuple, self.max_length, self.seq_len_multiple_of):
            audio_length = data.size(1)

            if exists(max_length) and pad_to_target_length:
                assert audio_length == max_length, f'audio length {audio_length} does not match max_length {max_length}.'

            data = rearrange(data, '1 ... -> ...')

            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]
        return output

    def process_audio2(self, file, sample_hz, data_key, pad_to_target_length=True):

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = torch.mean(data, dim=0).unsqueeze(0)

        # recursively crop the audio at random in the order of longest to shortest max_length_seconds, padding when necessary.
        # e.g. if max_length_seconds = (10, 4), pick a 10 second crop from the original, then pick a 4 second crop from the 10 second crop
        # also use normalized data when specified

        temp_data = data
        temp_data_normalized = zero_mean_unit_var_norm(data)

        num_outputs = len(self.target_sample_hz)
        data = [None for _ in range(num_outputs)]

        sorted_max_length_seconds = sorted(
            enumerate(self.max_length_seconds),
            key=lambda t: (t[1] is not None, t[1])) # sort by max_length_seconds, while moving None to the beginning

        for unsorted_i, max_length_seconds in sorted_max_length_seconds:

            if exists(max_length_seconds):
                audio_length = temp_data.size(1)
                target_length = int(max_length_seconds * sample_hz)

                if audio_length > target_length:
                    max_start = audio_length - target_length
                    start = torch.randint(0, max_start, (1, )) if self.random_crop else 0

                    temp_data = temp_data[:, start:start + target_length]
                    temp_data_normalized = temp_data_normalized[:, start:start + target_length]
                else:
                    if pad_to_target_length:
                        temp_data = F.pad(temp_data, (0, target_length - audio_length), 'constant')
                        temp_data_normalized = F.pad(temp_data_normalized, (0, target_length - audio_length), 'constant')

            data[unsorted_i] = temp_data_normalized if self.normalize[unsorted_i] else temp_data
        # resample if target_sample_hz is not None in the tuple
        data_tuple = tuple((resample(d, sample_hz, target_sample_hz) if exists(target_sample_hz) else d) for d, target_sample_hz in zip(data, self.target_sample_hz))
        # quantize non-normalized audio to a valid waveform
        data_tuple = tuple(d if self.normalize[i] else int16_to_float32(float32_to_int16(d)) for i, d in enumerate(data_tuple))

        output = []

        # process each of the data resample at different frequencies individually

        for data, max_length, seq_len_multiple_of in zip(data_tuple, self.max_length, self.seq_len_multiple_of):
            audio_length = data.size(1)

            if exists(max_length) and pad_to_target_length:
                assert audio_length == max_length, f'audio length {audio_length} does not match max_length {max_length}.'

            data = rearrange(data, '1 ... -> ...')

            if exists(seq_len_multiple_of):
                data = curtail_to_multiple(data, seq_len_multiple_of)

            output.append(data.float())

        # cast from list to tuple

        output = tuple(output)

        # return only one audio, if only one target resample freq

        if num_outputs == 1:
            return output[0]
        return output


