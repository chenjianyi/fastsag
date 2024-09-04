import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from beartype import beartype
from pathlib import Path
import shutil
import os
from torchaudio.functional import resample

from einops import rearrange, repeat, reduce

spec_min=-12
spec_max=2
def norm_spec(x):
    return (x -  spec_min) / ( spec_max -  spec_min) * 2 - 1
def denorm_spec(x):
    return (x + 1) / 2 * ( spec_max -  spec_min) +  spec_min

def beartype_jit(func):
    """decorator to enable beartype only if USE_BEARTYPE is set to 1"""
    return beartype(func) if os.environ.get('USE_BEARTYPE', '0') == '1' else func


@beartype_jit
def get_or_compute_clap_token_ids1(clap_token_ids, clap, conditioning_audio, conditioning_text):
    if not exists(clap_token_ids):
        assert exists(conditioning_audio) ^ exists(conditioning_text), "either condition on text or audio"
        assert exists(clap)
        clap.eval()
        if exists(conditioning_text):
            with torch.no_grad():
                clap_token_ids = clap(text_input=conditioning_text)
        else:
            with torch.no_grad():
                clap_token_ids = clap(audio_input=conditioning_audio)

    return clap_token_ids

@beartype_jit
def get_or_compute_clap_token_ids(clap_token_ids, clap, conditioning_v_audio, conditioning_nv_audio, embedding_key='fine_grained_embedding'):
    if not exists(clap_token_ids):
        assert exists(conditioning_v_audio) ^ exists(conditioning_nv_audio), "either condition on v_audio or nv_audio"
        assert exists(clap)
        clap.eval()
        if exists(conditioning_v_audio):
            with torch.no_grad():
                clap_token_ids = clap(v_audio_input=conditioning_v_audio, embedding_key=embedding_key)
        else:
            with torch.no_grad():
                clap_token_ids = clap(nv_audio_input=conditioning_nv_audio, embedding_key=embedding_key)

    return clap_token_ids


@beartype_jit
def get_or_compute_semantic_token_ids(semantic_token_ids, raw_audio, wav2vec):
    if not exists(semantic_token_ids):
        assert exists(raw_audio)
        assert exists(wav2vec)
        with torch.no_grad():
            wav2vec.eval()
            semantic_token_ids = wav2vec(raw_audio, flatten=False)

    return semantic_token_ids


@beartype_jit
def get_or_compute_acoustic_token_ids(coarse_token_ids, fine_token_ids, raw_audio, neural_codec, num_coarse_quantizers: int):

    if exists(raw_audio):
        assert not exists(coarse_token_ids) and not exists(fine_token_ids), "either provide coarse + fine ids or raw audio"
        assert exists(neural_codec), 'A neural audio codec must be provided if given raw wave for training'

        with torch.no_grad():
            neural_codec.eval()
            _, indices, _ = neural_codec(raw_audio, return_encoded=True)
            coarse_token_ids, fine_token_ids = indices[..., :num_coarse_quantizers], indices[..., num_coarse_quantizers:]

    return coarse_token_ids, fine_token_ids


def load_model(model, path):
    """helper class to load a model checkpoint"""
    path = Path(path)
    assert path.exists(), f'checkpoint does not exist at {str(path)}'
    pkg = torch.load(str(path))
    model.load_state_dict(pkg)

# helper functions

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ceil_div(numer, denom):
    return (numer + denom - 1) // denom

def remainder_needed_until_multiple(n, mult):
    return (ceil_div(n, mult) * mult) - n

def round_down_nearest_multiple(val, mult):
    return (val // mult) * mult

def curtail_to_multiple(t, mult):
    data_len = t.shape[-1]
    return t[..., :round_down_nearest_multiple(data_len, mult)]

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# tensor helpers

def generate_mask_with_prob(shape, mask_prob, device):
    seq = shape[-1]
    rand = torch.randn(shape, device = device)
    rand[:, 0] = -torch.finfo(rand.dtype).max
    num_mask = min(int(seq * mask_prob), seq - 1)
    indices = rand.topk(num_mask, dim = -1).indices
    mask = ~torch.zeros(shape, device = device).scatter(1, indices, 1.).bool()
    return mask

# attention related utils

def grad_shrink(t, alpha = 0.1):
    return t * alpha + t.detach() * (1 - alpha)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def mask_out_after_eos_id(t, eos_id, mask_value = -1, keep_eos = True):
    eos_mask = (t == eos_id).float()

    if keep_eos:
        eos_mask = F.pad(eos_mask, (1, -1))

    after_eos_mask = eos_mask.cumsum(dim = -1) > 0
    return t.masked_fill(after_eos_mask, mask_value)

def all_rows_have_eos_id(t, eos_id):
    eos_mask = (t == eos_id)
    return torch.any(eos_mask, dim = -1).all()

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# removing unique consecutives in the semantic token ids
# important detail noted by @eonglints

def append_eos_id(ids, eos_id):
    b, device = ids.shape[0], ids.device
    eos_ids = torch.ones(1, device = device).long() * eos_id
    eos_ids = repeat(eos_ids, '1 -> b 1', b = b)
    ids = torch.cat((ids, eos_ids), dim = -1)
    return ids

def batch_unique_consecutive(t, pad_value = 0.):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)

# to get embedding from sequence with padding token

@beartype_jit
def get_embeds(
    embeddings: nn.Embedding,
    codes: torch.Tensor,
    pad_id = -1,
    return_mask = False,
    mask_pad_pos_to = 0
):
    pad_mask = codes == pad_id
    codes_without_pad = codes.masked_fill(pad_mask, 0) # just retrieve first code as dummy
    embeds = embeddings(codes_without_pad)

    if exists(mask_pad_pos_to):
        embeds = embeds.masked_fill(rearrange(pad_mask, '... -> ... 1'), mask_pad_pos_to)

    if return_mask:
        return embeds, ~pad_mask

    return embeds

# audio processing helpers

def int16_to_float32(x):
    return (x / 32767.0).type(torch.float32)

def float32_to_int16(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

def zero_mean_unit_var_norm2(x):
    return (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-7)
    #return (x - x.mean(dim=-1, keepdim=True))

def zero_mean_unit_var_norm(x):
    a, b = -0.8, 0.8
    k = (b - a) / (x.max() - x.min())
    x = a + k * (x - x.min())
    return x

def prepare_audio(data, sample_hz, target_sample_hz, normalize=True, target_length_seconds=None):
    if data.shape[0] > 1:
        data = torch.mean(data, dim=0).unsqueeze(0)
    if normalize:
        data = zero_mean_unit_var_norm(data)
    if exists(target_length_seconds) and data.shape[1] > target_length_seconds * sample_hz:
        data = data[: , :int(target_length_seconds * sample_hz)]
    audio_for_wav2vec = resample(data, sample_hz, target_sample_hz)
    audio_for_wav2vec = int16_to_float32(float32_to_int16(audio_for_wav2vec))
    return audio_for_wav2vec

# helper for saving config

def copy_file_to_folder(file_path: str, folder_path: str):
    config_file = Path(file_path)
    folder = Path(folder_path)

    shutil.copy(str(config_file), str(folder / config_file.name))

class disable_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_latest_checkpoints(results_folder, max_step=None):
    highest_transformer_step = -1
    highest_optimizer_step = -1
    highest_scheduler_step = -1
    transformer_path = None
    optimizer_path = None
    scheduler_path = None
    max_step = float('inf') if max_step is None else max_step
    for file in os.listdir(results_folder):
        if file.endswith('.pt'):
            if 'transformer' in file:
                step = int(file.split('.')[2])
                if step > highest_transformer_step and step <= max_step:
                    highest_transformer_step = step
                    transformer_path = os.path.join(results_folder, file)
            elif 'optimizer' in file:
                step = int(file.split('.')[2])
                if step > highest_optimizer_step and step <= max_step:
                    highest_optimizer_step = step
                    optimizer_path = os.path.join(results_folder, file)
            elif 'scheduler' in file:
                step = int(file.split('.')[2])
                if step > highest_scheduler_step and step <= max_step:
                    highest_scheduler_step = step
                    scheduler_path = os.path.join(results_folder, file)

    assert highest_transformer_step == highest_optimizer_step, 'transformer and optimizer checkpoints are not aligned'
    if scheduler_path is not None:
        assert highest_transformer_step == highest_scheduler_step, 'transformer and scheduler checkpoints are not aligned'

    return (transformer_path, optimizer_path, scheduler_path), highest_transformer_step

def validate_train_args(args):
    assert not(exists(args.fine_tune_from) and exists(args.continue_from_dir)), 'choose one: fine tune from a checkpoint or continue from a directory'

    print(f'saving results to {args.results_folder}, using model config {args.model_config} and training config {args.training_config}, using rvq checkpoint {args.rvq_path} and kmeans checkpoint {args.kmeans_path}')
    if exists(args.continue_from_dir):
        print(f'continuing from latest checkpoint in {args.continue_from_dir}')
        assert not Path(args.continue_from_dir) == Path(args.results_folder), 'continue_from_dir must be different from results_folder'
    elif exists(args.fine_tune_from):
        print(f'fine tuning from checkpoint {args.fine_tune_from}. Make sure to use the same model config as the base model.')

def load_checkpoint_from_args(trainer, args):
    if exists(args.continue_from_dir):
        checkpoints, steps = get_latest_checkpoints(args.continue_from_dir, args.continue_from_step)
        print(f'loading checkpoints: {checkpoints}')
        trainer.load(*checkpoints, steps=steps+1)

def exists(val):
    return val is not None

class disable_print:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
