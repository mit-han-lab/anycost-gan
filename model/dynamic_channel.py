import random
from model.anycost_gan import G_CHANNEL_CONFIG, ConstantInput, ModulatedConv2d

CHANNEL_CONFIGS = [0.25, 0.5, 0.75, 1.0]


def get_full_channel_configs(model):
    full_channels = []
    for m in model.modules():
        if isinstance(m, ConstantInput):
            full_channels.append(m.input.shape[1])
        elif isinstance(m, ModulatedConv2d):
            if m.weight.shape[1] == 3 and m.weight.shape[-1] == 1:
                continue
            full_channels.append(m.weight.shape[1])  # get the output channels
    return full_channels


def set_sub_channel_config(model, sub_channels):
    ptr = 0
    for m in model.modules():
        if isinstance(m, ConstantInput):
            m.first_k_oup = sub_channels[ptr]
            ptr += 1
        elif isinstance(m, ModulatedConv2d):
            if m.weight.shape[1] == 3 and m.weight.shape[-1] == 1:
                continue
            m.first_k_oup = sub_channels[ptr]
            ptr += 1
    assert ptr == len(sub_channels), (ptr, len(sub_channels))  # all used


def remove_sub_channel_config(model):
    for m in model.modules():
        if hasattr(m, 'first_k_oup'):
            del m.first_k_oup


def reset_generator(model):
    remove_sub_channel_config(model)
    if hasattr(model, 'target_res'):
        del model.target_res


def get_current_channel_config(model):
    ch = []
    for m in model.modules():
        if hasattr(m, 'first_k_oup'):
            ch.append(m.first_k_oup)
    return ch


def sample_random_sub_channel(model, seed=None, set_channels=True):
    if seed is not None:  # whether to sync between workers
        random.seed(seed)

    rand_ratio = random.choice(CHANNEL_CONFIGS)
    if set_channels:
        set_uniform_channel_ratio(model, rand_ratio)
    return rand_ratio


def set_uniform_channel_ratio(model, ratio):
    full_channels = get_full_channel_configs(model)
    resolution = model.resolution
    org_channel_mult = full_channels[-1] * 1. / G_CHANNEL_CONFIG[resolution]

    channel_max = model.channel_max
    channels = {k: min(channel_max, int(v * ratio * org_channel_mult)) for k, v in G_CHANNEL_CONFIG.items()}
    channel_config = [v for k, v in channels.items() if k <= resolution]
    channel_config2 = []  # duplicate the config
    for c in channel_config:
        channel_config2.append(c)
        channel_config2.append(c)
    channel_config = channel_config2

    set_sub_channel_config(model, channel_config)

