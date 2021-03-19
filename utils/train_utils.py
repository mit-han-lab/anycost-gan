import torch
import torch.nn.functional as F
import random
import numpy as np
from models.dynamic_channel import CHANNEL_CONFIGS, sample_random_sub_channel

__all__ = ['requires_grad', 'accumulate', 'get_mixing_z', 'get_g_arch', 'adaptive_downsample256', 'get_teacher_multi_res',
           'get_random_g_arch', 'partially_load_d_for_multi_res', 'partially_load_d_for_ada_ch']


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_((1 - decay) * par2[k].data)


def get_mixing_z(batch_size, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return torch.randn(batch_size, 2, latent_dim, device=device)

    else:
        return torch.randn(batch_size, 1, latent_dim, device=device)


def get_g_arch(ratios, device='cuda'):
    out = []
    for r in ratios:
        one_hot = [0] * len(CHANNEL_CONFIGS)
        one_hot[CHANNEL_CONFIGS.index(r)] = 1
        out += one_hot
    return torch.from_numpy(np.array(out)).float().to(device)


def adaptive_downsample256(img):
    img = img.clamp(-1, 1)
    if img.shape[-1] > 256:
        return F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=True)
    else:
        return img


def get_teacher_multi_res(teacher_out, n_res):
    teacher_rgbs = [teacher_out]
    cur_res = teacher_out.shape[-1] // 2
    for _ in range(n_res - 1):
        # for simplicity, we use F.interpolate. Be sure to always use this.
        teacher_rgbs.insert(0, F.interpolate(teacher_out, size=cur_res, mode='bilinear',
                                             align_corners=True))
        cur_res = cur_res // 2
    return teacher_rgbs


def get_random_g_arch(generator, min_channel, divided_by, dynamic_channel_mode, seed=None):
    rand_ratio = sample_random_sub_channel(
        generator,
        min_channel=min_channel,
        divided_by=divided_by,
        seed=seed,
        mode=dynamic_channel_mode,
        set_channels=False
    )
    return get_g_arch(rand_ratio)


def partially_load_d_for_multi_res(d, sd, n_res=4):
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('convs.') and not k.startswith('convs.0.'):
            k_sp = k.split('.')
            k_sp[0] = 'blocks'
            k_sp[1] = str(int(k_sp[1]) - 1)
            new_sd['.'.join(k_sp)] = v
        else:
            new_sd[k] = v
    for i_res in range(1, n_res):  # just retain the weights
        new_sd['convs.{}.0.weight'.format(i_res)] = d.state_dict()['convs.{}.0.weight'.format(i_res)]
        new_sd['convs.{}.1.bias'.format(i_res)] = d.state_dict()['convs.{}.1.bias'.format(i_res)]
    d.load_state_dict(new_sd)


def partially_load_d_for_ada_ch(d, sd):
    # handling the new modulation FC
    blocks_with_mapping = []
    for k, v in d.state_dict().items():
        if '_mapping.' in k:
            sd[k] = v
            blocks_with_mapping.append('.'.join(k.split('.')[:2]))
    blocks_with_mapping = list(set(blocks_with_mapping))
    for blk in blocks_with_mapping:
        sd[blk + '.conv1.2.bias'] = sd.pop(blk + '.conv1.1.bias')
        sd[blk + '.conv2.3.bias'] = sd.pop(blk + '.conv2.2.bias')
    d.load_state_dict(sd)