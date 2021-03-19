import sys
sys.path.append('.')  # to run from the project root dir

import argparse
import pickle
import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm
import models


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


def extract_feature_from_samples():
    import math
    n_batch = math.ceil(args.n_sample * 1. / args.batch_size / hvd.size())
    features = None

    with torch.no_grad():
        for _ in tqdm(range(n_batch), disable=hvd.rank() != 0):
            latent = torch.randn(args.batch_size, 1, 512, device=device)
            img, _ = generator(latent)
            img = img.clamp(min=-1., max=1.)
            feat = inception(img)[0].view(img.shape[0], -1)  # the img will be automatically resized
            if features is None:
                features = feat.to('cpu')
            else:
                features = torch.cat((features, feat.to('cpu')), dim=0)

    return features


def compute_fid():
    pass


if __name__ == '__main__':
    import horovod.torch as hvd

    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, help='config name of the pretrained generator')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--inception', type=str, default=None, required=True)

    parser.add_argument('--channel_ratio', type=float, default=None)
    parser.add_argument('--target_res', type=int, default=None)

    args = parser.parse_args()

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    generator = models.get_pretrained('generator', args.config).to(device)
    generator.eval()

    # set sub-generator
    if args.channel_ratio:
        from models.dynamic_channel import set_uniform_channel_ratio, CHANNEL_CONFIGS

        assert args.channel_ratio in CHANNEL_CONFIGS
        set_uniform_channel_ratio(generator, args.channel_ratio)

    if args.target_res is not None:
        generator.target_res = args.target_res

    # compute the flops of the generator (is possible)
    if hvd.rank() == 0:
        try:
            from torchprofile import profile_macs

            macs = profile_macs(generator, torch.rand(1, 1, 512).to(device))
            params = sum([p.numel() for p in generator.parameters()])
            print(' * MACs: {:.2f}G, Params: {:.2f}M'.format(macs / 1e9, params / 1e6))
        except:
            print(' * Profiling failed. Passed.')

    inception = models.get_pretrained('inception').to(device)
    inception.eval()

    inception_features = extract_feature_from_samples()
    # now perform all gather
    inception_features = hvd.allgather(inception_features, name='inception_features').numpy()[:args.n_sample]

    if hvd.rank() == 0:
        print(f'extracted {inception_features.shape[0]} features')

    if hvd.rank() == 0:
        sample_mean = np.mean(inception_features, 0)
        sample_cov = np.cov(inception_features, rowvar=False)

        with open(args.inception, 'rb') as f:
            embeds = pickle.load(f)
            real_mean = embeds['mean']
            real_cov = embeds['cov']

        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
        print(args.inception)
        print('fid:', fid)
