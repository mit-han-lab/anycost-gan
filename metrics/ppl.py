import sys

sys.path.append('.')  # to run from the project root dir

import argparse
import math
import torch
import numpy as np
from tqdm import tqdm
import lpips
import models
import horovod.torch as hvd


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


def compute_ppl(g, n_sample, batch_size, space='w', sampling='end', eps=1e-4, crop=False):
    percept = lpips.LPIPS(net='vgg', verbose=False).to(device)

    distances = []

    n_batch = math.ceil(n_sample * 1. / batch_size / hvd.size())

    with torch.no_grad():
        for _ in tqdm(range(n_batch), disable=hvd.rank() != 0):
            noise = g.make_noise()

            inputs = torch.randn([batch_size * 2, g.style_dim], device=device)
            if sampling == "full":
                lerp_t = torch.rand(batch_size, device=device)
            elif sampling == "end":
                lerp_t = torch.zeros(batch_size, device=device)
            else:
                raise NotImplementedError

            if space == "w":
                latent = g.get_style(inputs)
                latent_t0, latent_t1 = latent[::2], latent[1::2]
                latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
                latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + eps)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

            latent_e = latent_e.unsqueeze(1)
            image, _ = g(latent_e, input_is_style=True, noise=noise)
            image = image.clamp(-1, 1)

            if crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3: c * 7, c * 2: c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                # following tf official implementation, we do not use F.interpolate
                # it will make a difference for 1024 resolution.
                n, c, h, w = image.shape
                image = image.view(n, c, h // factor, factor, w // factor, factor).mean([3, 5])

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (eps ** 2)
            distances.append(dist.to("cpu"))

    distances = torch.cat(distances, 0)
    # all gather
    distances = hvd.allgather(distances, name='distances').numpy()[:n_sample]

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )
    return filtered_dist.mean()


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Perceptual Path Length calculator")
    parser.add_argument("--config", type=str, help='config name of the pretrained generator')
    parser.add_argument('--channel_ratio', type=float, default=None)
    parser.add_argument('--target_res', type=int, default=None)

    parser.add_argument("--n_sample", type=int, default=100000, help="number of the samples for calculating PPL")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for the models (per gpu)")
    parser.add_argument("--space", default='w', choices=["z", "w"], help="space that PPL calculated with")
    parser.add_argument("--eps", type=float, default=1e-4, help="epsilon for numerical stability")
    parser.add_argument("--crop", action="store_true", help="apply center crop to the images")
    parser.add_argument("--sampling", default="end", choices=["end", "full"], help="set endpoint sampling method")

    args = parser.parse_args()

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    generator = models.get_pretrained('generator', args.config).to(device)
    generator.eval()

    if args.channel_ratio:
        from models.dynamic_channel import set_uniform_channel_ratio
        set_uniform_channel_ratio(generator, args.channel_ratio)

    if args.target_res is not None:
        generator.target_res = args.target_res

    ppl = compute_ppl(generator,
                      n_sample=args.n_sample,
                      batch_size=args.batch_size,
                      space=args.space,
                      sampling=args.sampling,
                      eps=args.eps,
                      crop=args.crop)
    if hvd.rank() == 0:
        print(' * PPL: {}'.format(ppl))
