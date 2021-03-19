import argparse
import os
import json
import random
import math
import time
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import lpips

import horovod.torch as hvd

import sys
sys.path.append('.')  # to run from the project root dir
from utils.datasets import NativeDataset
from utils.losses import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize
from torch.utils.tensorboard import SummaryWriter
import models
from models.anycost_gan import Generator, Discriminator, DiscriminatorMultiRes
from models.dynamic_channel import sample_random_sub_channel, set_uniform_channel_ratio, reset_generator, sort_channel
from metrics.fid import calc_fid

from utils.torch_utils import DistributedMeter
from utils.train_utils import *

device = 'cuda'
log_dir = 'log'
checkpoint_dir = 'checkpoint'

best_fid = 1e9


def train(epoch):
    generator.train()
    discriminator.train()
    g_ema.eval()
    sampler.set_epoch(epoch)

    with tqdm(total=len(data_loader),
              desc='Epoch #{}'.format(epoch + 1),
              disable=hvd.rank() != 0, dynamic_ncols=True) as t:
        global mean_path_length  # track across epochs

        ema_decay = 0.5 ** (args.batch_size * hvd.size() / (args.half_life_kimg * 1000.))

        # loss meters
        d_loss_meter = DistributedMeter('d_loss')
        r1_loss_meter = DistributedMeter('r1_loss')
        g_loss_meter = DistributedMeter('g_loss')
        path_loss_meter = DistributedMeter('path_loss')
        d_real_acc = DistributedMeter('d_real_acc')
        d_fake_acc = DistributedMeter('d_fake_acc')
        distill_loss_meter = DistributedMeter('distill_loss')

        for batch_idx, real_img in enumerate(data_loader):
            global_idx = batch_idx + epoch * len(data_loader) + 1
            if args.n_res > 1:
                real_img = [ri.to(device) for ri in real_img]  # a stack of images
            else:
                real_img = real_img.to(device)

            # 1. train D
            requires_grad(generator, False)
            requires_grad(discriminator, True)

            z = get_mixing_z(args.batch_size, args.latent_dim, args.mixing_prob, device)
            with torch.no_grad():
                if args.dynamic_channel:
                    rand_ratio = sample_random_sub_channel(
                        generator, min_channel=args.min_channel,
                        divided_by=args.divided_by,
                        mode=args.dynamic_channel_mode,
                    )
                fake_img, all_rgbs = generator(z, return_rgbs=True)
                all_rgbs = all_rgbs[-args.n_res:]
                reset_generator(generator)

            if args.n_res > 1:
                sampled_res = random.sample(all_resolutions, args.n_sampled_res)
                d_loss = 0.
                g_arch = get_g_arch(rand_ratio) if args.conditioned_d else None
                rand_g_arch = get_random_g_arch(  # randomly draw one for real images
                    generator, args.min_channel, args.divided_by, args.dynamic_channel_mode
                ) if args.conditioned_d else None
                for ri, fi in zip(real_img, all_rgbs):
                    if ri.shape[-1] in sampled_res:
                        real_pred = discriminator(ri, rand_g_arch)
                        fake_pred = discriminator(fi, g_arch)
                        d_loss += d_logistic_loss(real_pred, fake_pred)
            else:
                assert not args.conditioned_d  # not implemented yet
                fake_pred = discriminator(fake_img)
                real_pred = discriminator(real_img)
                d_loss = d_logistic_loss(real_pred, fake_pred)

            d_real_acc.update((real_pred > 0).sum() * 1. / real_pred.shape[0])
            d_fake_acc.update((fake_pred < 0).sum() * 1. / real_pred.shape[0])
            d_loss_meter.update(d_loss)

            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

            # reg D
            if args.d_reg_every > 0 and global_idx % args.d_reg_every == 0:
                reg_img = random.choice(real_img) if args.n_res > 1 else real_img
                reg_img.requires_grad = True

                if args.conditioned_d:
                    real_pred = discriminator(reg_img, g_arch)
                else:
                    real_pred = discriminator(reg_img)
                r1_loss = d_r1_loss(real_pred, reg_img)

                discriminator.zero_grad()
                (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
                d_optim.step()
                r1_loss_meter.update(r1_loss)

            # 2. train G
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            z = get_mixing_z(args.batch_size, args.latent_dim, args.mixing_prob, device)
            # fix the randomness (potentially apply distillation)
            noises = generator.make_noise()
            inject_index = None if z.shape[1] == 1 else random.randint(1, generator.n_style - 1)

            if args.dynamic_channel:
                rand_ratio = sample_random_sub_channel(generator, min_channel=args.min_channel,
                                                       divided_by=args.divided_by,
                                                       mode=args.dynamic_channel_mode)
            fake_img, all_rgbs = generator(z, noise=noises, inject_index=inject_index, return_rgbs=True)
            all_rgbs = all_rgbs[-args.n_res:]
            reset_generator(generator)

            # g loss
            if args.n_res > 1:
                sampled_rgbs = random.sample(all_rgbs, args.n_sampled_res)
                g_arch = get_g_arch(rand_ratio) if args.conditioned_d else None
                g_loss = sum([g_nonsaturating_loss(discriminator(r, g_arch)) for r in sampled_rgbs])
            else:
                g_loss = g_nonsaturating_loss(discriminator(fake_img))

            # distill loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_out, _ = teacher(z, noise=noises, inject_index=inject_index)
                teacher_rgbs = get_teacher_multi_res(teacher_out, args.n_res)
                distill_loss1 = sum([nn.MSELoss()(sr, tr) for sr, tr in zip(all_rgbs, teacher_rgbs)])
                distill_loss2 = sum([percept(adaptive_downsample256(sr), adaptive_downsample256(tr)).mean()
                                     for sr, tr in zip(all_rgbs, teacher_rgbs)])
                distill_loss = distill_loss1 + distill_loss2
                g_loss = g_loss + distill_loss * args.distill_loss_alpha
                distill_loss_meter.update(distill_loss * args.distill_loss_alpha)

            g_loss_meter.update(g_loss)

            generator.zero_grad()
            g_loss.backward()
            g_optim.step()

            # reg G
            if args.g_reg_every > 0 and global_idx % args.g_reg_every == 0:  # path len reg
                assert args.n_res == 1  # currently, we do not apply path reg after the original StyleGAN training
                path_batch_size = max(1, args.batch_size // args.path_batch_shrink)
                noise = get_mixing_z(path_batch_size, args.latent_dim, args.mixing_prob, device)
                fake_img, latents = generator(noise, return_styles=True)
                # moving update the mean path length
                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    fake_img, latents, mean_path_length
                )
                generator.zero_grad()
                weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
                # special trick to trigger sync gradient descent TODO: do we need it here?
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]
                weighted_path_loss.backward()
                g_optim.step()
                mean_path_length = hvd.allreduce(torch.Tensor([mean_path_length])).item()  # update across gpus
                path_loss_meter.update(path_loss)

            # moving update
            accumulate(g_ema, generator, ema_decay)

            info2display = {
                'd': d_loss_meter.avg.item(),
                'g': g_loss_meter.avg.item(),
                'r1': r1_loss_meter.avg.item(),
                'd_real_acc': d_real_acc.avg.item(),
                'd_fake_acc': d_fake_acc.avg.item()
            }
            if teacher is not None:
                info2display['dist'] = distill_loss_meter.avg.item()
            if args.g_reg_every > 0:
                info2display['path'] = path_loss_meter.avg.item()
                info2display['path-len'] = mean_path_length

            t.set_postfix(info2display)
            t.update(1)

            if hvd.rank() == 0 and global_idx % args.log_every == 0:
                n_trained_images = global_idx * args.batch_size * hvd.size()
                log_writer.add_scalar('Loss/D', d_loss_meter.avg.item(), n_trained_images)
                log_writer.add_scalar('Loss/G', g_loss_meter.avg.item(), n_trained_images)
                log_writer.add_scalar('Loss/r1', r1_loss_meter.avg.item(), n_trained_images)
                log_writer.add_scalar('Loss/path', path_loss_meter.avg.item(), n_trained_images)
                log_writer.add_scalar('Loss/path-len', mean_path_length, n_trained_images)
                log_writer.add_scalar('Loss/distill', distill_loss_meter.avg.item(), n_trained_images)

            if hvd.rank() == 0 and global_idx % args.log_vis_every == 0:  # log image
                with torch.no_grad():
                    g_ema.eval()
                    mean_style = g_ema.mean_style(10000)
                    sample, _ = g_ema(sample_z, truncation=args.vis_truncation, truncation_style=mean_style)
                    n_trained_images = global_idx * args.batch_size * hvd.size()
                    grid = utils.make_grid(sample, nrow=int(args.n_vis_sample ** 0.5), normalize=True,
                                           range=(-1, 1))
                    log_writer.add_image('images', grid, n_trained_images)


def validate(epoch):
    if args.dynamic_channel:  # we also evaluate the model with half channels
        set_uniform_channel_ratio(g_ema, 0.5)
        fid = measure_fid()
        reset_generator(g_ema)
        if hvd.rank() == 0:
            print(' * FID-0.5x: {:.2f}'.format(fid))
            log_writer.add_scalar('Metrics/fid-0.5x', fid,
                                  len(data_loader) * (epoch + 1) * args.batch_size * hvd.size())

    fid = measure_fid()
    if hvd.rank() == 0:
        log_writer.add_scalar('Metrics/fid', fid, len(data_loader) * (epoch + 1) * args.batch_size * hvd.size())
    global best_fid
    best_fid = min(best_fid, fid)
    if hvd.rank() == 0:
        print(' * FID: {:.2f} ({:.2f})'.format(fid, best_fid))
        state_dict = {
            "g": generator.state_dict(),
            "d": discriminator.state_dict(),
            "g_ema": g_ema.state_dict(),
            "g_optim": g_optim.state_dict(),
            "d_optim": d_optim.state_dict(),
            "epoch": epoch + 1,
            "fid": fid,
            "best_fid": best_fid,
            "mean_path_length": mean_path_length,
        }
        torch.save(state_dict, os.path.join(checkpoint_dir, args.job, 'ckpt.pt'))
        if best_fid == fid:
            torch.save(state_dict, os.path.join(checkpoint_dir, args.job, 'ckpt-best.pt'))


def measure_fid():
    # get all the resolutions to evaluate fid
    # WARNING: always assume that we measure the largest few resolutions
    n_res_to_eval = len(args.inception_path.split(','))  # support evaluating multiple resolutions
    inception_path_list = args.inception_path.split(',')

    n_batch = math.ceil(args.fid_n_sample / (args.fid_batch_size * hvd.size()))

    features_list = [None for _ in range(n_res_to_eval)]
    torch.manual_seed(int(time.time()) + hvd.rank() * 999)  # just make sure they use different seed
    # collect features
    with torch.no_grad():
        for _ in tqdm(range(n_batch), desc='FID', disable=hvd.rank() != 0):
            z = torch.randn(args.fid_batch_size, 1, 512, device=device)
            out, all_rgbs = g_ema(z, return_rgbs=True)
            for i_res in range(n_res_to_eval):
                img = all_rgbs[-i_res - 1]
                feat = inception(img.clamp(-1., 1.))[0].view(img.shape[0], -1).to('cpu')
                if features_list[i_res] is None:
                    features_list[i_res] = feat
                else:
                    features_list[i_res] = torch.cat((features_list[i_res], feat), dim=0)
    # compute the FID
    fid_dict = dict()
    for i_res, features in enumerate(features_list):
        features = hvd.allgather(features, name='fid_features{}'.format(i_res)).numpy()
        features = features[:args.fid_n_sample]
        if hvd.rank() == 0:  # only compute on node 1, save some CPU
            sample_mean = np.mean(features, 0)
            sample_cov = np.cov(features, rowvar=False)
            with open(inception_path_list[i_res], 'rb') as f:
                embeds = pickle.load(f)
            real_mean = embeds['mean']
            real_cov = embeds['cov']
            fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
            fid_dict[int(args.resolution / 2 ** i_res)] = fid
        else:
            fid_dict[int(args.resolution / 2 ** i_res)] = 1e9
    if hvd.rank() == 0:
        print('fid:', {k: round(v, 3) for k, v in fid_dict.items()})
    fid0 = hvd.broadcast(torch.tensor(fid_dict[args.resolution]).float(), root_rank=0, name='fid').item()
    return fid0  # only return the fid of the largest resolution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment setting
    parser.add_argument("--job", type=str, default=None, help='name of the run to help keep track')
    parser.add_argument("--data_path", type=str, default='/dataset/ffhq/images-wrap')
    parser.add_argument("--epochs", type=int, default=357)  # 25M images
    parser.add_argument("--batch_size", type=int, default=16, help='batch size per GPU')
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2, help='weight of the path reg')
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=8)
    parser.add_argument("--mixing_prob", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.002, help='the global learning rate')
    parser.add_argument('-j', '--workers', default=2, type=int, help='num workers per GPU')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument("--half_life_kimg", type=float, default=10.,
                        help='Half-life of the running average of generator weights')
    parser.add_argument("--tune_from", type=str, default=None)
    # fid setting
    parser.add_argument("--inception_path", type=str, default=None)
    parser.add_argument('--fid_n_sample', type=int, default=50000)
    parser.add_argument("--fid_batch_size", type=int, default=16)
    # log and visualization
    parser.add_argument("--n_vis_sample", type=int, default=16, help='n samples for visualization')
    parser.add_argument("--log_every", type=int, default=100, help='log training loss every')
    parser.add_argument("--log_vis_every", type=int, default=1000, help='log visualization every')
    parser.add_argument('--vis_truncation', type=float, default=0.5)
    # models setting
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--channel_multiplier", type=float, default=2)
    parser.add_argument("--latent_dim", type=int, default=512, help='dimension of the z/w')
    parser.add_argument("--n_mlp", type=int, default=8, help='z to w mapping')
    # distill teacher setting
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--t_channel_multiplier", type=float, default=2)
    parser.add_argument('--distill_loss_alpha', type=float, default=2.)
    # multi-res training
    parser.add_argument('--n_res', type=int, default=1, help='number of resolutions to support for training')
    parser.add_argument('--n_sampled_res', type=int, default=1, help='number of resolutions to sample per iter')
    # adaptive channel training
    parser.add_argument('--dynamic_channel', action='store_true', default=False)
    parser.add_argument('--dynamic_channel_mode', type=str, default='uniform')
    parser.add_argument('--sort_pretrain', action='store_true', default=False)
    parser.add_argument('--conditioned_d', action='store_true', default=False, help='D is conditioned on G')
    parser.add_argument('--min_channel', type=int, default=8)
    parser.add_argument('--divided_by', type=int, default=4)

    args = parser.parse_args()

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    # save memory when using dynamic channel
    # cudnn benchmark will lead to bug when computing G regularization, resulting in extremely slow computing
    cudnn.benchmark = (not args.dynamic_channel) and args.g_reg_every <= 0

    assert args.job is not None
    if hvd.rank() == 0:
        print(' * JOB:', args.job)
    # make log dirs
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, args.job), exist_ok=True)
    log_writer = SummaryWriter(os.path.join(log_dir, args.job)) if hvd.rank() == 0 else None

    if hvd.rank() == 0:  # save args
        with open(os.path.join(log_dir, args.job, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    # build dataset
    if args.n_res > 1:
        from utils.datasets import MultiResize, GroupRandomHorizontalFlip, GroupTransformWrapper

        transform = transforms.Compose([  # transforms that can return a pyramid of
            MultiResize(args.resolution, args.n_res),
            GroupRandomHorizontalFlip(),  # same flipping for all images
            GroupTransformWrapper(transforms.ToTensor()),
            GroupTransformWrapper(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(args.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
    dataset = NativeDataset(args.data_path, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # build discriminator
    if args.n_res > 1:
        discriminator = DiscriminatorMultiRes(
            args.resolution,
            channel_multiplier=args.channel_multiplier,
            n_res=args.n_res,
            modulate=args.conditioned_d
        ).to(device)
        all_resolutions = [args.resolution // (2 ** i) for i in range(args.n_res)]
    else:
        assert not args.conditioned_d  # not supported in this mode
        discriminator = Discriminator(args.resolution, channel_multiplier=args.channel_multiplier).to(device)
        all_resolutions = [args.resolution]

    # build generator
    generator = Generator(args.resolution, args.latent_dim, args.n_mlp,
                          channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.resolution, args.latent_dim, args.n_mlp,
                      channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()

    if hvd.rank() == 0:  # measure flops and #param
        print(' * D Params: {:.2f}M'.format(sum([p.numel() for p in discriminator.parameters()]) / 1e6))
        print(' * G Params: {:.2f}M'.format(sum([p.numel() for p in generator.parameters()]) / 1e6))
        try:
            from torchprofile import profile_macs

            generator.eval()
            macs = profile_macs(generator, [torch.rand(1, 512).to(device)])
            print(' * G MACs: {:.2f}G'.format(macs / 1e9))
        except:
            print(' * Profiling failed. Pass.')

    # tune from a previous checkpoint
    if args.tune_from:
        if hvd.rank() == 0:
            print(' * Tuning from {}'.format(args.tune_from))
        # 1. load G
        sd = torch.load(args.tune_from, map_location='cpu')
        # the generator arch is not changed over different settings
        generator.load_state_dict(sd['g_ema'])

        if args.sort_pretrain:
            if hvd.rank() == 0:
                print(' * Sorting the channels of the generator...')
            sort_channel(generator)

        # 2. load D
        if args.n_res > 1:
            if args.n_res > 1 and 'multires' not in args.tune_from:  # for multi-res training stage
                partially_load_d_for_multi_res(discriminator, sd['d'], args.n_res)
            elif args.dynamic_channel and 'adach' not in args.tune_from:  # for adaptive-channel training stage
                partially_load_d_for_ada_ch(discriminator, sd['d'])
            else:
                discriminator.load_state_dict(sd['d'])
        else:
            discriminator.load_state_dict(sd['d'])

    # init g_ema = generators
    accumulate(g_ema, generator, 0)

    # build teacher (if distill.)
    if args.teacher_ckpt is not None:
        if hvd.rank() == 0:
            print(' * Building teacher models...')
        teacher = Generator(args.resolution, args.latent_dim, args.n_mlp,
                            channel_multiplier=args.t_channel_multiplier).to(device)
        teacher_sd = torch.load(args.teacher_ckpt, map_location='cpu')
        teacher.load_state_dict(teacher_sd['g_ema'])
        teacher.eval()
        # perceptual loss
        percept = lpips.LPIPS(net='vgg', verbose=False).to(device)
    else:
        teacher = None

    # build inception model for FID computing
    inception = models.get_pretrained('inception').to(device)
    inception.eval()

    # build optimizer
    g_optim = optim.Adam(generator.parameters(), lr=args.lr, betas=(0., 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0., 0.99))

    g_optim = hvd.DistributedOptimizer(
        g_optim, named_parameters=generator.named_parameters(prefix='generator'),
    )
    d_optim = hvd.DistributedOptimizer(
        d_optim, named_parameters=discriminator.named_parameters(prefix='discriminator'),
    )

    resume_from_epoch = 0
    mean_path_length = 0.  # track mean path len globally
    if args.resume:
        ckpt_path = os.path.join(checkpoint_dir, args.job, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            if hvd.rank() == 0:
                print(" * Resuming from:", ckpt_path)
                sd = torch.load(ckpt_path, map_location='cpu')
                resume_from_epoch = sd['epoch']
                generator.load_state_dict(sd["g"])
                discriminator.load_state_dict(sd["d"])
                g_ema.load_state_dict(sd["g_ema"])

                g_optim.load_state_dict(sd["g_optim"])
                d_optim.load_state_dict(sd["d_optim"])
                best_fid = sd['best_fid']
                mean_path_length = sd['mean_path_length']

    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(generator.state_dict(), root_rank=0)
    hvd.broadcast_parameters(discriminator.state_dict(), root_rank=0)
    hvd.broadcast_parameters(g_ema.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(g_optim, root_rank=0)
    hvd.broadcast_optimizer_state(d_optim, root_rank=0)

    # draw a sample z for visualization
    torch.manual_seed(2020)
    sample_z = torch.randn(args.n_vis_sample, 1, args.latent_dim).float().to(device)

    for i_epoch in range(resume_from_epoch, args.epochs):
        train(i_epoch)
        validate(i_epoch)
