import sys

sys.path.append('.')  # to run from the project root dir

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import models
from utils.torch_utils import adaptive_resize
from thirdparty.celeba_hq_split import get_celeba_hq_split
from torchvision import transforms
import lpips
from utils.torch_utils import AverageMeter
from models.dynamic_channel import set_uniform_channel_ratio, remove_sub_channel_config
from utils.datasets import NativeDataset


def validate():
    generator.eval()
    encoder.eval()

    # loss meters
    lpips_loss_meter = AverageMeter()
    mse_loss_meter = AverageMeter()
    sub_lpips_loss_meter = AverageMeter()
    sub_mse_loss_meter = AverageMeter()
    consist_lpips_loss_meter = AverageMeter()
    consis_mse_loss_meter = AverageMeter()

    with tqdm(total=len(test_loader), desc='Testing') as t:
        with torch.no_grad():
            for real_img in test_loader:
                real_img = real_img.to(device)
                real_img = adaptive_resize(real_img, 256)

                pred_w = encoder(real_img)

                fake_image, _ = generator(pred_w, noise=None, randomize_noise=False, input_is_style=True)
                fake_image = adaptive_resize(fake_image.clamp(-1, 1), 256)

                lpips_loss = lpips_func(fake_image, real_img).mean()
                mse_loss = nn.MSELoss()(fake_image, real_img)

                lpips_loss_meter.update(lpips_loss.item(), real_img.shape[0])
                mse_loss_meter.update(mse_loss.item(), real_img.shape[0])

                if args.calc_consist:
                    # compute the consistency loss between full generator and sub-generators (3 configs)
                    for ch_ratio in [0.25, 0.5, 0.75]:
                        set_uniform_channel_ratio(generator, ch_ratio)
                        fake_sub_image, _ = generator(pred_w, noise=None, randomize_noise=False, input_is_style=True)
                        fake_sub_image = adaptive_resize(fake_sub_image.clamp(-1, 1), 256)
                        remove_sub_channel_config(generator)
                        consist_lpips_loss_meter.update(lpips_func(fake_sub_image, fake_image).mean().item(),
                                                        real_img.shape[0])
                        consis_mse_loss_meter.update(nn.MSELoss()(fake_sub_image, fake_image).item(), real_img.shape[0])

                        sub_lpips_loss_meter.update(lpips_func(fake_sub_image, real_img).mean().item(),
                                                    real_img.shape[0])
                        sub_mse_loss_meter.update(nn.MSELoss()(fake_sub_image, real_img).item(), real_img.shape[0])

                info2display = {
                    'lpips': lpips_loss_meter.avg,
                    'mse': mse_loss_meter.avg,
                    'sub-lpips': sub_lpips_loss_meter.avg,
                    'sub-mse': sub_mse_loss_meter.avg,
                    'consist-lpips': consist_lpips_loss_meter.avg,
                    'consist-mse': consis_mse_loss_meter.avg
                }

                t.set_postfix(info2display)
                t.update(1)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Computing attribute consistency between generators")
    parser.add_argument("--config", type=str, help='config name of the pretrained generator')
    # dataset
    parser.add_argument("--data_path", type=str, default='/dataset/ffhq/celeba-hq/')
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for the models (per gpu)")
    parser.add_argument('-j', '--workers', default=4, type=int)

    parser.add_argument("--lpips_net", type=str, default='vgg', choices=['vgg', 'alex'])

    parser.add_argument('--calc_consist', action='store_true', default=False, help='compute the consistency loss')
    args = parser.parse_args()

    # build models
    generator = models.get_pretrained('generator', args.config).to(device)
    generator.eval()

    encoder = models.get_pretrained('encoder', args.config).to(device)
    encoder.eval()

    # build test dataset
    val_transform = transforms.Compose([
        transforms.Resize(generator.resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    test_dataset = NativeDataset(args.data_path, transform=val_transform)
    train_idx, val_idx, test_idx = get_celeba_hq_split()
    test_dataset = torch.utils.data.Subset(test_dataset, test_idx)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    # build lpips loss
    lpips_func = lpips.LPIPS(net=args.lpips_net, verbose=False).to(device)
    lpips_func.eval()

    validate()
