import sys
sys.path.append('.')  # to run from the project root dir

import argparse
import math
import torch
from tqdm import tqdm
import models
import horovod.torch as hvd
from utils.torch_utils import adaptive_resize


def compute_attribute_consistency(g, sub_g, n_sample, batch_size):
    attr_pred = models.get_pretrained('attribute-predictor').to(device)
    attr_pred.eval()

    n_batch = math.ceil(n_sample * 1. / batch_size / hvd.size())

    accs = 0
    mean_style = g.mean_style(10000)
    with torch.no_grad():
        for _ in tqdm(range(n_batch), disable=hvd.rank() != 0):
            noise = g.make_noise()

            latent = torch.randn(args.batch_size, 1, 512, device=device)
            kwargs = {'styles': latent, 'truncation': 0.5, 'truncation_style': mean_style, 'noise': noise}
            img = g(**kwargs)[0].clamp(min=-1., max=1.)
            sub_img = sub_g(**kwargs)[0].clamp(min=-1., max=1.)
            img = adaptive_resize(img, 256)
            sub_img = adaptive_resize(sub_img, 256)

            attr = attr_pred(img).view(-1, 40, 2)
            sub_attr = attr_pred(sub_img).view(-1, 40, 2)

            attr = torch.argmax(attr, dim=2)
            sub_attr = torch.argmax(sub_attr, dim=2)
            this_acc = (attr == sub_attr).float().mean(0)
            accs = accs + this_acc / n_batch
    accs = hvd.allreduce(accs)
    return accs


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Computing attribute consistency between generators")
    parser.add_argument("--config", type=str, help='config name of the pretrained generator')
    parser.add_argument('--channel_ratio', type=float, default=None, help='channel ratio for the sub-generator')
    parser.add_argument('--target_res', type=int, default=None, help='resolution used for the sub-generator')

    parser.add_argument("--n_sample", type=int, default=10000, help="number of the samples for calculating PPL")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for the models (per gpu)")

    args = parser.parse_args()

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    generator = models.get_pretrained('generator', args.config).to(device).eval()

    sub_generator = models.get_pretrained('generator', args.config).to(device).eval()
    if args.channel_ratio:
        from models.dynamic_channel import set_uniform_channel_ratio
        set_uniform_channel_ratio(sub_generator, args.channel_ratio)

    if args.target_res is not None:
        sub_generator.target_res = args.target_res

    acc_list = compute_attribute_consistency(generator, sub_generator,
                                             n_sample=args.n_sample,
                                             batch_size=args.batch_size)
    acc_list = list(acc_list.to('cpu').numpy())
    attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    if hvd.rank() == 0:
        for at, ac in zip(attr_list, acc_list):
            print('{:30s}{:.2f}%'.format(at, ac * 100))
