import argparse
import pickle
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import math

import sys

sys.path.append('.')  # to run from the project root dir
import models
from utils.datasets import NativeDataset


def extract_features():
    feature_list = []
    n_iter = math.ceil(args.n_sample * 1. / args.batch_size)
    assert n_iter <= len(loader)
    with torch.no_grad():
        for img in tqdm(loader, total=n_iter):
            if len(feature_list) * args.batch_size >= args.n_sample:
                break
            img = img.to(device)
            feature = inception(img)[0].view(img.shape[0], -1)
            feature_list.append(feature.to('cpu'))
    return torch.cat(feature_list, 0)[: args.n_sample]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='Calculate Inception v3 features for datasets')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('-j', '--workers', default=32, type=int)
    parser.add_argument('--save_name', default=None, type=str)
    parser.add_argument('path', metavar='PATH', help='path to dataset (image version)')

    args = parser.parse_args()

    inception = models.get_pretrained('inception').to(device).eval()
    inception = nn.DataParallel(inception)

    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = NativeDataset(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)

    features = extract_features().numpy()

    print(f'extracted {features.shape[0]} features')

    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    name = os.path.splitext(os.path.basename(args.path))[0]

    print(' * Saving to', args.save_name)
    with open(args.save_name, 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'size': args.resolution, 'path': args.path}, f)
