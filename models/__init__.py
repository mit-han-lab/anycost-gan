from .anycost_gan import Generator
import torch
from torchvision import models
from utils.torch_utils import safe_load_state_dict_from_url

URL_TEMPLATE = 'https://hanlab18.mit.edu/projects/anycost-gan/files/{}_{}.pt'


def load_state_dict_from_url(url, key=None):
    if url.startswith('http'):
        sd = safe_load_state_dict_from_url(url, map_location='cpu', progress=True)
    else:
        sd = torch.load(url, map_location='cpu')
    if key is not None:
        return sd[key]
    return sd


def get_pretrained(model, config=None):
    if model in ['attribute-predictor', 'inception']:
        assert config is None
        url = URL_TEMPLATE.format('attribute', 'predictor')  # not used for inception
    else:
        assert config is not None
        url = URL_TEMPLATE.format(model, config)

    if model == 'generator':
        if config in ['anycost-ffhq-config-f', 'anycost-ffhq-config-f-flexible', 'stylegan2-ffhq-config-f']:
            resolution = 1024
            channel_multiplier = 2
        elif config == 'anycost-car-config-f':
            resolution = 512
            channel_multiplier = 2
        else:
            raise NotImplementedError
        model = Generator(resolution, channel_multiplier=channel_multiplier)
        model.load_state_dict(load_state_dict_from_url(url, 'g_ema'))
        return model
    elif model == 'encoder':
        # NOTICE: the encoders are trained with VGG LPIPS loss to keep consistent with optimization-based projection
        # the numbers in the papers are reported with encoders trained with AlexNet LPIPS loss
        if config in ['anycost-ffhq-config-f', 'anycost-ffhq-config-f-flexible', 'stylegan2-ffhq-config-f']:
            n_style = 18
            style_dim = 512
        else:
            raise NotImplementedError
        from models.encoder import ResNet50Encoder
        model = ResNet50Encoder(n_style=n_style, style_dim=style_dim)
        model.load_state_dict(load_state_dict_from_url(url, 'state_dict'))
        return model
    elif model == 'attribute-predictor':  # attribute predictor is general
        predictor = models.resnet50()
        predictor.fc = torch.nn.Linear(predictor.fc.in_features, 40 * 2)
        predictor.load_state_dict(load_state_dict_from_url(url, 'state_dict'))
        return predictor
    elif model == 'inception':  # inception models
        from thirdparty.inception import InceptionV3
        return InceptionV3([3], normalize_input=False, resize_input=True)
    elif model == 'boundary':
        if config in ['anycost-ffhq-config-f', 'anycost-ffhq-config-f-flexible', 'stylegan2-ffhq-config-f']:
            return load_state_dict_from_url(url)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


