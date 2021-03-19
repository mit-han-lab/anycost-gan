# get edit directions for FFHQ models
import sys
sys.path.append('.')  # to run from the project root dir
import torch
import torch.nn.functional as F
import models
from tqdm import tqdm

# configurations for the job
device = 'cuda'
# specify the attributes to compute latent direction
chosen_attr = ['Smiling', 'Young', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Mustache']
attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
             'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
             'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
             'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
             'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
             'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
space = 'w'  # chosen from ['z', 'w', 'w+']
config = 'anycost-ffhq-config-f'


@torch.no_grad()
def get_style_attribute_pairs():  # this function is written with horovod to accelerate the extraction (by n_gpu times)
    import horovod.torch as hvd
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.manual_seed(hvd.rank() * 999 + 1)
    if hvd.rank() == 0:
        print(' * Extracting style-attribute pairs...')
    # build and load the pre-trained attribute predictor on CelebA-HQ
    predictor = models.get_pretrained('attribute-predictor').to(device)
    # build and load the pre-trained anycost generator
    generator = models.get_pretrained('generator', config).to(device)

    predictor.eval()
    generator.eval()

    # randomly generate images and feed them to the predictor
    # configs from https://github.com/genforce/interfacegan
    randomized_noise = False
    truncation_psi = 0.7
    batch_size = 16
    n_batch = 500000 // (batch_size * hvd.size())

    styles = []
    attributes = []

    mean_style = generator.mean_style(100000).view(1, 1, -1)
    assert space in ['w', 'w+', 'z']
    for _ in tqdm(range(n_batch), disable=hvd.rank() != 0):
        if space in ['w', 'z']:
            z = torch.randn(batch_size, 1, generator.style_dim, device=device)
        else:
            z = torch.randn(batch_size, generator.n_style, generator.style_dim, device=device)
        images, w = generator(z,
                              return_styles=True,
                              truncation=truncation_psi,
                              truncation_style=mean_style,
                              input_is_style=False,
                              randomize_noise=randomized_noise)
        images = F.interpolate(images.clamp(-1, 1), size=256, mode='bilinear', align_corners=True)
        attr = predictor(images)
        # move to cpu to save memory
        if space == 'w+':
            styles.append(w.to('cpu'))
        elif space == 'w':
            styles.append(w.mean(1, keepdim=True).to('cpu'))  # originally duplicated
        else:
            styles.append(z.to('cpu'))
        attributes.append(attr.to('cpu'))

    styles = torch.cat(styles, dim=0)
    attributes = torch.cat(attributes, dim=0)

    styles = hvd.allgather(styles, name='styles')
    attributes = hvd.allgather(attributes, name='attributes')
    if hvd.rank() == 0:
        print(styles.shape, attributes.shape)
        torch.save(attributes, 'attributes_{}.pt'.format(config))
        torch.save(styles, 'styles_{}.pt'.format(config))


def extract_boundaries():
    styles = torch.load('styles_{}.pt'.format(config))
    attributes = torch.load('attributes_{}.pt'.format(config))
    attributes = attributes.view(-1, 40, 2)
    prob = F.softmax(attributes, dim=-1)[:, :, 1]  # probability to be positive [n, 40]

    boundaries = {}
    for idx, attr in tqdm(enumerate(attr_list), total=len(attr_list)):
        this_prob = prob[:, idx]

        from thirdparty.manipulator import train_boundary
        boundary = train_boundary(latent_codes=styles.squeeze().cpu().numpy(),
                                  scores=this_prob.view(-1, 1).cpu().numpy(),
                                  chosen_num_or_ratio=0.02,
                                  split_ratio=0.7,
                                  )
        key_name = '{:02d}'.format(idx) + '_' + attr
        boundaries[key_name] = boundary

    boundaries = {k: torch.tensor(v) for k, v in boundaries.items()}
    torch.save(boundaries, 'boundaries_{}.pt'.format(config))


# experimental; not yet used in the demo
# do not observe significant improvement right now
def project_boundaries():  # only project the ones used for demo
    from thirdparty.manipulator import project_boundary
    import numpy as np
    boundaries = torch.load('boundaries_{}.pt'.format(config))
    chosen_idx = [attr_list.index(attr) for attr in chosen_attr]
    sorted_keys = ['{:02d}'.format(idx) + '_' + attr_list[idx] for idx in chosen_idx]
    all_boundaries = np.concatenate([boundaries[k].cpu().numpy() for k in sorted_keys])  # n, 512
    similarity = all_boundaries @ all_boundaries.T
    projected_boundaries = []
    for i_b in range(len(sorted_keys)):
        # NOTE: the number of conditions is exponential;
        # we only take the 2 most related boundaries
        this_sim = similarity[i_b]
        this_sim[i_b] = -100.  # exclude self
        idx1, idx2 = np.argsort(this_sim)[-2:]  # most similar 2
        projected_boundaries.append(project_boundary(all_boundaries[i_b][None],
                                                     all_boundaries[idx1][None], all_boundaries[idx2][None]))
    boundaries = {k: v for k, v in zip(sorted_keys, torch.tensor(projected_boundaries))}
    torch.save(boundaries, 'boundary_projected_{}.pt'.format(config))


if __name__ == '__main__':
    # get_style_attribute_pairs()
    extract_boundaries()
    # project_boundaries()
