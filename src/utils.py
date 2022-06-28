import os
import sys
import h5py
import random

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader


def write_flush(*text_args, stream=sys.stdout):
    stream.write(', '.join(map(str, text_args)) + '\n')
    stream.flush()
    return

def load_data(data_dir, library_dir):

    patches_dir = os.path.join(data_dir, 'patches/')
    patches_files = os.listdir(patches_dir)

    images = {file_name : h5py.File(os.path.join(patches_dir, file_name))['x'][()]
                  for file_name in sorted(patches_files)}

    dfs = {file_name : pd.read_csv(os.path.join(library_dir, file_name.split('.')[0] + '.csv'), index_col=0)
           for file_name in patches_files}

    imgs = np.vstack([images[key] for key in patches_files])

    dfs_list = []

    for key in patches_files:
        nb_tiles = images[key].shape[0]
        df = dfs[key]
        dfs_list.extend([df[df.tile == tile] for tile in range(nb_tiles)])

    assert imgs.shape[0] == len(dfs_list)
    return imgs, dfs_list

def draw_conditions(bboxes, dim):

    condition = torch.zeros((1, 2, dim, dim))
    noise = torch.zeros((1, 2, dim, dim))

    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax, cls = map(int, bbox)

        ymin = max(0, ymin + 5)
        xmin = max(0, xmin + 5)

        ymax = min(dim, ymax - 5)
        xmax = min(dim, xmax - 5)

        condition[0, cls, ymin:ymax, xmin:xmax] = 1

        z = torch.randn(1, 1, ymax-ymin, xmax-xmin)
        noise[0, cls, ymin:ymax, xmin:xmax] = z

    return condition, noise

def sample_bboxes(df_bboxes, nb_samples):

    """
        For a purely negative tile take 3/4 * nb_samples
        For a purely positive tile take 5/4 * nb_samples
        For a mixed tile, take 1/4 * nb_samples positives and nb_samples positives

        e.g. For nb_samples = 8:
        Take 6 pos if positive
        Take 10 neg if negative
        Take 2 pos, 8 neg if mixed

        This guarantees 64 rois are taken, and, because positive and negative tiles
        are balanced, the roi classes are roughly balanced also.

    """

    df_neg = df_bboxes[df_bboxes['class'] == 0]
    df_pos = df_bboxes[df_bboxes['class'] == 1]

    if df_pos.shape[0] == 0:  # purely negative tile
        df_sample = df_neg.sample(3 * (nb_samples // 4), replace=True)

    elif df_neg.shape[0] == 0:  # purely positive tile
        df_sample = df_pos.sample(5 * (nb_samples // 4), replace=True)

    else:  # mixed tile
        df_sample = pd.concat([df_pos.sample(nb_samples, replace=True),
                               df_neg.sample(nb_samples // 4, replace=True)])

    return df_sample[['xmin', 'ymin', 'xmax', 'ymax', 'class']].values

def data_augmentation(x_batch, bbox_batch, img_dim):

    if np.random.randn() > 0.5:

        x_batch = x_batch.flip(dims=(3,))

        left = bbox_batch[:, 1].copy()
        right = bbox_batch[:, 3].copy()

        bbox_batch[:, 1] = img_dim - right
        bbox_batch[:, 3] = img_dim - left

    if np.random.randn() > 0.5:

        x_batch = x_batch.flip(dims=(2,))

        top = bbox_batch[:, 2].copy()
        bottom = bbox_batch[:, 4].copy()

        bbox_batch[:, 2] = img_dim - bottom
        bbox_batch[:, 4] = img_dim - top

    return x_batch, bbox_batch

def data_generator(imgs, bboxes, nb_batch, nb_rois=64):

    idx_non_empty = [idx for idx, df in enumerate(bboxes) if not df.empty]
    idx_pos = [idx for idx in idx_non_empty if 1 in bboxes[idx]['class'].values]
    idx_neg = [idx for idx in idx_non_empty if not 1 in bboxes[idx]['class'].values]
    
    img_dim = imgs.shape[2]

    while True:

        idx_pos_batch = np.random.choice(idx_pos, size=nb_batch // 2)
        idx_neg_batch = np.random.choice(idx_neg, size=nb_batch // 2)

        x_batch = np.vstack([imgs[idx_pos_batch], imgs[idx_neg_batch]])
        x_batch = torch.Tensor(np.moveaxis(x_batch, 3, 1) / 127.5 - 1)

        nb_samples = nb_rois // nb_batch
        df_bbox_batch = [bboxes[i] for i in list(idx_pos_batch) + list(idx_neg_batch)]
        bbox_data = [sample_bboxes(df_bbox, nb_samples) for df_bbox in df_bbox_batch]
        bbox_batch = np.vstack([np.hstack([i * np.ones((bboxes.shape[0], 1)), bboxes])
                                for i, bboxes in enumerate(bbox_data)])

        x_batch, bbox_batch = data_augmentation(x_batch, bbox_batch, img_dim)

        condition_batch = []

        for i in range(nb_batch):

            rois = bbox_batch[bbox_batch[:, 0]==i]
            condition, noise = draw_conditions(rois[:, 1:], img_dim)
            condition_batch.append(condition)

        condition_batch = torch.cat(condition_batch, axis=0)

        yield torch.Tensor(x_batch), condition_batch, torch.Tensor(bbox_batch)

"""
N.B. There is generally on a few hundred samples in the val/test data.
Hence, drawing 25 samples leads to duplicates with high probability
(see the birthday paradox). Furthermore, the duplicates will always
be consecutive in the batch, as the indices are sorted in the data_generator
function.
"""

def sample_images(output_dir, batches_done, G_AB, G_BA, hes_images_te, hes_bboxes_te, ihc_images_te, ihc_bboxes_te, device):
    """Saves a generated sample from the test set"""
#    imgs = next(iter(val_dataloader))

    gen_A = data_generator(hes_images_te, hes_bboxes_te, nb_batch=25)
    gen_B = data_generator(ihc_images_te, ihc_bboxes_te, nb_batch=25)

    real_A = next(gen_A)[0].to(device)
    real_B = next(gen_B)[0].to(device)
    G_AB.eval()
    G_BA.eval()
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)
    # Arrange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (output_dir, batches_done), normalize=False)


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
