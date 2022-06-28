import os
import argparse
import yaml
import collections
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.model_selection import train_test_split

from src import models, utils


def main(args, config):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    utils.write_flush(device)

    # Create sample and checkpoint directories
    os.makedirs('images/%s' % args.job_number, exist_ok=True)
    os.makedirs('saved_models/%s' % args.job_number, exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()

    input_shape = (3, 256, 256)

    # Initialize generator and discriminator
    G_AB = models.GeneratorResNet(input_shape, config.nb_residuals).to(device)
    G_BA = models.GeneratorResNet(input_shape, config.nb_residuals).to(device)
    D_A = models.Discriminator(input_shape).to(device)
    D_B = models.Discriminator(input_shape).to(device)
    D_ROI_A = models.DiscriminatorROI().to(device)
    D_ROI_B = models.DiscriminatorROI().to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=2e-4, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D_A_ROI = torch.optim.Adam(D_ROI_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D_B_ROI = torch.optim.Adam(D_ROI_B.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=utils.LambdaLR(config.nb_epochs, 0, 12).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=utils.LambdaLR(config.nb_epochs, 0, 12).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=utils.LambdaLR(config.nb_epochs, 0, 12).step
    )
    lr_scheduler_D_A_ROI = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A_ROI, lr_lambda=utils.LambdaLR(config.nb_epochs, 0, 12).step
    )
    lr_scheduler_D_B_ROI = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B_ROI, lr_lambda=utils.LambdaLR(config.nb_epochs, 0, 12).step
    )

    # Buffers of previously generated samples
    fake_A_buffer = utils.ReplayBuffer()
    fake_B_buffer = utils.ReplayBuffer()

    hes_images, hes_dfs_list = utils.load_data(config.hes_dir, config.hes_library)
    ihc_images, ihc_dfs_list = utils.load_data(config.ihc_dir, config.ihc_library)

    # Data generators
    hes_images_tr, hes_images_te, hes_bboxes_tr,hes_bboxes_te = train_test_split(
        hes_images, hes_dfs_list, test_size=0.1, random_state=42)

    ihc_images_tr, ihc_images_te, ihc_bboxes_tr, ihc_bboxes_te = train_test_split(
        ihc_images, ihc_dfs_list, test_size=0.1, random_state=42)

    # ----------
    #  Training
    # ----------

    gen_A = utils.data_generator(hes_images_tr, hes_bboxes_tr, nb_batch=config.nb_batch, nb_rois=config.nb_rois)
    gen_B = utils.data_generator(ihc_images_tr, ihc_bboxes_tr, nb_batch=config.nb_batch, nb_rois=config.nb_rois)

    for epoch in range(config.nb_epochs):

        for i in range(config.steps_per_epoch):

            data = next(gen_A)
            real_A, condition_A, bboxes_A = (data[0].to(device),
                                             data[1].to(device),
                                             data[2].to(device))

            data = next(gen_B)
            real_B, condition_B, bboxes_B = (data[0].to(device),
                                             data[1].to(device),
                                             data[2].to(device))

            fake = torch.zeros((config.nb_batch, *D_A.output_shape)).to(device)
            valid = torch.ones((config.nb_batch, *D_A.output_shape)).to(device)

            fake_roi = torch.zeros((config.nb_rois,)).to(device)
            valid_roi = torch.ones((config.nb_rois,)).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # ROI loss
            validity_ROI_A = D_ROI_A(fake_A, condition_B, bboxes_B)
            validity_ROI_B = D_ROI_B(fake_B, condition_A, bboxes_A)

            loss_ROI_A = criterion_GAN(validity_ROI_A, valid_roi)
            loss_ROI_B = criterion_GAN(validity_ROI_B, valid_roi)

            loss_ROI = (loss_ROI_A + loss_ROI_B) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + config.lambda_roi * loss_ROI + config.lambda_cyc * loss_cycle + config.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------------------
            #  Train Discriminator ROI A
            # --------------------------

            optimizer_D_A_ROI.zero_grad()

            roi_outputs = D_ROI_A(real_A, condition_A, bboxes_A)
            real_loss = criterion_GAN(roi_outputs, valid_roi)

            roi_outputs = D_ROI_A(fake_A.detach(), condition_B, bboxes_B)
            fake_loss = criterion_GAN(roi_outputs, fake_roi)

            d_ROI_A_loss = (real_loss + fake_loss) / 2
            d_ROI_A_loss.backward()
            optimizer_D_A_ROI.step()

            # --------------------------
            #  Train Discriminator ROI B
            # --------------------------

            optimizer_D_B_ROI.zero_grad()

            roi_outputs = D_ROI_B(real_B, condition_B, bboxes_B)
            real_loss = criterion_GAN(roi_outputs, valid_roi)

            roi_outputs = D_ROI_B(fake_B.detach(), condition_A, bboxes_A)
            fake_loss = criterion_GAN(roi_outputs, fake_roi)

            d_ROI_B_loss = (real_loss + fake_loss) / 2
            d_ROI_B_loss.backward()
            optimizer_D_B_ROI.step()

            # --------------
            #  Log Progress
            # --------------

            # Print log
            utils.write_flush(
                '\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D_ROI_A loss: %f] [D_ROI_B loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f]'
                % (epoch, config.nb_epochs, i, config.steps_per_epoch, loss_D.item(), d_ROI_A_loss.item(), d_ROI_B_loss.item(), loss_G.item(), loss_GAN.item(), loss_cycle.item(), loss_identity.item()))

            batches_done = epoch * config.steps_per_epoch + i

            if batches_done % 100 == 0:
                utils.sample_images(args.job_number, batches_done, G_AB, G_BA, hes_images_te, hes_bboxes_te, ihc_images_te, ihc_bboxes_te, device)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        lr_scheduler_D_A_ROI.step()
        lr_scheduler_D_B_ROI.step()

        if epoch % 5 == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (args.job_number, epoch))
            torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (args.job_number, epoch))
            torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (args.job_number, epoch))
            torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (args.job_number, epoch))
            torch.save(D_ROI_A.state_dict(), 'saved_models/%s/D_A_ROI_%d.pth' % (args.job_number, epoch))
            torch.save(D_ROI_B.state_dict(), 'saved_models/%s/D_B_ROI_%d.pth' % (args.job_number, epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Region-guided CycleGAN for stain transfer on whole slide images')
    parser.add_argument('job_number', type=int)
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    utils.write_flush(str(args))
    
    with open(args.config, 'r') as fp:
        cfg = yaml.safe_load(fp)

    config = collections.namedtuple('Config', cfg.keys())(*cfg.values())

    main(args, config)
