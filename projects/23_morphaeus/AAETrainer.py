from torch.autograd import Variable
import torch.utils.data
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from time import time
import wandb
import logging

from core.Trainer import Trainer
from dl_utils.config_utils import *
import matplotlib.pyplot as plt

import copy

# S. Pidhorskyi, R. Almohsen, and G. Doretto. Generative probabilistic novelty detection with adversarial autoencoders.
# Advances in Neural Information Processing Systems, volume 31.


class PTrainer(Trainer):
    """
    code adapted from https://github.com/podgorskiy/GPND

    License:
        Apache License 2.0
    """
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)

        lr = training_params['optimizer_params']['lr']
        self.cross_batch = model.cross_batch
        self.latent_size = model.z_dim

        self.G_optimizer = optim.Adam(model.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(model.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.GE_optimizer = optim.Adam(list(model.E.parameters()) + list(model.G.parameters()), lr=lr, betas=(0.5, 0.999))
        self.ZD_optimizer = optim.Adam(model.ZD.parameters(), lr=lr, betas=(0.5, 0.999))

        self.BCE_loss = nn.BCELoss()

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        if model_state is not None:
            self.model.load_state_dict(model_state)  # load weights
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        epoch_losses_D, epoch_losses_G, epoch_losses_ZD, epoch_losses_E, epoch_losses_GE = [], [], [], [], []
        epoch_losses_pl = []
        self.early_stop = False
        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss_G, batch_loss_D, batch_loss_E, batch_loss_GE, batch_loss_ZD, batch_loss_pl, count_images = \
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

            self.model.G.train()
            self.model.D.train()
            self.model.E.train()
            self.model.ZD.train()

            if (epoch + 1) % 90 == 0:
                self.G_optimizer.param_groups[0]['lr'] /= 4
                self.D_optimizer.param_groups[0]['lr'] /= 4
                self.GE_optimizer.param_groups[0]['lr'] /= 4
                self.ZD_optimizer.param_groups[0]['lr'] /= 4
                print("learning rate change!")

            for data in self.train_ds:
                # Input

                images = data[0].to(self.device)
                transformed_images = self.transform(images) if self.transform is not None else images
                b, c, w, h = images.shape
                count_images += b

                y_real_ = torch.ones((transformed_images.shape[0])).to(self.device)
                y_fake_ = torch.zeros((transformed_images.shape[0])).to(self.device)

                y_real_z = torch.ones(1 if self.cross_batch else transformed_images.shape[0]).to(self.device)
                y_fake_z = torch.zeros(1 if self.cross_batch else transformed_images.shape[0]).to(self.device)

                #############################################

                self.model.D.zero_grad()

                D_result = self.model.D(transformed_images).squeeze()
                D_real_loss = self.BCE_loss(D_result, y_real_)

                z = torch.randn((transformed_images.shape[0], self.latent_size)).view(-1, self.latent_size, 1, 1).to(self.device)
                z = Variable(z)

                x_fake = self.model.G(z).detach()
                D_result = self.model.D(x_fake).squeeze()

                D_fake_loss = self.BCE_loss(D_result, y_fake_)

                D_train_loss = D_real_loss + D_fake_loss
                D_train_loss.backward()

                batch_loss_D += D_train_loss.item() * images.size(0)

                self.D_optimizer.step()

                #############################################

                self.model.G.zero_grad()

                z = torch.randn((transformed_images.shape[0], self.latent_size)).view(-1, self.latent_size, 1, 1).to(self.device)
                z = Variable(z)

                x_fake = self.model.G(z)
                D_result = self.model.D(x_fake).squeeze()

                G_train_loss = self.BCE_loss(D_result, y_real_)

                batch_loss_G += G_train_loss.item() * images.size(0)

                G_train_loss.backward()
                self.G_optimizer.step()

                #############################################

                self.model.ZD.zero_grad()

                z = torch.randn((transformed_images.shape[0], self.latent_size)).view(-1, self.latent_size).to(self.device)
                z = z.requires_grad_(True)

                ZD_result = self.model.ZD(z).squeeze()
                ZD_real_loss = self.BCE_loss(ZD_result, y_real_z)

                z = self.model.E(transformed_images).squeeze().detach()

                ZD_result = self.model.ZD(z).squeeze()
                ZD_fake_loss = self.BCE_loss(ZD_result, y_fake_z)

                ZD_train_loss = (ZD_real_loss + ZD_fake_loss) * 2.0
                batch_loss_ZD += ZD_train_loss.item() * images.size(0)

                ZD_train_loss.backward()

                self.ZD_optimizer.step()

                # #############################################

                self.model.E.zero_grad()
                self.model.G.zero_grad()

                z = self.model.E(transformed_images)
                x_d = self.model.G(z)

                ZD_result = self.model.ZD(z.squeeze()).squeeze()

                E_train_loss = self.BCE_loss(ZD_result, y_real_z) * 1.0

                Recon_loss = F.binary_cross_entropy(x_d, images.detach()) * 2.0 # 10.0

                (Recon_loss + E_train_loss).backward()

                self.GE_optimizer.step()

                loss_pl = self.criterion_PL(x_d, images.detach())
                batch_loss_pl += loss_pl.item() * images.size(0)
                batch_loss_GE += Recon_loss.item() * images.size(0)
                batch_loss_E += E_train_loss.item() * images.size(0)

            epoch_loss_D = batch_loss_D / count_images if count_images > 0 else batch_loss_D
            epoch_loss_G = batch_loss_G / count_images if count_images > 0 else batch_loss_G
            epoch_loss_ZD = batch_loss_ZD / count_images if count_images > 0 else batch_loss_ZD
            epoch_loss_E = batch_loss_E / count_images if count_images > 0 else batch_loss_E
            epoch_loss_GE = batch_loss_GE / count_images if count_images > 0 else batch_loss_GE
            epoch_loss_pl = batch_loss_pl / count_images if count_images > 0 else batch_loss_pl
            epoch_losses_D.append(epoch_loss_D)
            epoch_losses_G.append(epoch_loss_G)
            epoch_losses_ZD.append(epoch_loss_ZD)
            epoch_losses_E.append(epoch_loss_E)
            epoch_losses_GE.append(epoch_loss_GE)

            epoch_losses_pl.append(epoch_loss_pl)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss_GE, end_time - start_time, count_images))
            wandb.log({"Train/Loss_D_": epoch_loss_D, '_step_': epoch})
            wandb.log({"Train/Loss_G_": epoch_loss_G, '_step_': epoch})
            wandb.log({"Train/Loss_ZD_": epoch_loss_ZD, '_step_': epoch})
            wandb.log({"Train/Loss_E_": epoch_loss_E, '_step_': epoch})
            wandb.log({"Train/Loss_GE_": epoch_loss_GE, '_step_': epoch})
            wandb.log({"Train/Loss_PL_": epoch_loss_pl, '_step_': epoch})
            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_E_weights': self.GE_optimizer.state_dict(),
                        'optimizer_G_weights': self.G_optimizer.state_dict(),
                        'optimizer_ZD_weights': self.ZD_optimizer.state_dict(),
                        'optimizer_D_weights': self.D_optimizer.state_dict(), 'epoch': epoch},
                       self.client_path + '/latest_model.pt')

            # Run validation
            self.test(self.model.state_dict(),
                      self.val_ds, 'Val', opt_weights=[self.GE_optimizer.state_dict(),
                                                       self.G_optimizer.state_dict(),
                                                       self.ZD_optimizer.state_dict(),
                                                       self.D_optimizer.state_dict()], epoch=epoch)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        self.test_model.load_state_dict(model_weights)
        self.test_model.eval()
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        test_total = 0
        with torch.no_grad():
            for data in test_data:
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)

                x_, z_dict = self.test_model(x)
                # Forward pass
                loss_rec = F.binary_cross_entropy(x_, x.detach()) * 2.0
                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

        img = x.detach().cpu()[0].numpy()
        rec = x_.detach().cpu()[0].numpy()

        elements = [img, rec, np.abs(rec - img)]
        v_maxs = [1, 1, 0.5]
        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        diffp.set_size_inches(len(elements) * 4, 4)
        for i in range(len(axarr)):
            axarr[i].axis('off')
            v_max = v_maxs[i]
            c_map = 'gray' if v_max == 1 else 'inferno'
            axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=0, vmax=v_max, cmap=c_map)

        wandb.log({task + '/Example_': [
            wandb.Image(diffp, caption="Iteration_" + str(epoch))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        epoch_val_loss = metrics[task + '_loss_rec'] / test_total

        if epoch_val_loss < self.min_val_loss and task == 'Val':
            self.min_val_loss = epoch_val_loss
            self.best_weights = copy.deepcopy(model_weights)
            self.best_opt_weights = copy.deepcopy(opt_weights)
            torch.save({'model_weights': model_weights, 'optimizer_E_weights': opt_weights[0],
                        'optimizer_G_weights': opt_weights[1],
                        'optimizer_ZD_weights': opt_weights[2],
                        'optimizer_D_weights': opt_weights[3], 'epoch': epoch},
                       self.client_path + '/best_model.pt')
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)